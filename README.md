# SCOA / Enhanced YOLOX

面向药品泡罩图像的药片与缺陷腔体检测项目，提供结构约束光学增强（SCOA）、EMA 候选难度选择，以及 YOLOX-s 的训练、评价和推理工具。

相关论文：*Structure-Constrained Optical Augmentation for Pharmaceutical Blister Defect Detection*。

## 功能

- 在共享的有限光学候选池中生成局部模糊／锐化与眩光扰动。
- 结合光度保护和实例级结构筛查，保留满足约束的增强候选。
- 使用 EMA 检测器计算候选难度，并在图像中出现的类别间进行平衡汇总。
- 提供 YOLOX-s 基线、SCOA 核心配置，以及包含 CBAM、WIoU-v3 和 DIoU-NMS 的集成配置。
- 支持 COCO 格式标注、断点续训、固定图像子集评价和单图推理。

SCOA 在训练阶段选择输入图像；推理阶段不执行光学候选生成和筛选。

## 目录

```text
configs/               训练与评价配置
scoa/                  数据处理、光学增强、筛查、选择和检测器实现
tools/                 数据检查、结果汇总和计时等工具
tests/                 自动化检查
weights/               模型文件及简要使用说明
third_party/           YOLOX 来源记录与许可证
docs/                  方法与实现说明
validation_records/    已有软件检查记录
requirements*.txt      环境依赖
NOTICE.md              来源与第三方说明
```

## 安装

以下命令在项目根目录执行。先获取项目：

```powershell
git clone https://github.com/nanxing2138/SCOA-Blister-Detection.git
cd SCOA-Blister-Detection
```

GPU 参考依赖为 Python 3.9、PyTorch 1.12.1 和 CUDA 11.6，需配合兼容的 NVIDIA 驱动。Windows 下可创建独立环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-torch-reference.txt
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

下文的 `python` 指已安装依赖的环境中的 Python。Windows 可运行 `.venv\Scripts\Activate.ps1` 激活环境，或将命令中的 `python` 替换为 `.venv\Scripts\python.exe`。Linux 下使用 `.venv/bin/python`。

若使用 CPU，在独立环境中改用以下依赖组合，不与上面的 CUDA 组合叠加安装：

```powershell
python -m venv .venv-cpu
.venv-cpu\Scripts\python.exe -m pip install -r requirements-cpu.txt
.venv-cpu\Scripts\python.exe -m pip install -r requirements.txt
.venv-cpu\Scripts\python.exe -m pip install -e .
```

CPU 依赖固定为 PyTorch 2.2.2 / torchvision 0.17.2，支持的 Python 范围见 [pyproject.toml](pyproject.toml)。安装后可检查环境和源码：

```powershell
python tools/verify_vendor.py
python -m pytest -q
```

## 数据准备

数据入口使用 COCO JSON 标注，默认目录为：

```text
data/blister/
  images/
  annotations/
    train.json
    val.json
    test.json
```

在 [configs/paper.yaml](configs/paper.yaml) 中设置 `data.root`、`image_dir` 及各划分的标注路径。图像的 `file_name` 相对于 `data.root/image_dir`。数据文件需另行准备，未包含在代码仓库中。

类别映射固定如下；网络内部类别索引为对应 ID 减 1：

| COCO category_id | 类别 |
|---|---|
| 1 | medicine |
| 2 | empty |
| 3 | broke |
| 4 | lack |

每个图像条目包含 `id`、`file_name`、`width` 和 `height`；每个标注条目包含 `id`、`image_id`、`category_id` 和原图像素坐标的 `bbox: [x, y, width, height]`。`iscrowd` 和 `ignore` 省略或置为 0。`categories` 使用上表中的四组 ID 和名称。

为避免数据泄漏，应将同一物理泡罩及其相邻帧／多视角图像放在同一划分；可在图像条目中增加 `entity_id` 供检查工具使用。完成配置后执行：

```powershell
python tools/check_dataset.py --config configs/paper.yaml
```

工具检查图像解码、尺寸、类别、框、精确重复图像，以及已登记实体 ID 的跨划分重复情况。近重复图像和相邻帧仍需结合采集信息检查。

## 模型文件

仓库提供 [weights/yolox_s.pth](weights/yolox_s.pth)，文件信息、校验值和加载方式见 [weights/README.md](weights/README.md)。默认训练配置通过 `train.pretrained` 读取此文件。

## 训练

| 配置 | 内容 | 默认输出 |
|---|---|---|
| [configs/basic.yaml](configs/basic.yaml) | YOLOX-s 与基本增强 | `runs/basic/` |
| [configs/paper.yaml](configs/paper.yaml) | SCOA 核心配置 D | `runs/scoa/` |
| [configs/integrated.yaml](configs/integrated.yaml) | SCOA + CBAM + WIoU-v3，评价使用 DIoU-NMS | `runs/integrated/` |

默认设置为固定 640 × 640 输入、300 轮、batch size 4、随机种子 42。具体增强与优化参数以配置文件为准。

运行基线：

```powershell
python -m scoa.train --config configs/basic.yaml --device cuda
```

运行 SCOA 核心配置，并指定输出目录：

```powershell
python -m scoa.train --config configs/paper.yaml --device cuda --set name=D --set output=runs/D
```

运行集成配置：

```powershell
python -m scoa.train --config configs/integrated.yaml --device cuda
```

通过重复的 `--set key=value` 参数覆盖配置，例如 `--set train.seed=3407`。CPU 运行将 `--device cuda` 改为 `--device cpu`；需要时增加 `--threads 2`。

核心候选池对照可通过 `optical.policy` 选择：

| 策略 | 光度保护 | 结构筛查 | 可行候选选择 |
|---|---|---|---|
| A | 启用 | 关闭 | 均匀采样 |
| B | 启用 | 启用 | 均匀采样 |
| C | 启用 | 关闭 | EMA 难度选择 |
| D | 启用 | 启用 | EMA 难度选择 |

C / D 在第 1–14 轮均匀采样，第 15 轮开始按难度选择。无可行候选时使用保持标签不变的预处理原图；难度评分无效时在非空可行池中均匀采样。

每个输出目录下按 `seed_<种子>` 保存：

- `resolved_config.yaml`、`run_metadata.json`：本次运行配置与元信息。
- `epochs.jsonl`、`candidates.jsonl`：逐轮指标与候选选择记录。
- `latest.pth`：最近一轮的恢复检查点。
- `best.pth`：按验证集未舍入 AP50 选择的 EMA 检查点；同分保留更早轮次。

继续已有训练时使用同一配置和覆盖项：

```powershell
python -m scoa.train --config configs/paper.yaml --device cuda --set name=D --set output=runs/D --resume runs/D/seed_42/latest.pth
```

保留该目录中的 `best.pth`。恢复从轮次边界继续；新实验请使用独立输出目录。

## 评价

以下命令使用上述训练产生的四类检测检查点。评价配置的模型结构和输入尺寸须与选中检查点一致。

SCOA 核心配置：

```powershell
python -m scoa.evaluate --config configs/paper.yaml --checkpoint runs/D/seed_42/best.pth --device cuda --set name=D --output runs/D/seed_42/test.json
```

集成配置：

```powershell
python -m scoa.evaluate --config configs/integrated.yaml --checkpoint runs/integrated/seed_42/best.pth --device cuda --output runs/integrated/seed_42/test_diou.json
```

默认评价测试集；通过 `--split val` 改为验证集，或通过 `--ids path/to/image_ids.json` 指定图像 ID 列表。默认置信度过滤阈值为 0.001，NMS 阈值为 0.45，最多保留 100 个检测结果；P/R 工作点阈值为 0.25。参数可在 `evaluation` 配置中调整。

评价输出指标 JSON 和同名 `.predictions.json` 文件，并记录选中轮次、检查点与标注文件校验值、评价图像 ID 和后处理设置。

## 单图推理

```powershell
python -m scoa.infer --checkpoint runs/integrated/seed_42/best.pth --image path/to/image.png --device cuda --output runs/predict
```

输出 `runs/predict/predictions.json` 和 `runs/predict/prediction.png`。可用 `--score 0.25` 设置置信度阈值，用 `--nms standard` 或 `--nms diou` 指定后处理。

## 使用说明与反馈

- 修改数据集路径后，训练和评价应使用一致的路径及类别映射。
- 比较不同策略时，应固定数据划分、随机种子、模型配置和评价设置。
- 长时间训练的候选日志可能较大，运行前请预留输出空间。
- 提交问题时请附运行命令、配置、Python / PyTorch 版本及错误信息，可通过 [Issues](https://github.com/nanxing2138/SCOA-Blister-Detection/issues) 反馈。

## 来源与许可

YOLOX 模型代码基于 [Megvii-BaseDetection/YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)。第三方许可证见 [third_party/YOLOX/LICENSE](third_party/YOLOX/LICENSE)，来源与适配记录见 [NOTICE.md](NOTICE.md)、[third_party/upstream.json](third_party/upstream.json) 和 [third_party/model_file_sha256.json](third_party/model_file_sha256.json)。
