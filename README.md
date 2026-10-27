# SCOA / Enhanced YOLOX 论文驱动复现

这是对上传论文 **Structure-Constrained Optical Augmentation for Pharmaceutical Blister Defect Detection** 的可移植 Python/PyTorch 重建。先读 [论文理解与缺失信息](docs/01_paper_and_plan.md)，再读 [论文—代码映射](docs/02_paper_code_map.md)。最终实际运行情况见 [验证记录](docs/03_validation.md)。

现已补充[官方预训练权重与接入验证](docs/05_pretrained_weights.md)。本仓库公开源码不包含预训练权重；运行 `python tools/download_weights.py --model s` 可取得官方COCO初始化文件。

**本仓库公开依据论文重建的参考实现；真实数据、固定划分及原始实验记录未随仓库分发。这里不是产生论文原有表格结果的原始研究代码。所有论文表格数值只作参考，未运行的真实结果保持 null。**

以下命令在项目根目录执行。本次完成44项测试、完整合成闭环、真实COCO权重迁移，以及集成模型按epoch保存/续训的精确状态对照。详细修复与完整性边界见 [代码完整性检查报告](docs/06_completeness_audit.md)。历史验证记录保留，最新状态以这份报告为准；下文命令不依赖本机盘符。

## 阅读顺序

- 了解算法：先读下方“论文任务与方法”“关键公式与实现对应”。
- 换一台电脑运行：从“1. 在另一台机器安装”开始，再执行合成流程检查。
- 开始真实实验：准备固定划分数据，核对“论文训练设置”，再运行核心或集成配置。
- 评估复现程度：查看“论文结果与实际验证状态”和“已知边界”。

## 论文任务与方法

研究任务是在药品泡罩图像中检测正常药片与缺陷腔体。输入是一张图像，输出是矩形框、四类类别和置信度；它属于目标检测，不输出像素级缺陷分割，也不直接计算整板产品的放行/剔除判定。

| COCO类别ID | 类别名 | 论文标注含义 |
|---:|---|---|
| 1 | medicine | 完好包装内的正常药片，框覆盖药片 |
| 2 | empty | 包装完好、没有药片，框覆盖完整腔体 |
| 3 | broke | 包装损坏且有药片，框覆盖完整腔体 |
| 4 | lack | 包装损坏且没有药片，框覆盖完整腔体 |

每个腔体只赋一个类别；broke不再额外标medicine。损伤跨多个腔体时逐腔体标注。代码内部类别索引为0–3，COCO文件类别ID为1–4。

论文的核心是 **Structure-Constrained Optical Augmentation（SCOA）**：在训练时，从一个共享的光学候选池中，先排除破坏缺陷外观结构的候选，再选择对当前EMA检测器较困难的候选。CBAM、WIoU v3和DIoU-NMS是可独立开关的支持模块。

训练流程如下；推理只运行选定的EMA检测网络与后处理，不再生成或筛选增强候选。

```text
原图与标注
  ├─ 80%：基本增强 → Mosaic/affine（第1–285轮）、HSV、水平翻转
  └─ 20%：左上letterbox → 共享光学draw → 3个幅度候选
            → 缺陷框光度守卫 → 可选局部结构筛选
            → 均匀选择 / 冻结EMA困难排序 → 空集或异常时按规则回退
两条分支的输出 → YOLOX重新计算SimOTA匹配并训练 → 成功更新后推进EMA
每轮验证 → 用未舍入AP50选择best EMA → 冻结后在test/外部条件上评价
```

### 核心A–D对照

| 配置 | 光学分支 | 光度守卫 | 结构筛选 | 困难选择 | 对应配置项 |
|---|---|---|---|---|---|
| basic-only | 关闭 | — | — | — | `optical.policy=basic` |
| A | 20% | 开 | 关 | 关 | `optical.policy=A` |
| B | 20% | 开 | 开 | 关 | `optical.policy=B` |
| C | 20% | 开 | 关 | 第15轮开始 | `optical.policy=C` |
| D / SCOA | 20% | 开 | 开 | 第15轮开始 | `optical.policy=D` |

A–D对同一image ID、epoch、seed共享候选生成结果，只改变筛选或选择规则。C/D在第1–14轮均匀选择；A/B始终均匀选择。所有组都先应用光度守卫；空可行集返回预处理原图。光学操作保持框和padding不变，基本分支的几何操作则同步变换框。

basic-only在每个输入上进入基本分支；光学开启时，HSV/flip只出现在80%的基本分支内。因此，光学开启后的整体HSV/flip概率为64%/40%，basic-only为80%/50%。D与basic-only的比较同时包含这项分支概率变化；A–D共享池对照更适合分析筛选与选择的贡献。

### 关键公式与实现对应

**光学候选，论文式(1)–(7)。** 一个公共draw决定1–3个旋转椭圆高斯光斑，以及一个局部模糊/锐化区域；三候选只改变眩光幅度系数：

$$
I_j=\operatorname{clip}(I_{\mathrm{trans}}+a_jG,0,1),\qquad a_j\in\{0.5,1.0,1.5\}.
$$

光斑幅度在0.15–0.40、两轴标准差在8–32像素、角度在0–π之间采样，支持截断为三倍标准差椭圆。局部滤波圆半径32–96像素、混合强度0.2–0.5，模糊与锐化各50%；模糊σ为0.3–1.2，锐化σ为0.5–1.0、锐化强度0.2–0.5。滤波核大小为2ceil(3σ)+1，仅在有效内容区以反射边界滤波后局部混合。实现见 `scoa/optics.py`。

**光度守卫，式(8)–(9)。** 对empty/broke/lack框，要求连续框中心的解析光斑增量不超过0.15，新过曝像素比例不超过0.30。过曝用最大颜色通道≥0.98判断；“新过曝”相对两种光学扰动之前的原图计算，分母包含原本已过曝像素。实现见 `scoa/structure.py:photometric`。

**局部结构约束，式(10)–(12)。** 物理RGB亮度为0.299R+0.587G+0.114B，存储为BGR时也按此物理通道解释。将每个缺陷框划为4×4网格，仅由原图保留至少2像素且总体方差≥1e−4的tile；没有合格tile时退为整个有效框。每个保留区域的分数为：

$$
s_{ijr}=\operatorname{clip}\!\left(\frac{2\operatorname{cov}(L_r,L_{jr})+\epsilon_q}{\operatorname{var}(L_r)+\operatorname{var}(L_{jr})+\epsilon_q},0,1\right),\quad \epsilon_q=10^{-4}.
$$

取升序第ceil(0.25n)个分数，不插值；例如16个tile取第4小。每个缺陷框的该分数均须≥0.75。少于2个有效像素的缺陷框会使A–D直接回退并记录。没有缺陷框时结构条件为空真；medicine不受此结构筛选显式保护。实现见 `reference_regions`、`structure_scores`。

**困难排序，式(13)–(15)。** 冻结EMA在原图上做SimOTA匹配，得到每个实例的正位置集合；所有候选复用同一匹配。实例损失是在这些位置上平均的“四类one-hot BCE之和 + objectness BCE + 5×普通IoU损失”：

$$
\ell_i(X)=\frac{1}{|P_i|}\sum_{u\in P_i}\left[\sum_{k=1}^{4}\operatorname{BCE}(p_{uk}(X),e_{c_i,k})+\operatorname{BCE}(o_u(X),1)+5(1-\operatorname{IoU}(\hat b_u(X),b_i))\right].
$$

候选难度先计算相对原图的实例损失增量，再在类内平均、在出现的类别间平均；类别集合包括medicine：

$$
H_j=\frac{1}{|C_I|}\sum_{c\in C_I}\frac{1}{|I_c|}\sum_{i\in I_c}[\ell_i(I_j)-\ell_i(I)].
$$

从可行集取最大H，同分选最小候选索引；H可以为负。空标注、任一实例无匹配或非有限评分都回退到非空可行集的均匀随机选择，不能丢弃未匹配实例。排序不更新EMA、BN或WIoU均值；学生训练时重新做匹配。实现见 `scoa/difficulty.py`、`scoa/selection.py`。

**EMA与训练损失，式(16)–(17)。** 第一次成功训练更新后计数从1开始：

$$
\mu_t=0.9998(1-e^{-u_t/2000}),\qquad\bar\theta_t=\mu_t\bar\theta_{t-1}+(1-\mu_t)\theta_t.
$$

$$
L_{\mathrm{train}}=L_{\mathrm{cls}}+5L_{\mathrm{reg}}+L_{\mathrm{obj}}+\mathbf1[e>285]L_1.
$$

核心A–D使用训练回归损失1−IoU²；困难评分始终使用上式的1−IoU，不随训练WIoU开关改变。集成WIoU v3取α=1.9、δ=3、均值初值1、更新率1e−4、下限1e−7，比值和包围框对角线detach，中心距离保留梯度。EMA只平滑浮点参数/缓冲。

### 支持模块与数据维度

CBAM插在YOLOPAFPN相对的8个CSP主分支Bottleneck序列后、旁路拼接前：`backbone.dark2.1`、`backbone.dark3.1`、`backbone.dark4.1`、`backbone.dark5.2`、`C3_p4`、`C3_p3`、`C3_n3`、`C3_n4`。对应hidden channels为32/64/128/256/128/64/128/256，reduction=16，空间分支为无bias的7×7卷积、BN和sigmoid。

640输入的三尺度特征是 `[B,128,80,80]`、`[B,256,40,40]`、`[B,512,20,20]`。原始预测为 `[B,8400,9]`，最后9维为cx/cy/w/h、objectness logit、4个class logit。逐类NMS后输出至多100个框。DIoU-NMS使用IoU减中心距平方/最小包围矩形对角线平方，抑制值>0.45的低分同类框。

完整“章节/公式→文件/函数”表见 [论文—代码对应表](docs/02_paper_code_map.md)。

## 项目内容

```text
configs/                 论文、baseline、integrated、合成smoke配置
weights/                 官方COCO权重下载位置、来源与SHA256；二进制不随仓库分发
scoa/
  _vendor/               官方YOLOX 0.3.0实际模型与SimOTA源码
  optics.py              椭圆高斯眩光、局部模糊/锐化、共享3候选
  structure.py           光度守卫、参考tile、contrast-structure、顺序统计量
  difficulty.py          固定原图SimOTA匹配、实例损失、类别平衡困难评分
  selection.py           A/B/C/D、随机/异常回退、SMR-B与legacy
  model.py               YOLOX-s构建、CBAM、WIoU-v3、EMA、优化器
  data.py / basic.py     COCO、左上letterbox、Mosaic/affine/HSV/flip
  augment.py             AM-photo与论文定义的AugMax-P
  train.py               SGD/AMP训练、恢复、验证AP50选择EMA
  evaluate.py            固定split/条件ID推理与评价
  metrics.py             101点AP、10种IoU、缺陷micro P/R
  postprocess.py         逐类标准NMS/DIoU-NMS，全局top100
  infer.py / benchmark.py 单图推理、论文计时协议
tools/                   数据检查、实验清单、审计、真实结果汇总
tests/                   数学、边界、状态与实际前反向测试
third_party/             上游来源、commit、SHA256、Apache-2.0许可证
source_evidence/         本地论文核对材料；不随公开仓库分发
experiments/             不带虚构实测结果的对照文件
docs/                    论文理解、映射、验证、实验对照与权重说明
validation_records/      已实际执行的数值、流程与权重加载检查记录
```

全部命令在解压后的项目根目录运行。项目源码不使用用户电脑的绝对路径，也不会自动探测GPU型号或修改全局Python环境。

## 1. 在另一台机器安装

论文参考组合是 Python 3.9、torch 1.12.1、CUDA 11.6。建议用专用环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-torch-reference.txt
python -m pip install -r requirements.txt
python -m pip install -e .
```

Linux/macOS激活命令为 `source .venv/bin/activate`。CUDA参考组合适用于兼容的NVIDIA环境；这里不推断目标机器的驱动条件。CPU流程检查可选 Python 3.9–3.12 和 `requirements-cpu.txt`，替换第一条torch依赖安装命令；CPU检查组合与论文软件版本不同。

若只做CPU流程检查，在独立环境中安装以下组合，**不要再叠加安装CUDA参考torch依赖文件**：

```powershell
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
python -m pip install -e .
```

论文作者报告的环境为Windows11、Python3.9、PyTorch1.12.1、CUDA11.6、cuDNN8.4.1、RTX3080Ti12GB和i7-12700K；这是实验设备记录，不是测得的最低硬件要求。本项目已运行验证使用隔离的Python3.12/torch2.2.2+cpu。完整300轮CUDA参考组合尚未验证。

项目内模型与NMS不依赖torchvision编译算子；reference依赖文件仍保留论文年代对应的torchvision配对。不要把不同torch/torchvision版本任意混装。

## 2. 最小可运行示例：合成数据闭环

```powershell
python tools/make_synthetic.py
python -m pytest -q
python -m scoa.train --config configs/smoke.yaml --device cpu --threads 2
python -m scoa.evaluate --config configs/smoke.yaml --checkpoint runs/synthetic_smoke/seed_42/best.pth --device cpu --threads 2 --output runs/synthetic_smoke/test.json
python -m scoa.infer --checkpoint runs/synthetic_smoke/seed_42/best.pth --image data/synthetic/images/test_20001.png --device cpu --threads 2 --output runs/synthetic_smoke/inference
```

这是64×64、1轮、随机初始化的**流程测试**。实际运行网络、SimOTA、损失、反向、SGD、EMA、保存与恢复、推理和AP评价；不会达到有意义的真实检测精度。**合成指标不能证明复现论文实验。** 重复训练时使用 `--set output=runs/smoke_second`；已有latest文件会阻止无意覆盖。

也可一次执行并记录全部退出码：`python tools/run_smoke.py --output runs/new_verification`。当前已通过44个数值/状态/输入与评价一致性测试及完整合成训练闭环；详细边界见最新检查报告。默认会运行测试，不必另外重复执行一遍pytest。

## 3. 准备真实数据与固定划分

### 论文的数据划分

论文§4.1报告8532张1920×1080图像，拍摄对象来自同一产线、同一产品。各类别目标数如下：

| 划分 | 图像数 | medicine | empty | broke | lack | 总目标数 |
|---|---:|---:|---:|---:|---:|---:|
| train | 6826 | 59368 | 2452 | 1992 | 1328 | 65140 |
| val | 853 | 7421 | 306 | 249 | 166 | 8142 |
| test | 853 | 7421 | 307 | 249 | 166 | 8143 |

同一物理泡罩、相邻帧和多视角应留在同一划分。论文固定图像/实体ID未公开；按相同比例另做随机划分并不等同原论文划分。项目不附带真实数据，也不通过下载COCO权重获得药品图像。

### COCO目录与字段

```text
data/blister/
  images/                  file_name相对该目录，可以含子目录
  annotations/
    train.json
    val.json
    test.json
```

使用COCO JSON，`categories`必须严格为：

```json
[{"id":1,"name":"medicine"},{"id":2,"name":"empty"},{"id":3,"name":"broke"},{"id":4,"name":"lack"}]
```

每个annotation需要 `id,image_id,category_id,bbox:[x,y,w,h],iscrowd:0`，ignore省略或为0。每个image需要 `id,file_name,width,height`；建议增加 `entity_id` 供同一物理泡罩跨split检查。broke/lack标整腔体，每腔体一个类别。用**作者固定split ID**，不要自行随机重新划分后声称同划分复现。

下面只示意文件格式，图像文件需要自行提供；它不是论文中的真实样本或数据划分：

```json
{
  "images": [{"id": 1, "file_name": "train/example.png", "width": 1920, "height": 1080, "entity_id": "blister_0001"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [100, 120, 160, 200], "area": 32000, "iscrowd": 0, "ignore": 0}],
  "categories": [{"id": 1, "name": "medicine"}, {"id": 2, "name": "empty"}, {"id": 3, "name": "broke"}, {"id": 4, "name": "lack"}]
}
```

不要预先把标注归一化到0–1，也不要把框改成YOLO文本格式；当前数据入口读取原图像素坐标的COCO xywh。`file_name`相对 `data.root/image_dir`。原始uint8 BGR图像经双线性resize后，放在640×640画布左上角，padding=114；对论文1920×1080原图，有效内容为640×360，其他长宽比按原比例处理。网络收到BGR CHW float32、0–255；只有光学算子临时使用0–1，输出再乘255。连续框使用半开边界，像素按中心坐标归属区域。

```powershell
python tools/check_dataset.py --config configs/paper.yaml --paper-counts
```

该工具检查图像解码、尺寸、框、类别、精确重复、已有实体ID跨split和论文数量。它不能替代相邻帧/近重复人工复核，也不能从数量证明split身份一致。外部组可各自提供一份COCO JSON，并在评价时覆盖 `data.test=annotations/external_illumination.json`；光学条件子集使用 `--ids fixed_stratum_ids.json`。

若使用自己的数据而非论文原划分，可不传 `--paper-counts` 做格式检查；这只能支持新数据上的方法实验。要改变数据根目录，修改正式配置的 `data.root`；训练与评价必须使用同一根目录设置。标注检查工具没有 `--set` 参数，需要读取已编辑的配置。

## 4. 正式核心训练与推理

### 论文训练设置

| 设置 | 论文/正式配置 |
|---|---|
| 检测器 | YOLOX-s，官方0.3.0基底；depth=.33、width=.5 |
| 输入、训练轮数、batch | 固定640×640、300轮、batch4 |
| 优化器 | SGD，lr=.000625，momentum=.9，weight_decay=.0005，AMP |
| 学习率 | 5轮warmup后cosine；本实现采用上游二次warmup、起点0、min_lr_ratio=.05 |
| 基本几何增强 | 四图Mosaic；旋转±5°、平移±.10、scale .8–1.2、shear±2° |
| 基本颜色/翻转 | HSV概率.8、gain5/30/30；水平翻转概率.5；MixUp关闭 |
| 最后15轮 | 第286–300轮关闭Mosaic/affine，启用L1；分支内HSV/flip保留，光学分支未因此关闭 |
| 光学候选 | 概率.20、J=3、幅度.5/1/1.5，结构4×4、q=.25、τ=.75 |
| 困难排序 | C/D第15轮开启，class-balanced；EMA decay ramp见上方公式 |
| 随机种子 | 42、3407、2026，分别训练；生成、选择与基本增强使用独立流 |
| best选择 | 每轮验证，未舍入AP50最高的EMA；同分保留更早epoch |

二次warmup、最小学习率比、Nesterov、BN/bias无decay及Mosaic具体抽样等属于从上游采用的实现选择，不能当作论文逐项披露的事实，详见后面的假设表。修改batch、轮数或输入尺寸会偏离本表；配置不会自动按新batch缩放学习率。

### COCO初始化权重的获取

公开源码不包含官方COCO预训练 `weights/yolox_s.pth`；先运行 `python tools/download_weights.py --model s` 下载。`configs/paper.yaml`及其baseline/integrated派生配置默认使用它。合成smoke配置显式保持 `pretrained: null`，用于原来的随机初始化流程测试。

准备真实数据后直接运行：

```powershell
python -m scoa.train --config configs/paper.yaml --device cuda --set name=D --set output=runs/D
python -m scoa.evaluate --config configs/paper.yaml --device cuda --checkpoint runs/D/seed_42/best.pth --set name=D --output runs/D/seed_42/test_standard.json
```

权重来自[官方YOLOX release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)，官方0.3.0 README将它列为标准YOLOX-s权重。来源、官方asset ID、文件大小和本地SHA256存于 `weights/yolox_s.sha256.json`。论文未给hash，所以不能确认与作者手中的二进制完全相同。它是80类COCO初始化权重，不是已经训练好的药品四类检测权重。

只拿到源码包时，可运行 `python tools/download_weights.py --model s` 另行下载；下载使用临时文件，校验官方文件大小后才写入最终文件名。已经附带权重时无需重下。运行 `python tools/verify_pretrained.py` 可检查本地校验值、核心/集成模型的实际权重加载与640前向。

训练按形状匹配加载，3尺度的80类分类预测weight/bias会跳过，保留本项目4类分类层初始化；新增CBAM参数和WIoU运行均值也保留独立初始化。加载/跳过清单写入训练元数据。`scoa.infer`要求本项目训练格式的checkpoint：论文结果推理推荐使用验证最优的 `best.pth`；`latest.pth`同样可加载，但不一定是最优轮次。它不直接接收COCO初始化文件。

### baseline、核心D与集成配置

baseline直接运行 `python -m scoa.train --config configs/basic.yaml --device cuda`，输出到 `runs/basic/seed_42/`。上面的核心D示例输出到 `runs/D/seed_42/`。集成配置开启CBAM/WIoU/SCOA，最终评价用DIoU-NMS：

```powershell
python -m scoa.train --config configs/integrated.yaml --device cuda
python -m scoa.evaluate --config configs/integrated.yaml --device cuda --checkpoint runs/integrated/seed_42/best.pth --output runs/integrated/seed_42/test_diou.json
python -m scoa.evaluate --config configs/integrated.yaml --device cuda --checkpoint runs/integrated/seed_42/best.pth --set evaluation.nms=standard --set name=integrated-standard --output runs/integrated/seed_42/test_standard.json
```

核心变体只需覆盖配置：`--set optical.policy=A`、`B`、`C`、`D`。baseline用 `configs/basic.yaml`；完整集成用 `configs/integrated.yaml`。正式运行保持640输入、300轮、batch4、三seed；不要把smoke配置用于论文结果。

```powershell
python -m scoa.train --config configs/integrated.yaml --device cuda --set train.pretrained=weights/yolox_s.pth --set train.seed=3407
python -m scoa.train --config configs/paper.yaml --device cuda --set name=D --set output=runs/D --resume runs/D/seed_42/latest.pth
python -m scoa.infer --checkpoint runs/integrated/seed_3407/best.pth --image path/to/image.png --device cuda --nms diou --output runs/predict
```

resume必须使用与原训练一致的配置覆盖项。例如用 `name=D` 训练，恢复也需附加该项。恢复允许改变数据根目录、输出目录、worker数量和预训练路径，其余训练设置与标注hash必须一致。保存epoch边界的全部optimizer/scaler/随机状态；不支持半个epoch断点恢复。

可用 `--stop-after-epoch 1` 在第1轮末保存并停止，而不改变配置中的总轮数、学习率和增强计划，再用 `--resume`继续。`python tools/check_resume.py --output runs/new_resume_check --workers 1` 可重跑集成模型的CPU精确等价性检查；这项检查不证明跨设备/跨版本或中途半个epoch续训的逐bit一致。

每个seed目录输出：`resolved_config.yaml`、`run_metadata.json`、`candidates.jsonl`、`epochs.jsonl`、`latest.pth`、`best.pth`。best只按未舍入验证AP50选EMA，平手保留更早轮。candidates记录共享draw、所有候选守卫、结构分数、可行集、被选索引、困难得分及实例损失/匹配索引；长训练日志可能较大。

checkpoint验证统一使用 `train.checkpoint_nms=standard`；`evaluation.nms=diou`只控制最终评价。这样完整集成与两种后处理复用同一权重，避免为每种NMS重新挑epoch。

输出指标为0–1比例，论文为百分数。评价帧统一为letterbox输入坐标；单图展示时另换回原始图坐标，评价不混用两种坐标。

单图推理会输出 `predictions.json` 和 `prediction.png`。JSON包含原图坐标 `xyxy_original`、输入帧坐标 `xyxy_input`、类别ID/名称与score；推理默认显示阈值.25，可用 `--score` 调整。该展示阈值不应替换AP评价的.001阈值。

## 5. 三seed与组件实验

先生成可审查的实验清单；加 `--execute` 才会依次执行。core=15次训练，components=24次训练，每个组件配置用两种NMS复用best权重，不能算额外训练复制。

```powershell
python tools/experiment_matrix.py --suite core --pretrained weights/yolox_s.pth
python tools/experiment_matrix.py --suite components --pretrained weights/yolox_s.pth --output experiments/components
python tools/experiment_matrix.py --suite auxiliary --pretrained weights/yolox_s.pth --output experiments/auxiliary
```

执行示例：在以上命令末尾加 `--execute`。大规模实验请先完成数据与1个真实小运行检查；不会因为项目交付而自动启动300轮作业。

如果只运行核心D的三seed实验，可在PowerShell中执行下面的循环。每个seed完成300轮训练后，用该seed的best EMA评价固定test；这些是正式实验命令，不属于快速smoke检查。

```powershell
foreach ($experimentSeed in 42,3407,2026) {
    python -m scoa.train --config configs/paper.yaml --device cuda --set name=D --set output=runs/D --set "train.seed=$experimentSeed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $experimentSeed" }
    python -m scoa.evaluate --config configs/paper.yaml --device cuda --checkpoint "runs/D/seed_$experimentSeed/best.pth" --set name=D --output "runs/D/seed_$experimentSeed/test_standard.json"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for seed $experimentSeed" }
}
```

S1=`optical.policy=B, optical.structure_mode=whole`；S2为`mean`；S3/B为`quantile`。这些训练阈值均为0.75；论文表1(b)另外校准的审核阈值不应替换训练阈值。D-instance覆盖 `optical.difficulty_average=instance`。

SMR-B必须有D离线日志及10个stage/stratum的真实统计。不要伪造缺失stratum：

```powershell
python tools/fit_smr.py --logs runs/D/seed_42/candidates.jsonl runs/D/seed_3407/candidates.jsonl runs/D/seed_2026/candidates.jsonl --output experiments/smr_weights.json
python -m scoa.train --config configs/paper.yaml --device cuda --set optical.policy=SMR-B --set optical.smr_weights=experiments/smr_weights.json --set train.pretrained=weights/yolox_s.pth --set name=SMR-B --set output=runs/SMR-B
```

拟合工具给出正权重与各层幅度差；如果缺某seed/stage/stratum，正式训练会报错。新拟合的权重不是作者未公开的权重。legacy、AM-photo、AugMax-P可通过policy选择。legacy原始实现未公开，本项目按文中10次first-accepted规则重建；AM-photo的PIL量化和算子细节也属于显式实现假设。AugMax-P实现论文适配，不等同AugMax-DuBIN。

## 6. 真实结果对照、人工审核与计时

### 评价协议：论文§4.4

| 项目 | 本项目按论文执行的规则 |
|---|---|
| 检测score | objectness概率 × 最大类别概率，按YOLOX方式确定预测类别 |
| AP保留阈值 | score≥.001，按分数降序做同类一对一匹配 |
| AP采样 | 101个recall坐标；IoU=.50,.55,…,.95，共10个阈值 |
| AP50/AP75 | 在相应IoU阈值上对四类AP平均 |
| AP50:95 | 对四类、十个IoU阈值的AP平均 |
| Precision/Recall | score≥.25、IoU≥.50；重复框算FP，错类按同类匹配形成FP/FN |
| Defect P/R | 合并empty/broke/lack的TP/FP/FN后计算，不能先算逐类P/R再平均 |
| 抑制与数量 | 逐类标准或DIoU-NMS，阈值.45；抑制后全部类别一起取top100 |
| 坐标与ignore | 640×640输入帧、连续坐标、ignore=iscrowd=0 |

缺陷指标为：

$$
P_D=\frac{\sum_{c\in D}TP_c}{\sum_{c\in D}(TP_c+FP_c)},\qquad R_D=\frac{\sum_{c\in D}TP_c}{\sum_{c\in D}(TP_c+FN_c)},\quad D=\{\mathrm{empty,broke,lack}\}.
$$

机器输出用0–1比例，乘100才是论文的百分数。工具对无分母的指标输出 `null`；在自定义子集上，AP会排除没有GT的类别，因此缺类子集的AP不能直接等同全四类测试AP。正式论文test含全部四类。

### 三seed统计与实际结果汇总

```powershell
python tools/compare_results.py --output experiments/paper_comparison.json
python tools/compare_results.py --results runs/D/seed_42/test_standard.json runs/D/seed_3407/test_standard.json runs/D/seed_2026/test_standard.json --output experiments/comparison_actual.json
```

不传结果会生成参考值和null实测值；传入真实逐seedtest JSON后才计算实际均值、样本SD和配对差。三个required seed状态会独立标记。工具拒绝synthetic记录、val/train记录和重复seed；用户仍需确认标注hash和固定split身份。组件清单中的名字为 `D_C1W1` 等，比较整体论文数值前需在真实评价时设置 `name=integrated`，并使用DIoU-NMS；baseline为 `name=basic`，核心为 `name=A/B/C/D`。

结果汇总还会核对实际评价图像ID、标注hash、640坐标帧、阈值、NMS与训练组件。`integrated`对应论文DIoU参考74.03；`integrated-standard`对应标准NMS参考73.78。同一标注文件的不同子集不能混为同一test集合；较早版本缺少评价ID签名的JSON需要重新运行评价。

可选的官方COCOapi数值对照：安装 `requirements-evaluation-optional.txt`，运行 `python tools/coco_crosscheck.py`；也可传 `--records path/to/test.predictions.json` 检查真实预测。当前固定诊断样例的最大AP误差约1.1e−16；它不等同作者未公开评价器的逐项恢复。

汇总使用seed42/3407/2026的均值和样本标准差（分母n−1）。B−A、D−B等配对效应应先在同seed未舍入数值上相减，再汇总；不能从论文的均值±SD反推逐seed结果。评价JSON保留 `seed`、`selected_epoch`、权重/标注SHA256、`synthetic`、坐标帧和后处理方式，可追溯每次结果。

### 人工cue审计与外部条件

人工cue审计CSV列为 `pair_id,source_entity_id,rule,class,accepted,label`。label只能为discernible/lost/uncertain/ambiguous_reference；rule区分photo和structure。每个(pair_id,rule)一条最终仲裁记录，不能把两名评分者当两个独立样本。

```powershell
python tools/audit_cues.py --csv real_review.csv --output experiments/cue_audit_actual.json
python -m scoa.benchmark --checkpoint runs/integrated/seed_42/best.pth --image path/to/real.png --device cuda --output experiments/latency_actual.json
```

审核工具只汇总真实的人工判读，不会用结构分数自动生成语义标签。论文校准集150对、独立审核300对，另有120个选中输出对；需要保留源图/实体分离与两名评分者初评、仲裁记录。接受cue loss的分母仅含接受后明确的discernible/lost对，拒绝率的分母为全部discernible对，uncertain与ambiguous reference应单列。

论文的内部glare分级数量为420/283/150，transmission为400/300/153；外部batch/illumination/packaging为300/240/300。复核这些结果需要作者相应固定ID、ROI及原图。`--ids`只按给定ID筛选，不会根据图像自动恢复论文分级。分层AP分别计算，不能按图像数加权平均来还原完整test AP。

计时默认3轮，每轮200 warmup、1000计时；CPU单调钟包围预处理、传输、网络和后处理并同步GPU，CUDA events前向另记。它不衡量生产线持续吞吐。训练日志时间不包括验证与checkpoint写盘，也不含DataLoader等候；与论文完整epoch时间的对照应在目标机器按相同监测窗口补测。

## 论文结果与实际验证状态

下表的AP50:95来自论文表2/3/5/8，单位为%，均为三seed均值±样本SD。**“论文报告”不是本项目在真实数据上测出的结果。**

| 配置 | 论文报告AP50:95 | 论文位置 | 本项目真实数据实测 |
|---|---:|---|---|
| basic-only YOLOX-s | 66.53±0.43 | 表2(a) | 尚未运行 |
| A：共享池随机 | 68.35±0.34 | 表2(a) | 尚未运行 |
| B：结构+随机 | 69.45±0.39 | 表2(a) | 尚未运行 |
| C：光度+困难 | 69.05±0.28 | 表2(a) | 尚未运行 |
| D：SCOA | 70.05±0.40 | 表2(a) | 尚未运行 |
| SCOA+CBAM+WIoU，标准NMS | 73.78±0.40 | 表3 | 尚未运行 |
| 同一集成权重，DIoU-NMS | 74.03±0.42 | 表3 | 尚未运行 |
| SMR-B | 69.88±0.41 | 表2(a) | 尚未运行，另缺作者拟合权重 |
| AM-photo | 68.95±0.36 | 表2(a) | 尚未运行 |
| AugMax-P | 69.95±0.24 | 表8(a) | 尚未运行 |

目前已实际完成的检查如下，证据见 [验证记录](docs/03_validation.md)、[权重接入记录](docs/05_pretrained_weights.md) 及 `validation_records/20261027/`：

| 检查范围 | 实际完成内容 | 能说明什么 |
|---|---|---|
| 数学/维度/状态 | 44项测试通过；涵盖tile、评分、EMA、WIoU、NMS、L1、半精度大框与输入校验 | 实现与所列数学/工程约束一致 |
| epoch续训 | 集成模型连续2轮与1轮后恢复再1轮的模型/EMA/优化器等状态精确一致；workers=1 | 当前CPU组合的epoch边界续训可复现 |
| 合成闭环 | 核心和集成的训练、SGD/EMA、checkpoint加载、评价、单图推理 | 小规模程序流程可执行 |
| 真实COCO权重 | 两种模型各加载456个匹配tensor，跳过6个分类tensor；640前向和合成训练通过 | 官方初始化文件与当前网络可兼容迁移 |
| 评价器 | 与pycocotools固定诊断样例的AP最大绝对误差约1.1e−16 | 该样例下符合COCO式AP语义 |
| 真实实验 | 固定split图像/标签与作者档案缺失 | 当前不能核验论文AP、cue比例或时延 |

随机初始化的64输入、1轮合成验证AP为0，是实际小流程输出，既不能证明也不能否定论文方法的真实效果。严禁把它或COCO诊断样例的AP填入论文对照列。

## 7. 已知边界

### 论文已说明、实现假设和缺失材料

| 类型 | 内容 | 对复现的影响 |
|---|---|---|
| 论文明确说明 | 候选数/幅度、光度阈值、结构网格/分位、困难选择起始轮、损失权重、主要训练/评价设置 | 正式配置直接落实 |
| 从上游采用的实现选择 | Nesterov、BN/bias无decay、二次warmup、min_lr_ratio=.05、Mosaic概率1及具体抽样 | 影响收敛，不能声称作者逐项确认 |
| 光学/数值实现选择 | OpenCV BORDER_REFLECT_101、光学结果回到0–255时保留float32 | 作者仅说明反射边界/输入表示，像素细节可能不同 |
| 随机与采样选择 | seed/epoch/image-ID/stream稳定哈希；有限shuffle、不截断最后batch | A–D池内可对应，但不保证与作者随机轨迹相同 |
| 其它状态/验证选择 | WIoU按非空训练forward更新均值；统一标准NMS验证AP50选checkpoint | 更新粒度/选择器细节需作者源码确认 |
| 数值稳定修复 | AMP训练解码、平方IoU与匹配IoU改为FP32算术，数学公式不变 | 降低FP16面积/指数运算溢出的风险；GPU完整训练仍未验证 |
| 未公开材料 | 真实数据及固定split ID、作者pretraining hash、SMR权重、自定义评价器和逐seed档案 | 不能证明同划分、同初始化或全部结果复现 |

- 当前可交付的是核心/配套/上述辅助策略的运行实现；真实8532图实验、840张外部条件、人工审核、曲线与时延尚需真实材料。
- 没有实现YOLOv8s或RT-DETR的独立训练仓库适配器；它们属于横向系统比较，不能以此项目中的YOLOX结果代替。论文未公开其全部配置/权重身份，因此这部分仍未复现。
- 论文没有披露所有随机抽样与优化细节。独立随机流、SGD默认、Mosaic与反射模式等见信息缺口表；相同数字种子不能保证与作者逐步一致。
- 参考torch1.12.1/CUDA11.6完整300轮尚未验证。隔离CPU验证只覆盖验证记录中明确列出的组合。确定性算法无法保证跨平台/版本逐bit相同。
- 项目不含真实训练结果或可用的预训练缺陷检测权重；合成smoke的checkpoint只供流程检查。

此外，本项目实际参数数为8,938,843（核心）和8,964,445（集成），而论文记为8.910M/8.936M。当前无法确定是统计口径还是未公开结构细节导致；保留官方基底与论文指定插入点，不为匹配舍入数字改造网络。

完整信息清单见 [论文理解与复现方案](docs/01_paper_and_plan.md)，逐项实验边界见 [实验对照](docs/04_experiment_comparison.md)。

## 常见运行问题

| 现象 | 处理方式 |
|---|---|
| 找不到 `data/blister/annotations/train.json` | 真实数据未附带。准备COCO目录，或用 `configs/smoke.yaml` 跑合成流程 |
| 找不到 `weights/yolox_s.pth` | 运行 `python tools/download_weights.py --model s` |
| 出现6个分类tensor未加载 | COCO80类转药品4类的预期行为；检查其它匹配tensor已加载 |
| 给 `scoa.infer` 传COCO权重后缺少config/ema | 先训练后推理；论文结果使用best.pth，latest.pth虽可加载但未必最优 |
| 已有latest.pth，训练拒绝继续 | 用相同设置 `--resume`，或通过 `--set output=runs/new_run` 新建输出 |
| resume配置/标注hash变化 | 复用原配置中的name、模型、seed和训练设置；不要换split后沿用旧优化器状态 |
| evaluation model/input_size不匹配 | 选择与checkpoint一致的config；完整集成不能用纯baseline配置评价 |
| 没有CUDA或只想检查流程 | 安装CPU依赖，使用 `--device cpu --threads 2`；正式GPU训练结果仍待验证 |
| 缺陷precision为null | 阈值下没有缺陷预测，分母为0；它不是0%或100%，不要擅自填值 |
| SMR-B缺少seed/stage/stratum | 先收集真实D日志并拟合；未观测的层不会由工具伪造 |

## 来源与许可

检测基底固定到[YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)，commit记录在`third_party/upstream.json`，官方模型文件SHA256记录在`third_party/model_file_sha256.json`。保留[Apache-2.0许可证](third_party/YOLOX/LICENSE)。本地最小vendor副本只有head utils导入改动；未重写或替换SimOTA为假算法。

WIoU算术对照[作者源码](https://github.com/Instinct323/Wise-IoU/blob/v2/iou.py)，参数使用上传论文的1.9/3.0/1e−4，不套用上游当前默认。CBAM结构对照[作者实现](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)，插入位置以本论文§3.5为准。
