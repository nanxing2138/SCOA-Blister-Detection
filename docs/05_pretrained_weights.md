# 官方预训练权重接入记录

本次从网上取得的是**COCO预训练YOLOX-s**，用于初始化论文训练。它不是论文作者训练完成的medicine/empty/broke/lack检测模型，也不提供论文AP复现证据。

## 来源与文件

- 官方仓库：Megvii-BaseDetection/YOLOX。
- 官方0.3.0 README的标准YOLOX-s行直接指向release `0.1.1rc0`，而非旧版legacy权重。
- 发布页：https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
- 原始文件：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
- 本地文件：`weights/yolox_s.pth`。
- GitHub asset ID：42725001。
- 官方API及实际下载大小：72,089,125 bytes（约68.75MiB）。
- 本地SHA256：`f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30`。

来源元数据存于 `weights/yolox_s.sha256.json`。该旧release没有提供upstream digest，上述SHA256为下载后本地计算；论文未披露作者使用文件的hash，不能确认是同一二进制。

## 接入方式

`configs/paper.yaml`默认 `train.pretrained: weights/yolox_s.pth`；basic和integrated配置继承此设置。路径相对项目根目录，换电脑解压后无需修改本机绝对路径。

`configs/smoke.yaml`显式 `pretrained: null`，保持原来的随机初始化流程测试；若要检查本次真实COCO权重的训练接入，可用：

```text
python -m scoa.train --config configs/smoke.yaml --device cpu --threads 2 --set train.pretrained=weights/yolox_s.pth --set train.allow_scratch=false --set output=runs/pretrained_smoke
```

准备好真实数据后，正式核心训练：

```text
python -m scoa.train --config configs/paper.yaml --device cuda --set name=D
```

完整集成训练：

```text
python -m scoa.train --config configs/integrated.yaml --device cuda
```

仅按名称与形状同时匹配的tensor加载。COCO的三尺度分类层为80类，本项目为4类，恰好跳过6个tensor：`head.cls_preds.{0,1,2}.weight/bias`。其余共享主干、neck、回归/目标性分支均复用。新增CBAM和WIoU运行均值保留初始化。不能把COCO类别标签直接解释成药品类别，也不能将该初始化文件传给只接受本项目训练checkpoint的 `scoa.infer`。

## 已执行验证

`python tools/verify_pretrained.py`已通过，完整结果在 `validation_records/20261027/pretrained_verification.json`：

| 验证项 | 核心模型 | CBAM+WIoU集成 |
|---|---:|---:|
| 实际加载的匹配tensor | 456 | 456 |
| 跳过的源分类tensor | 6 | 6 |
| 对照checkpoint逐tensor相等 | 通过 | 通过 |
| 未匹配的新层初始化保持不变 | 通过 | 通过 |
| 640输入输出形状 | [1,8400,9] | [1,8400,9] |
| 所有前向值有限 | 通过 | 通过 |

验证未使用用户硬件信息，仅在项目已有隔离CPU验证运行时执行。本次还运行了与权重加载相关的4项现有训练契约测试，全部通过。

合成数据上的预训练初始化训练检查记录另存于 `validation_records/20261027/pretrained_training/`；它只验证训练入口实际使用了此权重，不能据此衡量真实缺陷检测效果。

## 分发

当前定稿包位于 `delivery/SCOA_checked_20261027.zip`，将真实权重、来源记录、源码及检查报告一起打包，不包含隔离Python运行时、真实数据或合成训练checkpoint。此前的旧项目和旧压缩包已按用户要求移入回收站；处理记录见 `validation_records/audit_20261027/cleanup.json`。打包工具的 `--include-weights` 会在打包前复核权重SHA256。

GitHub公开源码不包含上述本地交付压缩包或权重二进制。克隆仓库后，先运行 `python tools/download_weights.py --model s` 下载官方初始化权重，再执行需要该文件的命令。
