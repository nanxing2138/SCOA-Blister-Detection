# 验证记录

日期：2026-10-27（Asia/Shanghai）。仅为本项目建立隔离CPU验证依赖，未盘点用户硬件、CUDA或已有深度学习环境。验证组合为Python3.12、torch2.2.2+cpu、NumPy1.26.4、OpenCV4.8.1.78；与论文torch1.12.1/CUDA11.6组合不同。

## 阶段1：证据与基底

已提取823个段落索引、12张表和文档外链；数学对象另行渲染读取，参数录入报告与配置。官方YOLOX0.3.0 tag解析至commit `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`。`tools/verify_vendor.py`通过：6个实际模型文件与上游原始SHA256对应，只有head utils导入适配。语法编译通过。

## 阶段2：数学、维度与状态测试

第一组 `python -m pytest -q`：**23 passed**，81.20秒。随后新增的 `tests/test_training_contract.py`：**4 passed**，7.35秒；合计27个测试用例通过。原始JUnit记录保存在`validation_records/20261027/`，不是手工填的测试输出。

覆盖内容：

- 1920×1080→左上640×360内容、padding114、BGR及标签缩放。
- 连续半开框的像素中心归属、少于2像素的缺陷框处理。
- 物理RGB亮度、总体方差/协方差、亮度平移、负相关clip、平坦参考的whole-box回退。
- 16个tile取第4小的无插值分位数；仅由参考图决定tile。
- 新过曝分母保留原过曝像素，解析高斯中心值与三倍标准差截断，共享候选draw与padding保持。
- 类别平衡和实例平均的区别、负困难得分。
- 640输入的三尺度形状以及8400×9原始预测；与原始YOLOX eval输出逐项数值一致。
- 基线与CBAM/WIoU集成的真实反向传播、有限梯度、优化器更新与EMA推进。
- 固定SimOTA匹配评分、同图困难增量为0；评分前后EMA全部参数/缓冲逐bit不变。
- 首次EMA更新公式、整数缓冲不被平滑；WIoU退化/空输入及eval状态稳定。
- A–D共享pool、无标注评分随机回退、无效框和空可行集返回原图。
- 标准/DIoU抑制、退化预测移除；完美框、重复框、错类与AP/P/R两种score阈值。
- AugMax-P真实5步输入梯度搜索，student权重/BN/梯度缓冲不被修改；simplex投影与正SMR权重拟合。
- 训练损失含真实L1的前反向、warmup/cosine端点、第286轮关闭Mosaic/affine、预训练形状匹配加载。

运行：`python -m pytest -q`。测试中的诊断数组、权重和合成图只用于公式/执行路径验证，不是论文实验结果。

## 阶段3：实际合成训练—评价—推理

完整10步命令在 `validation_records/20261027/execution.json`，所有exit_code均为0。可用 `python tools/run_smoke.py --output runs/new_verification` 在另一台电脑重跑。

合成数据为4张train、2张val、2张test，每图4个示意腔体；输入64、batch2、1轮、随机初始化、CPU、不启用AMP。为了直接覆盖困难选择，smoke配置将光学概率设为1、第1轮启用困难评分。这些都与正式论文训练不同。

| 真实执行项目 | 结果 |
|---|---|
| 核心D训练 | 2次成功optimizer/EMA更新；mean loss=10.659932136535645 |
| 集成CBAM+WIoU训练 | 2次成功optimizer/EMA更新；mean loss=9.979569911956787 |
| 各配置候选日志 | 各4个光学事件；1个实际执行困难评分，3个空可行集回退 |
| checkpoint | best/latest保存成功，评价和单图推理实际加载best EMA |
| 合成val/test | AP为0；这是一轮随机初始化的实际输出，不是方法有效性结论 |
| 推理产物 | predictions.json与prediction.png实际写出；阈值0.25下0个检测 |
| 数据工具 | 合成COCO解码、框、hash与实体跨split检查通过 |
| 实验清单 | 成功生成核心三seed共15个训练身份，未执行正式作业 |

训练时间仅记录本次CPU小流程的步骤时间，不与论文GPU时延作比较。固定像素尺度的光斑在64输入上相对很大，因此回退较多，不用于估计640实验的回退率。

## 阶段4：评价器独立交叉检查

安装可选 `requirements-evaluation-optional.txt` 后运行：

```text
python tools/coco_crosscheck.py --output runs/coco_crosscheck.json
```

pycocotools2.0.11在固定诊断样例上的结果：AP50=0.7718646864686469，AP75=0.584983498349835，AP50:95=0.5912953795379537。自定义实现的AP50/AP75完全一致，AP50:95最大绝对误差为1.1102230246251565e−16。

该样例含重复框、错类、漏检及0.5/0.9等不同IoU；不是论文数据。COCOapi的maxDets是逐类别，本项目先执行论文要求的全局top100，所以在真实记录交叉检查前必须保留该预筛选。这个检查也不能证明等同作者未公开的自定义评价器。

## 发现的论文—实现差异

按实际 `sum(p.numel() for p in model.parameters())`：

| 配置 | 本项目实际参数数 | 论文报告 |
|---|---:|---:|
| YOLOX-s四类 | 8,938,843 | 8.910 M |
| 加八处CBAM（WIoU无可训练参数） | 8,964,445 | 8.936 M |

新增CBAM为25,602个参数。基底与模块位置均经源码/张量检查，但总数与论文不一致。可能涉及统计口径或未披露实现差异，目前无法确定。没有为了匹配论文的舍入数字擅自删改网络；这项差异需要作者代码/统计脚本才能解决。

## 未验证与当前不能复现

**未验证**：torch1.12.1/CUDA11.6的300轮AMP训练、真实三seedAP、正式GPU显存/时延、完整断点续训的逐步等价性；辅助策略虽有代码与部分数值测试，未完成各自300轮训练。

**后续补充**：2026-10-27已从官方release下载真实COCO预训练文件，完成核心/集成模型456个匹配tensor的逐项加载验证及640前向。详见 `docs/05_pretrained_weights.md` 和对应JSON记录；这不改变真实药品实验尚未验证的状态。

**当前缺材料无法复现**：与作者一致的固定split、语义cue人工审核与阈值校准、SMR作者拟合权重、内部光学grade/外部组、逐seed配对统计、原论文曲线与日志归档。

**未实现的横向适配**：YOLOv8s、RT-DETR独立训练与权重转换；这不是本项目核心SCOA代码的运行验证范围。

完整论文对照表见 `docs/04_experiment_comparison.md`；`experiments/paper_comparison.json`的实测字段均为null，未混入合成数据。
