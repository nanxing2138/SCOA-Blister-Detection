# 论文与代码对应表

| 论文位置 | 文件/主要函数 | 输入→输出及约束 |
|---|---|---|
| §4.1标注 | `scoa/data.py:CocoDataset` | COCO xywh、ID1–4 → HWC BGR与[N,5] xyxy+零基class |
| §4.2预处理 | `scoa/data.py:letterbox` | 任意H×W → 左上角s×s，框按同一比例缩放，返回valid矩形 |
| 式1–7 | `scoa/optics.py:draw_pool,glare_at` | [0,1] HWC+valid+独立RNG → 3个候选、公共draw及解析光斑 |
| 式8–9 | `scoa/structure.py:photometric` | 原图、候选、缺陷框、幅度 → 逐框新过曝比例及中心增量 |
| 式10–12 | `scoa/structure.py:reference_regions,structure_scores` | 仅原图定义tile；总体方差/协方差 → 各候选各缺陷Q值 |
| Algorithm1 | `scoa/selection.py:Selector.select` | 原图和labels→选中HWC、审计record；空集/异常回退 |
| SimOTA | `scoa/difficulty.py:FixedAssignmentScorer`，`scoa/_vendor/yolo_head.py:get_assignments` | 原图[B,8400,9] → 固定fg位置、GT索引 |
| 式13 | `scoa/difficulty.py:instance_losses` | 固定位置+每候选decoded boxes/logits → [N]实例损失 |
| 式14–15 | `scoa/difficulty.py:difficulty` | 实例损失差 → 类内/类间平均；可选实例平均对照 |
| 式16 | `scoa/model.py:ModelEMA` | 每次成功optimizer更新后更新FP32浮点state，保存updates |
| YOLOX-s | `scoa/model.py:build_model`，`scoa/_vendor/` | 输入[B,3,s,s] → stride8/16/32 head；上游0.3.0实装 |
| §3.5 CBAM | `scoa/model.py:CBAM,build_model` | 8个CSP的m顺序后添加attention，保留原始checkpoint键 |
| 式17/WIoU | `scoa/model.py:WIoU3`，上游head.get_losses | [P,4] cxcywh→逐正样本loss；全batch按num_fg归一化 |
| 式18 | `scoa/postprocess.py:suppress,postprocess` | decoded predictions → [K,6] xyxy,score,class，K≤100 |
| §4.2训练 | `scoa/train.py:train` | COCO train→SGD+AMP+EMA；验证AP50选权重，记录随机状态 |
| §4.4评价 | `scoa/metrics.py:evaluate_records` | 640帧标注与预测→101点AP50/AP75/AP50:95+缺陷microP/R |
| §4.5计时 | `scoa/benchmark.py` | 200warmup/1000timed×3，CPU同步端到端与CUDA events前向分开 |
| §4.6对照 | `scoa/selection.py`、`scoa/augment.py` | S1/S2/S3，SMR-B权重采样，D-instance，AM-photo与legacy |
| §4.7 AugMax-P | `scoa/augment.py:augmax` | 3固定链+simplex投影+5次上升，冻结student/固定匹配 |
| 式19 cue统计 | `tools/audit_cues.py` | 人工真标签CSV→接受cue loss/拒绝discernible；零分母null |
| 式21–22 | `tools/compare_results.py` | 真实逐seedJSON→样本SD、同seed配对差；不从均值伪造seed |

`scoa/_vendor`中6个模型文件取自官方YOLOX 0.3.0，唯一接口修改是head的两个utils导入指向本地最小依赖实现；完整下载源码与许可证存于third_party。项目自有模块是研究重建，不能称作者公开实现。

后续完整性检查增加 `scoa/model.py:StableYOLOXHead/StableSquaredIoULoss`，以子类保持训练解码和平方IoU计算为FP32；`scoa/geometry.py:bboxes_iou`也用FP32面积计算。它们不修改上述6个vendor文件或损失公式。`scoa/checkpoints.py`统一可信checkpoint加载；评价输出新增实际image-ID签名以防不同子集被混合比较。详见 `docs/06_completeness_audit.md`。
