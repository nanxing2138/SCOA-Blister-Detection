# 代码完整性检查与迁移报告

检查日期：2026-10-27。当前项目根目录：`D:\yaopindaima`。

## 结论

**核心SCOA/YOLOX项目已形成可运行闭环，当前44项测试通过；不能把它称为论文所有实验已经完整复现。** 源码、正式/合成配置、依赖、官方COCO权重、训练/评价/推理入口、README、许可证和检查记录均已保存在新目录。

没有真实数据、作者固定split ID与实验档案，因此真实AP、人工cue审核、外部条件结果和完整CUDA时延仍无法核验。YOLOv8s/RT-DETR独立横向训练适配、完整审核阈值校准及作者归档的全部曲线不属于当前已完成范围。

## 本次检查发现并修复的事项

| 项目 | 发现的问题 | 修复与边界 |
|---|---|---|
| AMP数值稳定 | FP16下640尺度框的面积可超过65504，IoU面积/并集会溢出；训练解码也曾保留半精度 | `StableYOLOXHead`在训练解码时转FP32；平方IoU及匹配IoU的面积计算用FP32。仍是原1−IoU²及原匹配逻辑，未改六个上游模型源码文件 |
| AugMax异常回退 | 原图评分非有限时会保留初始随机混合，而论文要求非有限搜索回原图 | 非有限原图/搜索统一返回原图；空标注/无匹配保留初始混合。新增实际非有限网络输出回归测试 |
| 配置检查 | 仅校验少数字段，概率越界、NaN、错误枚举和YAML拼写可晚到训练阶段才出错 | 增加字段、范围、枚举、warmup/末尾阶段、SMR幅度及extends循环检查 |
| 数据标识 | 重复annotation ID未拒绝，file_name路径边界不明确 | 拒绝重复标注ID；统一路径解析，限制在image_dir内；检查工具使用相同解析方式 |
| checkpoint格式 | 把COCO初始化权重传给推理会得到难定位的KeyError；新旧torch.load默认行为不同 | 统一可信文件加载，显式检查config/ema/epoch；错误直接说明COCO文件应给train.pretrained |
| 续训资源/输出 | 恢复时先把整份checkpoint加载到GPU，增加显存占用；新输出目录可能缺失先前best | 在CPU加载后复制到模型；恢复新目录时保留best，拒绝覆盖更晚的已有latest；已有日志也阻止无意重新训练覆盖 |
| 续训可验证性 | 此前只验证保存/加载，没有实际比较连续训练与恢复后的状态 | 增加不改总计划的`--stop-after-epoch`以及`tools/check_resume.py`；集成模型按epoch恢复已精确验证 |
| 结果对照 | 同一标注文件的不同子集、集成模型不同NMS可被混入同一论文参考 | 评价记录实际图像ID及签名、NMS阈值和训练组件；汇总核对这些信息。新增`integrated-standard`单独对应73.78参考 |
| 计时入口 | 零次计时未提前拒绝，多GPU编号的CUDA event未显式记录目标设备流 | 校验迭代参数，事件记录在目标流；CPU短流程已运行，GPU分支仍未实测 |
| 依赖与分发 | pip元数据缺少核心依赖声明，vendor许可证未单独放入安装包；打包全局忽略data目录会省去上游同名源码子目录 | 补充核心依赖和可选测试/评价依赖，打包vendor LICENSE；仅在项目根排除运行数据/输出，保留上游data源码 |
| 审核工具 | 输出父目录未自动建立，人工审核类别/空实体标识缺少验证 | 增加明确校验与目录创建，不自动生成任何语义判断 |

FP32适配是数值实现修复，不等于已经验证论文GPU训练组合。CUDA11.6/torch1.12.1完整300轮仍需目标机器和真实数据验证。

## 实际执行的验证

全部在新目录中执行，使用项目内部隔离的Python3.12/torch2.2.2+cpu；未查询用户GPU型号、显存或CUDA配置。

| 检查 | 实际结果 | 记录 |
|---|---|---|
| 完整测试集 | **44 passed**，65.07秒 | `validation_records/audit_20261027/tests.xml` |
| 上游源码一致性 | 6个模型文件通过原始SHA256及唯一允许导入差异检查 | `tools/verify_vendor.py` |
| 核心/集成合成闭环 | 10个命令全部exit0；含两种模型的训练与checkpoint评价、核心单图推理 | `validation_records/audit_20261027/smoke/execution.json` |
| 官方权重 | 两种模型各456个匹配tensor加载、6个源分类tensor跳过；640×640输出[1,8400,9]且有限 | `validation_records/audit_20261027/pretrained_verification.json` |
| COCOapi对照 | 固定诊断样例最大AP差1.1102230246251565e−16 | `validation_records/audit_20261027/coco_crosscheck.json` |
| 实际续训 | CBAM+WIoU+SCOA，连续2轮与1轮+恢复1轮；model、ema、optimizer、scaler、更新数、global_step、best AP50、torch RNG逐项精确相等 | `validation_records/audit_20261027/resume/result.json` |
| Windows数据加载 | 上述续训检查使用workers=1，各自成功4次优化/EMA更新 | 同上 |
| 计时入口 | CPU、64输入、warmup1/迭代2/1轮实际完成；明确不是论文计时协议 | `validation_records/audit_20261027/benchmark_smoke.json` |
| 组件实验清单 | 生成24个训练身份、每个复用2种NMS评价命令；未启动正式作业 | `runs/audit_20261027/component_matrix/` |
| 安装入口 | `pip install --no-deps -e .`在本地隔离环境成功构建并安装 | 本次工具实际退出码0 |

本次合成AP仍只用于流程检查，不进入真实论文结果对照。旧版缺少image-ID签名的评价JSON需重新评价，不能用手工补签名替代实际输出。

## 按范围判断“是否完全”

| 范围 | 实现状态 | 验证程度 |
|---|---|---|
| YOLOX-s/SimOTA/真实训练损失 | 已实现，上游源码保留 | 640前向、小输入真实反向/更新、空标注和L1通过 |
| SCOA共享候选/光度/结构/类别平衡EMA | 已实现 | 数学、状态、维度、异常回退、合成训练通过 |
| CBAM/WIoU v3/DIoU-NMS | 已实现 | 权重迁移、前反向、集成闭环和状态恢复通过 |
| A/B/C/D、S1/S2、D-instance | 已实现 | 核心与统计/分支路径已检查，未完成每组300轮真实训练 |
| AM-photo/AugMax-P/legacy/SMR-B | 已实现重建版本 | AugMax真实输入梯度搜索、SMR拟合等检查通过；作者精确实现/权重不可得 |
| 训练/保存/按epoch恢复/评价/推理 | 已实现 | 新目录闭环与CPU精确续训对照通过 |
| COCO初始化权重 | 已下载并接入 | 大小72,089,125 bytes，来源/SHA256与真实加载均核对 |
| 真实固定split和三seedAP | 缺材料 | 未运行、未验证，结果保留null |
| 人工cue审核、外部条件和作者曲线 | 工具/接口部分具备，档案缺失 | 不可声称已复现 |
| YOLOv8/RT-DETR横向训练 | 未提供独立适配器 | 未实现/未验证 |
| 参考CUDA全程、生产线吞吐 | 尚未验证 | CPU小流程和文中9.10ms不能替代正式GPU/产线测试 |

参数统计差异仍保留：核心8,938,843、集成8,964,445，与论文8.910M/8.936M不一致。没有作者统计脚本/代码，当前不能确定原因。

## 在另一台电脑复查

先按README安装匹配的CPU或CUDA依赖，然后在项目根目录依次运行：

```text
python -m pytest -q
python tools/run_smoke.py --output runs/my_check --skip-tests
python tools/verify_pretrained.py --output runs/my_check/pretrained.json
python tools/check_resume.py --output runs/my_resume --workers 1
```

完整训练仍需要用户提供真实COCO标注与固定split。`.validation_env`是本次本地检查用目录，不视为可跨电脑复制使用的依赖环境；交付ZIP排除此目录、运行数据和合成checkpoint，包含源码、配置、README、官方权重和验证证据。

## 迁移与清理

迁移前先复制，再对源码、权重及历史记录289个文件核对SHA256；全部初始文件保留，修复只发生在新目录。原始清单见 `validation_records/audit_20261027/migration_before_audit.json`。

新目录及临时交付包核验后，旧项目目录和3份旧ZIP/对应校验文件已全部移入回收站，原路径均已清空。永久删除曾被工具安全策略拦截，因此采用可恢复的回收站方式完成清理；没有清空回收站。实际记录见 `validation_records/audit_20261027/cleanup.json`。原始论文DOCX和“药品”父目录中的其它论文/代码工程保留。
