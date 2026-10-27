# 来源与范围

本项目以用户提供论文为研究规格，由Codex重建。没有取得作者未公开的研究代码、真实数据或实验档案；不表示作者审定的参考实现。

`third_party/YOLOX` 与 `scoa/_vendor` 的模型代码源自 Megvii-BaseDetection/YOLOX 0.3.0，遵循随附 Apache-2.0 LICENSE。保留原版权注释，适配记录见论文—代码映射与来源SHA256。

`source_evidence` 是本次用户提供文档的本地提取证据；项目压缩包供用户审查与运行，公开发布前应自行确认论文材料的分享权限。

源码包不包含隔离Python运行时、任何真实药品数据或合成训练checkpoint。当前检查版交付包包含官方COCO预训练权重及来源校验记录；权重来自YOLOX官方release，遵循其上游发布条款，不属于论文作者公开的药品检测模型。`validation_records` 包含明确标为synthetic/diagnostic的流程检查和COCO权重迁移检查；不可用作论文真实实验的替代证据。

公开仓库不包含本地论文提取材料 `source_evidence`、数据目录、训练输出、交付压缩包、隔离Python环境或权重二进制；官方预训练权重的获取方式见 `weights/README.md`。既有检查记录描述其实际执行范围，不代表论文全部实验已复现。
