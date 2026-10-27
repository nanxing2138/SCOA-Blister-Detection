# 官方COCO初始化权重

`yolox_s.pth` 来自 Megvii-BaseDetection/YOLOX 官方release 0.1.1rc0，也是YOLOX0.3.0 README为标准YOLOX-s列出的文件。对应80类COCO预训练，供本项目四类药品检测训练初始化使用。

来源：https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0

文件链接：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

`yolox_s.sha256.json`记录下载来源、官方asset ID/大小及本地SHA256。官方旧release没有公布digest，不能将本地SHA256说成作者公布的校验值；论文也没有给出其预训练文件hash。

正式配置默认读取 `weights/yolox_s.pth`。80→4类时仅加载形状匹配的权重，3尺度分类预测层的weight/bias重新初始化；新增CBAM及WIoU状态也单独初始化。它不是已训练的药品缺陷识别模型，不能直接替代 `scoa.infer` 需要的本项目训练checkpoint。

运行 `python tools/verify_pretrained.py` 可在CPU上复核SHA256、基础/集成网络加载与前向。
