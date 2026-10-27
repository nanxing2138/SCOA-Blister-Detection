# 模型权重

[yolox_s.pth](yolox_s.pth) 为 YOLOX-s 权重文件，大小为 72,089,125 字节（约 68.75 MiB）。该检查点含 80 类分类输出，当前项目将其用于训练初始化；加载时按参数名称和形状匹配，四类分类头由训练过程学习。

默认配置：

```yaml
train:
  pretrained: weights/yolox_s.pth
```

评价与推理使用本项目训练产生的 `runs/<实验名>/seed_<种子>/best.pth`，其中保存模型配置、EMA 参数和选中轮次。`yolox_s.pth` 通过 `train.pretrained` 加载，不传给评价／推理的 `--checkpoint` 参数。

SHA256：

```text
f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30
```

文件来源与校验记录见 [yolox_s.sha256.json](yolox_s.sha256.json)；上游发布地址为 [YOLOX 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)。
