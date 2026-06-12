# 01 SAM 候选掩膜生成

这一阶段使用 Hugging Face Transformers 的 SAM automatic-mask-generation pipeline 生成候选掩膜。脚本支持整图推理、规则网格切分、tile 放大后推理、掩膜回缩拼接，以及单掩膜、并集和 JSON 摘要导出。

```powershell
python methods/01_sam_generation/run_sam_automatic_mask.py `
  --model-path weights/sam `
  --image data/example.bmp `
  --output-dir outputs/example_sam `
  --split 36 --split-upscale --save-individual-masks --save-upscaled-masks
```

输入为本地图像和 SAM 权重目录；输出包括 overlay、union mask、mask records 与可选的 tile/mask 文件。切分放大有利于小球召回，但会增加计算量并产生重复、边界截断和背景伪候选，因此后续各路线都把它视为候选生成器，而不是最终判定器。

这是项目所有方法的共同起点。
