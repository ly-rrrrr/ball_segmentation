# 05 CNN v3 原型分类路线

CNN v3 把每个 SAM 候选转换为固定尺寸的三通道 crop，通过分类网络和类原型共同判断候选。历史标注包含 `real_ball`、`shadow_ball`、`double_ball_cluster`、`shadow_double_ball_cluster`、`single_ball_background_mixed` 与 `interference`；具体训练类别可由参数选择。

已训练权重位于 [Google Drive v3/v4.1 权重文件夹](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link)。

```powershell
python methods/05_cnn_v3/export_combined_manual_cnn_dataset_upscaled_fast.py `
  --summary outputs/example_sam/summary.json `
  --source-pair annotation/labels/example.csv:data/example.bmp `
  --output-dir work/cnn_v3_dataset

python methods/05_cnn_v3/train_cnn_prototype_classifier.py `
  --labels-csv work/cnn_v3_dataset/labels.csv --output-dir checkpoints/cnn_v3

python methods/05_cnn_v3/apply_cnn_prototype_classifier.py `
  --model checkpoints/cnn_v3/model.pt `
  --labels-csv work/inference/labels.csv --output-dir outputs/cnn_v3
```

`export_v3_scale_normalized_dataset.py` 记录了按参考目标尺寸归一化的实验。v3 能学习规则难以表达的纹理与上下文，但对倍率、背景和数据来源变化较敏感，因此后来发展出背景一致性和显式尺度特征路线。
