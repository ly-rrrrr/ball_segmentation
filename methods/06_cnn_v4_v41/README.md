# 06 CNN v4 / v4.1 尺度感知路线

v4 阶段研究背景一致性：对候选外部背景施加扰动，并约束预测在背景变化下保持稳定，以减轻模型记忆特定膜面或采集条件的问题。仓库保留的完整实现是 v4.1，它进一步加入候选面积、包围盒尺寸、长宽比等显式尺度特征，并使用 2K 六分类人工标注。

已训练权重位于 [Google Drive v3/v4.1 权重文件夹](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link)。

```powershell
python methods/06_cnn_v4_v41/export_2k_manual_cnn_dataset_v4_from_candidates.py `
  --candidates-json annotation/candidates/ball2k_six_class_candidates.json `
  --labels-csv annotation/labels/ball2k_six_class_manual_labels.csv `
  --output-dir work/cnn_v41_dataset

python methods/06_cnn_v4_v41/train_cnn_scale_aware_classifier_v41.py `
  --labels-csv work/cnn_v41_dataset/labels.csv --output-dir checkpoints/cnn_v41

python methods/06_cnn_v4_v41/apply_cnn_scale_aware_classifier_v41.py `
  --model checkpoints/cnn_v41/model.pt `
  --labels-csv work/inference/labels.csv --output-dir outputs/cnn_v41
```

该路线同时需要外部原图/SAM 候选数据和训练权重；权重已发布，其他资源状态见 `docs/resources.md`。v4.1 是当前最完整的学习式路线，但仍受人工标签覆盖、候选生成召回率和跨域数据量限制。
