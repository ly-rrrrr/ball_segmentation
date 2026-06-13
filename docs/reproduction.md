# Reproduction Commands

以下命令从仓库根目录执行。需要外部资源的步骤以“外部”标记；脚本会在指定输出目录生成 summary、mask、crop、预测 CSV 和 overlay 等结果。

## 1. SAM 候选生成（外部：SAM 权重；建议 GPU）

官方模型：[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge)。下载完整模型目录到 `weights/sam/` 后运行：

```powershell
python methods/01_sam_generation/run_sam_automatic_mask.py `
  --model-path weights/sam `
  --image data/ball2k/1.bmp `
  --output-dir outputs/ball2k/1/sam_split9_upscale `
  --split 9 --split-upscale `
  --save-split-tiles --save-individual-masks --save-upscaled-masks `
  --device 0
```

效果：生成 tile、逐候选 mask、union、overlay 和记录候选坐标/分数/路径的 summary JSON。

## 2. 面积与形状过滤（外部：SAM summary 与 mask）

```powershell
python methods/02_area_shape_filter/filter_sam_masks_by_area.py `
  --reference-summary outputs/reference/sam_summary.json `
  --target-summary outputs/target/sam_summary.json `
  --output-dir outputs/target/area_shape `
  --shape-filter ball_or_cluster --combine-mode and
```

效果：拟合参考面积区间，计算几何特征，输出过滤后的 summary、union 和 overlay。

## 3. 种子启发式（外部：原图与 SAM 结果）

```powershell
python methods/03_seed_heuristic/filter_sam_masks_by_ball_seeds.py `
  --image data/example.bmp `
  --summary outputs/example/sam_summary.json `
  --output-dir outputs/example/seed_filter
```

效果：输出 seed map/overlay，并为每个候选记录种子命中、边界支持、高分回退及最终保留原因。

## 4. Learned seed（外部：干净参考与目标 SAM 结果）

```powershell
python methods/04_learned_seed/filter_sam_masks_by_learned_standard_seeds.py `
  --clean-image data/clean.bmp --clean-summary outputs/clean/sam_summary.json `
  --target-image data/target.bmp --target-summary outputs/target/sam_summary.json `
  --output-dir outputs/target/learned_seed
```

将入口替换为 `filter_sam_masks_by_learned_seed_arc_shape.py` 可运行弧段/形状增强版本。

## 5. CNN v3（外部：SAM 数据、人工标签、训练或已发布权重；训练建议 GPU）

已训练权重：[Google Drive v3/v4.1 权重文件夹](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link)。将 v3 checkpoint 放到 `checkpoints/cnn_v3/cnn_prototype_model.pt`。

```powershell
python methods/05_cnn_v3/export_combined_manual_cnn_dataset_upscaled_fast.py `
  --summary outputs/a8638/sam_summary.json `
  --source-pair a8638:annotation/candidates/a8638_split36_area_mid_candidates.json:annotation/labels/a8638_split36_area_mid_manual_labels.csv `
  --output-dir work/cnn_v3_dataset

python methods/05_cnn_v3/train_cnn_prototype_classifier.py `
  --labels-csv work/cnn_v3_dataset/labels.csv `
  --output-dir checkpoints/cnn_v3

python methods/05_cnn_v3/apply_cnn_prototype_classifier.py `
  --model checkpoints/cnn_v3/cnn_prototype_model.pt `
  --labels-csv work/cnn_v3_inference/labels.csv `
  --image data/target.bmp --summary outputs/target/sam_summary.json `
  --output-dir outputs/target/cnn_v3
```

效果：导出三通道 crop，训练分类器与类别原型，并输出候选概率、保留/拒绝 overlay 和统计 summary。

## 6. CNN v4.1（外部：2K 原图/SAM masks、训练或已发布权重；训练建议 GPU）

已训练权重：[Google Drive v3/v4.1 权重文件夹](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link)。将 v4.1 checkpoint 放到 `checkpoints/cnn_v41/cnn_scale_aware_model.pt`。

```powershell
python methods/06_cnn_v4_v41/export_2k_manual_cnn_dataset_v4_from_candidates.py `
  --candidates-json annotation/candidates/ball2k_six_class_candidates.json `
  --labels-csv annotation/labels/ball2k_six_class_manual_labels.csv `
  --output-dir work/cnn_v41_dataset

python methods/06_cnn_v4_v41/train_cnn_scale_aware_classifier_v41.py `
  --labels-csv work/cnn_v41_dataset/labels.csv `
  --output-dir checkpoints/cnn_v41

python methods/06_cnn_v4_v41/apply_cnn_scale_aware_classifier_v41.py `
  --model checkpoints/cnn_v41/cnn_scale_aware_model.pt `
  --labels-csv work/cnn_v41_inference/labels.csv `
  --image data/ball2k/1.bmp --summary outputs/ball2k/1/sam_summary.json `
  --output-dir outputs/ball2k/1/cnn_v41
```

效果：导出六分类数据，训练包含背景一致性和尺度特征的模型，并生成最终类别 overlay 与过滤结果。

## 7. 批处理

```powershell
python scripts/batch_process_cnn_v3_upscaled_dynamic.py `
  --input-root data/new_membrane `
  --output-root outputs/new_membrane `
  --sam-model weights/sam `
  --cnn-model checkpoints/cnn_v3/cnn_prototype_model.pt `
  --gpu 0
```

批处理会依次执行 SAM、推理数据导出和 CNN 应用，并为每个样本及 worker 写出汇总 JSON。
