# ball_segmentation

基于 Segment Anything Model（SAM）的小球/颗粒候选分割与过滤工具集，面向显微图、膜面图像及相似的小目标分割任务。

仓库保留了从面积/形状规则、种子启发式，到 CNN v3、尺度感知 CNN v4.1 的多种技术方案。除 SAM 候选生成通常作为上游输入外，其余方法是可以独立选择或组合的过滤方案，并不是必须按某个编号依次执行的流水线。

## 结果预览

- [基础方法代表结果](examples/basic/)：SAM 候选、面积/形状过滤和种子启发式输出。
- [CNN v4.1 结果查看器](examples/cnn_v41/ball2k_result_viewer.html)：高密度与低密度两个 2K 代表样本的原图、SAM 与最终结果对比。
- [方法演进与实验结论](docs/method_history.md)：包括历史路线、局限和负面实验结果。

## 安装与权重

建议使用 Python 3.10+。请先根据本机 CUDA 环境安装合适的 PyTorch，然后安装项目依赖：

```powershell
python -m pip install -e .
```

需要的模型资源：

- SAM ViT-H：[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge)，下载完整 Hugging Face 模型目录至 `weights/sam/`。
- CNN v3 / v4.1：[Google Drive 权重文件夹](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link)。
- v3 默认位置：`checkpoints/cnn_v3/cnn_prototype_model.pt`。
- v4.1 默认位置：`checkpoints/cnn_v41/cnn_scale_aware_model.pt`。

完整数据和其他外部资源见 [资源清单](docs/resources.md)。

## 最短完整流程

下面演示从一张 BMP 图像生成 SAM 候选，再使用已训练的 CNN v4.1 权重进行过滤。命令从仓库根目录执行。

### 生成 SAM 候选

```powershell
python methods/01_sam_generation/run_sam_automatic_mask.py `
  --model-path weights/sam `
  --image data/example.bmp `
  --output-dir outputs/example_sam `
  --split 9 --split-upscale `
  --save-split-tiles --save-individual-masks --save-upscaled-masks `
  --device 0
```

主要输出：

- `outputs/example_sam/example_sam_auto_summary.json`
- `outputs/example_sam/example_sam_auto_overlay.png`
- tile 图像、逐候选 mask 和 upscaled mask

### 导出 v4.1 推理数据

```powershell
python methods/06_cnn_v4_v41/export_split_upscale_cnn_inference_dataset.py `
  --summary outputs/example_sam/example_sam_auto_summary.json `
  --output-dir work/example_cnn_v41
```

该步骤会生成 CNN 所需的 crop、mask、masked crop 和 `work/example_cnn_v41/labels.csv`。

### 应用 CNN v4.1

```powershell
python methods/06_cnn_v4_v41/apply_cnn_scale_aware_classifier_v41.py `
  --model checkpoints/cnn_v41/cnn_scale_aware_model.pt `
  --labels-csv work/example_cnn_v41/labels.csv `
  --image data/example.bmp `
  --summary outputs/example_sam/example_sam_auto_summary.json `
  --output-dir outputs/example_cnn_v41 `
  --keep-threshold 0.75
```

主要输出包括候选分类概率 CSV、保留/拒绝 overlay 和 `cnn_scale_aware_apply_summary.json`。训练、标注和其他方法的完整命令见 [复现文档](docs/reproduction.md)。

## 可用方法

| 方法 | 适用场景 | 是否训练 | 入口与说明 |
|---|---|---:|---|
| SAM candidate generation | 从原图生成候选 mask；支持切分、tile 放大和拼接 | 否 | [SAM 候选生成](methods/01_sam_generation/) |
| Area and shape filtering | 尺寸和形态较稳定、需要强可解释性的场景 | 否 | [面积与形状过滤](methods/02_area_shape_filter/) |
| Seed heuristic filtering | 原图存在可检测球心局部对比，需要与 SAM 互证 | 否 | [种子启发式过滤](methods/03_seed_heuristic/) |
| Learned-seed filtering | 有干净参考样本，希望学习局部种子外观 | 参考样本拟合 | [Learned-seed 方法](methods/04_learned_seed/) |
| CNN v3 | 使用人工标签学习候选纹理、mask 和上下文 | 是 | [CNN v3 原型分类](methods/05_cnn_v3/) |
| CNN v4 / v4.1 | 跨倍率、跨背景场景；需要背景一致性和尺度特征 | 是 | [CNN v4/v4.1 尺度感知分类](methods/06_cnn_v4_v41/) |

方法目录名称中的数字仅用于稳定排序和保持现有链接，不代表这些方案具有强制执行顺序。

## 标注与外部资源

`annotation/` 保存人工标签、与标签绑定的候选元数据和静态查看器。人工标签依赖生成时的候选 ID；重新运行 SAM 后，不能仅按行号或视觉位置复用旧标签，必须验证 `candidate_id`、`tile_id` 和 `mask_index` 等价。

- [标注资产说明](annotation/README.md)
- [模型、数据与 Google Drive 资源](docs/resources.md)
- [完整复现命令](docs/reproduction.md)

大体积原始数据、逐 mask 结果、训练数据集和模型权重不提交到 Git。仓库只保留源码、人工标签、必要候选元数据和少量代表结果。

## 仓库结构

```text
methods/       各种可选方法的源码与独立说明
annotation/    人工标签、候选元数据、标注工具和查看器
scripts/       串联多个方法的批处理脚本
examples/      少量代表性输入与结果
docs/          复现命令、资源清单、方法历史和设计记录
```

## 进一步阅读

- [完整复现与训练命令](docs/reproduction.md)
- [外部资源与权重放置](docs/resources.md)
- [方法历史与实验结论](docs/method_history.md)
- [批处理工作流](scripts/README.md)
- [标注数据约束](annotation/README.md)

## License

本项目使用 [Apache License 2.0](LICENSE)。
