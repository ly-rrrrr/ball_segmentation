# External Resources

Git 仓库只保存最小完整源码、人工标签、必要候选元数据和代表结果。SAM 使用 Hugging Face 官方模型仓库；CNN v3 与 v4.1 权重已经通过同一个 Google Drive 文件夹共享。其余资源仍以 `Pending upload` 标记。文件夹内权重的准确文件名、大小和 SHA-256 可在归档固定后补充。

| Resource | Archive/source | Extract to | Size | SHA-256 | Download | Used by |
|---|---|---|---:|---|---|---|
| 2K 原始 BMP（8 张） | `ball2k_images.zip` | `data/ball2k/` | 40.08 MiB local | Pending upload | Pending upload | annotation, CNN v4/v4.1 |
| 2K split9 + v4.1 完整结果 | `ball2k_v41_results.zip` | `outputs/ball2k_v41/` | 57.76 MiB local | Pending upload | Pending upload | viewer, result review |
| 完整 8 样本公开查看器资产 | `public_ball2k_viewer.zip` | `outputs/public_ball2k_viewer/` | 74.91 MiB local | Pending upload | Pending upload | visual comparison |
| 历史源码/人工脚本快照 | `v3_v41_source_manual_scripts.zip` | `archive/v3_v41_source_manual_scripts/` | 112.90 MiB local | Pending upload | Pending upload | provenance only |
| SAM ViT-H 官方权重 | `facebook/sam-vit-huge` | `weights/sam/` | Hosted by Hugging Face | Upstream managed | [Hugging Face](https://huggingface.co/facebook/sam-vit-huge) | method 01 and all downstream routes |
| CNN v3 checkpoint | Google Drive folder | `checkpoints/cnn_v3/` | To be recorded | To be recorded | [v3/v4.1 weights](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link) | method 05 |
| CNN v4.1 checkpoint | Google Drive folder | `checkpoints/cnn_v41/` | To be recorded | To be recorded | [v3/v4.1 weights](https://drive.google.com/drive/folders/1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu?usp=drive_link) | method 06 |
| 导出的 CNN 训练数据 | `cnn_training_datasets.zip` | `work/` | Pending upload | Pending upload | Pending upload | methods 05-06 |

## 发布步骤

1. 从本地目录创建固定内容的 ZIP，不把缓存和重复批量输出混入归档。
2. 使用 `Get-FileHash <archive> -Algorithm SHA256` 记录 SHA-256。
3. 上传到 Google Drive，设置为“知道链接的任何人可查看”。
4. 将真实文件大小、SHA-256 和公开链接补入上表。
5. 在新环境下载并解压到 `Extract to` 指定位置。

当前本地资源不会因仓库整理而删除；`.gitignore` 只是阻止它们被 Git 跟踪。

## 权重放置

- SAM：从 `facebook/sam-vit-huge` 下载完整 Hugging Face 模型目录，放入 `weights/sam/`，确保其中包含配置、processor 配置和模型权重文件。
- CNN v3：从共享文件夹下载 v3 checkpoint，放入 `checkpoints/cnn_v3/`；默认命令期望文件名为 `cnn_prototype_model.pt`。
- CNN v4.1：从共享文件夹下载 v4.1 checkpoint，放入 `checkpoints/cnn_v41/`；默认命令期望文件名为 `cnn_scale_aware_model.pt`。
