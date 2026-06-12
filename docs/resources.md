# External Resources

Git 仓库只保存最小完整源码、人工标签、必要候选元数据和代表结果。下列资源计划使用公开 Google Drive 链接分发。当前尚未上传，因此链接与压缩包 SHA-256 标记为 `Pending upload`；上传后应先固定归档内容，再填写实际大小和校验值。

| Resource | Archive | Extract to | Size | SHA-256 | Google Drive | Used by |
|---|---|---|---:|---|---|---|
| 2K 原始 BMP（8 张） | `ball2k_images.zip` | `data/ball2k/` | 40.08 MiB local | Pending upload | Pending upload | annotation, CNN v4/v4.1 |
| 2K split9 + v4.1 完整结果 | `ball2k_v41_results.zip` | `outputs/ball2k_v41/` | 57.76 MiB local | Pending upload | Pending upload | viewer, result review |
| 完整 8 样本公开查看器资产 | `public_ball2k_viewer.zip` | `outputs/public_ball2k_viewer/` | 74.91 MiB local | Pending upload | Pending upload | visual comparison |
| 历史源码/人工脚本快照 | `v3_v41_source_manual_scripts.zip` | `archive/v3_v41_source_manual_scripts/` | 112.90 MiB local | Pending upload | Pending upload | provenance only |
| SAM 模型权重 | `sam_weights.zip` | `weights/sam/` | Pending upload | Pending upload | Pending upload | method 01 and all downstream routes |
| CNN v3 checkpoint | `cnn_v3_checkpoint.zip` | `checkpoints/cnn_v3/` | Pending upload | Pending upload | Pending upload | method 05 |
| CNN v4.1 checkpoint | `cnn_v41_checkpoint.zip` | `checkpoints/cnn_v41/` | Pending upload | Pending upload | Pending upload | method 06 |
| 导出的 CNN 训练数据 | `cnn_training_datasets.zip` | `work/` | Pending upload | Pending upload | Pending upload | methods 05-06 |

## 发布步骤

1. 从本地目录创建固定内容的 ZIP，不把缓存和重复批量输出混入归档。
2. 使用 `Get-FileHash <archive> -Algorithm SHA256` 记录 SHA-256。
3. 上传到 Google Drive，设置为“知道链接的任何人可查看”。
4. 将真实文件大小、SHA-256 和公开链接补入上表。
5. 在新环境下载并解压到 `Extract to` 指定位置。

当前本地资源不会因仓库整理而删除；`.gitignore` 只是阻止它们被 Git 跟踪。
