# Batch Workflows

这里的脚本串联多个方法阶段，因此不归属于单一方法目录：

- `batch_process_new_membrane_cnn_v3_upscaled.py`：固定 split36 的 SAM + CNN v3 工作流。
- `batch_process_cnn_v3_upscaled_dynamic.py`：可配置切分和多 worker 分片的 CNN v3 工作流。
- `batch_apply_v3_scale_normalized.py`：在已有 SAM summary 上导出尺度归一化 crop 并应用 v3 模型。

脚本默认使用当前 Python 解释器和仓库内的新方法路径。数据、SAM 权重和 CNN 权重应放在 `.gitignore` 已排除的位置，或通过参数传入。
