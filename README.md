# ball_segmentation

基于 Segment Anything Model（SAM）的小球/颗粒候选分割与过滤项目。本仓库按方法演进保留从无训练规则到尺度感知 CNN 的完整技术路线，而不是只保留最终版本。

## 方法索引

| 方法 | 核心思想 | 需要训练 | 入口 | 状态 |
|---|---|---:|---|---|
| [01 SAM generation](methods/01_sam_generation/) | 切分、放大并生成候选掩膜 | 否 | `run_sam_automatic_mask.py` | 基础候选生成 |
| [02 Area/shape](methods/02_area_shape_filter/) | 面积分布与几何特征过滤 | 否 | `filter_sam_masks_by_area.py` | 历史规则路线 |
| [03 Seed heuristic](methods/03_seed_heuristic/) | 原图球心种子与 SAM 候选互证 | 否 | `filter_sam_masks_by_ball_seeds.py` | 历史启发式路线 |
| [04 Learned seed](methods/04_learned_seed/) | 从干净参考图学习种子外观，再结合弧段/形状 | 参考样本拟合 | 两个 learned-seed 脚本 | 过渡路线 |
| [05 CNN v3](methods/05_cnn_v3/) | 三通道候选 crop + 原型分类 | 是 | export/train/apply | 已保留 |
| [06 CNN v4/v4.1](methods/06_cnn_v4_v41/) | 背景一致性与显式尺度特征 | 是 | export/train/apply | 当前完整学习路线 |

详细演进与负面实验结论见 [docs/method_history.md](docs/method_history.md)。

## 仓库内容

```text
methods/       六条方法路线的源码与说明
annotation/    人工标签、标签绑定的候选元数据和静态查看器
scripts/       串联多个方法的批处理脚本
examples/      少量代表性输入与结果
docs/          方法历史、资源清单和复现命令
```

完整原始数据、逐 mask 输出、训练数据集和模型权重不提交到 Git。公开 Google Drive 资源尚待上传，目标目录和状态记录在 [docs/resources.md](docs/resources.md)。

## 安装

建议使用 Python 3.10+。请按本机 CUDA 环境安装合适的 PyTorch，再安装项目：

```powershell
python -m pip install -e .
```

## 快速查看

- 基础方法结果：[examples/basic](examples/basic/)
- CNN v4.1 两个代表样本：[examples/cnn_v41/ball2k_result_viewer.html](examples/cnn_v41/ball2k_result_viewer.html)
- 完整分阶段命令：[docs/reproduction.md](docs/reproduction.md)

## 标注注意事项

人工标签依赖生成时的候选 ID。重新运行 SAM 后，不得仅按行号或视觉位置复用旧标签；必须验证 `candidate_id`、`tile_id` 和 `mask_index` 等价。参见 [annotation/README.md](annotation/README.md)。

## GitHub 同步

当前远程仓库为 [ly-rrrrr/ball_segmentation](https://github.com/ly-rrrrr/ball_segmentation)。整理过程采用普通提交，不重写历史；本地大文件继续保留，但由 `.gitignore` 排除。
