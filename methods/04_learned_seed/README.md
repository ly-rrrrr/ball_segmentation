# 04 Learned-Seed 方法

这一阶段保留两种从干净参考图中学习种子外观的尝试：

- `filter_sam_masks_by_learned_standard_seeds.py` 从参考掩膜收集局部径向特征，学习标准种子分布，再扫描目标图。
- `filter_sam_masks_by_learned_seed_arc_shape.py` 在 learned seed 上叠加面积、形状、边界弧段与多球几何判定，并包含可选的单球和双球救回逻辑。

```powershell
python methods/04_learned_seed/filter_sam_masks_by_learned_standard_seeds.py `
  --clean-image data/clean.bmp --clean-summary outputs/clean_sam/summary.json `
  --target-image data/target.bmp --target-summary outputs/target_sam/summary.json `
  --output-dir outputs/target_learned_seed
```

输入仍是图像与 SAM summary。这里“学习”的是从参考正样本估计的局部特征模型，并非端到端神经网络；掩膜聚合、阈值、面积和形状判定仍是启发式。该路线连接了纯规则方法与后续 CNN 分类器，保留它有助于理解哪些误差来自种子检测，哪些来自候选级分类。
