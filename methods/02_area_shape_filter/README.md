# 02 面积与形状过滤

该路线用参考样本拟合合理面积区间，并结合圆度、长宽比、extent、solidity、convexity、圆形 IoU 和边界接触等几何特征过滤 SAM 候选。它不需要训练数据，适合尺寸和成像条件较稳定的场景。

```powershell
python methods/02_area_shape_filter/filter_sam_masks_by_area.py `
  --reference-summary outputs/reference_sam/summary.json `
  --target-summary outputs/target_sam/summary.json `
  --output-dir outputs/target_area `
  --shape-filter ball_or_cluster --combine-mode and
```

输入是 SAM summary 和其中引用的掩膜文件；输出为过滤后的 summary、overlay 与 union mask。优点是可解释、无需权重；局限是阈值容易随倍率、球尺寸、聚团形态和背景域变化。这是 SAM 之后最早的规则过滤路线。
