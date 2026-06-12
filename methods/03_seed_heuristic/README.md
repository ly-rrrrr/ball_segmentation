# 03 种子启发式过滤

该路线直接从灰度图中寻找“球心种子”：计算中心与环形邻域的径向对比度，按响应百分位筛选，再用 NMS 去重并检查径向平衡。SAM 掩膜只有覆盖足够种子时才保留；tile 边界附近允许邻近种子支持，极高 SAM 分数也可作为可配置的无种子回退。

```powershell
python methods/03_seed_heuristic/filter_sam_masks_by_ball_seeds.py `
  --image data/example.bmp `
  --summary outputs/example_sam/summary.json `
  --output-dir outputs/example_seed `
  --seed-percentile 97.5 --min-seeds-in-mask 1
```

输出包含 seed map、seed overlay、过滤后的 overlay/union 和带判定原因的 JSON。它比单纯面积规则更关注局部球状信号，但在低对比、阴影、粘连球和跨成像域场景中仍依赖手工阈值。这条路线是后续 learned-seed 与 CNN 方法的重要基础。
