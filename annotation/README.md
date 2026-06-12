# Annotation Assets

`labels/` 保存人工标签，`candidates/` 保存生成这些标签时使用的候选记录，`viewers/` 保存静态标注与结果查看器。候选 JSON/CSV 虽然比普通源码大，但它们是解释历史标签所必需的最小元数据。

旧标签不能直接附加到重新生成的候选上。只有确认 `candidate_id`、`tile_id` 和 `mask_index` 与原候选等价后，标签才可复用；SAM 版本、切分方式或候选排序发生变化都可能破坏这种对应关系。

生成 2K 标注候选：

```powershell
python annotation/build_2k_six_class_annotation_candidates.py `
  --image-root data/ball2k `
  --result-root outputs/ball2k `
  --output-json annotation/candidates/new_candidates.json `
  --output-csv annotation/candidates/new_candidates.csv
```

`normalize_candidate_paths.py` 可将候选 JSON 中位于项目目录内的绝对路径转换为便携路径。完整原图和掩膜不进入 Git，获取方式见 `docs/resources.md`。
