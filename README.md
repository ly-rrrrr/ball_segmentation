# ball_segmentation

`ball_segmentation` 是一个基于 Segment Anything Model（SAM）的球状目标/颗粒掩码分割项目。项目当前包含两条主要流程：

1. 使用本地 Hugging Face SAM 模型对显微/膜面图像进行自动掩码生成。
2. 根据参考图像的掩码面积分布，以及可选的圆度、长宽比、填充率、凸性等形状特征，对目标图像中的候选掩码进行二次筛选，并输出合并掩码和可视化叠加图。

项目适合用于对膜面图像中的球状颗粒、孔洞或相似小目标进行批量分割与筛选。当前仓库只提交代码、原始示例图和少量代表性分割结果图；完整的逐 mask PNG 与大体积 summary JSON 可按本文命令重新生成。

## 项目结构

```text
ball_segmentation/
├── demo/
│   ├── *.bmp
│   ├── *_sam_auto_split36_upscale/
│   │   ├── *_sam_auto_union.png
│   │   └── *_sam_auto_overlay.png
│   └── *_area_filtered_from_858_split36/
│       ├── *_area_filtered_union.png
│       └── *_area_filtered_overlay.png
└── tools/
    ├── run_sam_automatic_mask.py
    └── filter_sam_masks_by_area.py
```

## 环境依赖

建议使用 Python 3.10+，并准备一个本地 SAM 模型目录。脚本直接从本地目录加载 Hugging Face 模型，不会在命令中指定在线下载。

主要 Python 依赖：

```bash
pip install numpy pillow torch transformers
```

如果使用 GPU，请确保 PyTorch 与本机 CUDA 版本匹配。

## 1. 运行 SAM 自动分割

入口脚本：

```bash
python tools/run_sam_automatic_mask.py --help
```

基本命令：

```bash
python tools/run_sam_automatic_mask.py \
  --model-path /path/to/local/sam-model \
  --image demo/858f3853776825b04e3608e2442d904a.bmp \
  --output-dir demo/858f3853776825b04e3608e2442d904a_sam_auto_split36_upscale \
  --split 36 \
  --split-upscale \
  --save-individual-masks \
  --save-upscaled-masks \
  --union-max-area-ratio 0.9 \
  --points-per-batch 256 \
  --device 0
```

Windows PowerShell 示例：

```powershell
python tools/run_sam_automatic_mask.py `
  --model-path D:\models\sam `
  --image demo\858f3853776825b04e3608e2442d904a.bmp `
  --output-dir demo\858f3853776825b04e3608e2442d904a_sam_auto_split36_upscale `
  --split 36 `
  --split-upscale `
  --save-individual-masks `
  --save-upscaled-masks `
  --union-max-area-ratio 0.9 `
  --points-per-batch 256 `
  --device 0
```

执行后的效果：

- 将输入图像切成 `6 x 6` 共 36 个 tile。
- 在 `--split-upscale` 模式下，每个 tile 会先放大到原图尺寸再送入 SAM，以增强小目标的自动分割能力。
- 输出每个候选掩码的二值图、可选的 SAM 输入尺度掩码、合并后的 union 掩码、叠加可视化图和 JSON 统计文件。
- 完整运行后，summary JSON 会记录 `split_tiles_upscale` 模式、tile 数量、候选掩码数量、每个掩码面积和输出路径。由于这类文件体积较大，仓库默认不提交完整 summary。

常见输出文件：

```text
*_sam_auto_summary.json   # 分割统计、每个掩码面积、路径和 tile 信息，默认不提交
*_sam_auto_union.png      # 合并后的二值掩码
*_sam_auto_overlay.png    # 原图上的蓝色叠加可视化
*_sam_auto_masks/         # 回贴到原图尺度的单个掩码，默认不提交
*_sam_auto_upscaled_masks/# SAM 输入尺度的单个掩码，默认不提交
```

## 2. 按面积和形状筛选掩码

入口脚本：

```bash
python tools/filter_sam_masks_by_area.py --help
```

基于参考图像面积分布筛选目标图像：

```bash
python tools/filter_sam_masks_by_area.py \
  --reference-summary demo/858f3853776825b04e3608e2442d904a_sam_auto_split36_upscale/858f3853776825b04e3608e2442d904a_sam_auto_summary.json \
  --target-summary demo/a86387bac9e4ab20fe9391a400b9e219_sam_auto_split36_upscale/a86387bac9e4ab20fe9391a400b9e219_sam_auto_summary.json \
  --output-dir demo/a86387bac9e4ab20fe9391a400b9e219_area_filtered_from_858_split36 \
  --reference-min-area 50 \
  --reference-max-quantile 0.9 \
  --bin-width 100 \
  --mode-window-ratio 0.75 \
  --trim-std 2.5 \
  --trim-iters 5 \
  --num-std 2.0
```

Windows PowerShell 示例：

```powershell
python tools/filter_sam_masks_by_area.py `
  --reference-summary demo\858f3853776825b04e3608e2442d904a_sam_auto_split36_upscale\858f3853776825b04e3608e2442d904a_sam_auto_summary.json `
  --target-summary demo\a86387bac9e4ab20fe9391a400b9e219_sam_auto_split36_upscale\a86387bac9e4ab20fe9391a400b9e219_sam_auto_summary.json `
  --output-dir demo\a86387bac9e4ab20fe9391a400b9e219_area_filtered_from_858_split36 `
  --reference-min-area 50 `
  --reference-max-quantile 0.9 `
  --bin-width 100 `
  --mode-window-ratio 0.75 `
  --trim-std 2.5 `
  --trim-iters 5 `
  --num-std 2.0
```

执行后的效果：

- 从参考 summary 中收集候选掩码面积。
- 可先按最小面积、最大分位数、主峰面积窗口和迭代高斯裁剪拟合主目标面积分布。
- 根据 `mu ± num_std * sigma` 得到目标面积保留区间。
- 对目标 summary 中的候选掩码逐个判断是否保留。
- 输出新的 union 掩码、overlay 图和筛选 summary。
- 执行完成后会生成 `*_area_filtered_overlay.png`、`*_area_filtered_union.png` 和 `*_area_filtered_summary.json`。仓库保留 overlay/union 作为代表性效果图，完整 summary 默认不提交。

## 可选形状筛选

`filter_sam_masks_by_area.py` 还支持形状约束：

```bash
python tools/filter_sam_masks_by_area.py \
  --target-summary demo/a86387bac9e4ab20fe9391a400b9e219_sam_auto_split36_upscale/a86387bac9e4ab20fe9391a400b9e219_sam_auto_summary.json \
  --output-dir demo/a86387_shape_filtered \
  --combine-mode shape \
  --shape-filter ball \
  --min-circularity 0.55 \
  --max-aspect-ratio 1.6 \
  --min-extent 0.45 \
  --max-extent 0.95 \
  --min-solidity 0.8 \
  --min-circle-iou 0.5
```

常用参数含义：

- `--combine-mode area|shape|and|or`：决定面积过滤和形状过滤如何组合。
- `--shape-filter off|ball|ball_or_cluster`：关闭形状筛选、筛选单个球状目标，或允许球状簇。
- `--shape-mask-source auto|mask_file|upscaled_mask_file`：选择用于计算形状特征的掩码来源。
- `--reject-border-touching`：剔除接触图像边界的掩码。

## 本次整理与 GitHub 同步命令

本地仓库初始化与提交：

```bash
git init
git add README.md .gitignore tools \
  demo/858f3853776825b04e3608e2442d904a.bmp \
  demo/a86387bac9e4ab20fe9391a400b9e219.bmp \
  demo/858f3853776825b04e3608e2442d904a_sam_auto_split36_upscale/*_overlay.png \
  demo/858f3853776825b04e3608e2442d904a_sam_auto_split36_upscale/*_union.png \
  demo/a86387bac9e4ab20fe9391a400b9e219_sam_auto_split36_upscale/*_overlay.png \
  demo/a86387bac9e4ab20fe9391a400b9e219_sam_auto_split36_upscale/*_union.png \
  demo/a86387bac9e4ab20fe9391a400b9e219_area_filtered_from_858_split36/*_overlay.png \
  demo/a86387bac9e4ab20fe9391a400b9e219_area_filtered_from_858_split36/*_union.png
git commit -m "Initial ball segmentation project"
```

创建 GitHub 远程仓库并推送到账号 `ly-rrrrr` 下的同名仓库：

```bash
gh repo create ly-rrrrr/ball_segmentation --private --source . --remote origin --push
```

如果希望仓库公开，将 `--private` 改为 `--public`：

```bash
gh repo create ly-rrrrr/ball_segmentation --public --source . --remote origin --push
```

执行成功后的效果：

- GitHub 上会出现新仓库 `ly-rrrrr/ball_segmentation`。
- 本地 `origin` 远程地址会指向该仓库。
- 当前目录中已提交的代码、原始示例图、代表性分割结果图和说明文档会同步到远程默认分支。
- 后续修改可以使用 `git add`、`git commit`、`git push` 继续更新远程仓库。

## 注意事项

- `demo/` 本地可能包含大量 mask PNG 和较大的 JSON 统计文件；这些文件已在 `.gitignore` 中忽略，仓库只保留少量代表性结果图。
- 单个文件目前未超过 GitHub 普通仓库 100 MB 的硬限制；如后续加入模型权重或更大的原始数据，建议使用 Git LFS 或不要直接提交权重文件。
- `--model-path` 应指向本地 SAM 模型目录，模型权重不包含在当前项目中。
