# Engineering-Focused README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the repository root README for engineering users while preserving existing technical content and presenting methods as independent or composable options without sequence numbering.

**Architecture:** Replace only `README.md`. Keep detailed training, history, annotation, and resource material in existing linked documents; make the root page a concise operational entry point with verified commands and direct download links.

**Tech Stack:** Markdown, PowerShell command examples, Git.

---

### Task 1: Rewrite the GitHub Entry Page

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the README information order**

Write sections in this order: project overview, representative results, installation and weights, shortest SAM-to-CNN-v4.1 workflow, independent method options, annotation/resources, repository layout, detailed documentation, and license.

- [ ] **Step 2: Add the shortest complete workflow**

Include exact commands using:

```text
methods/01_sam_generation/run_sam_automatic_mask.py
methods/06_cnn_v4_v41/export_split_upscale_cnn_inference_dataset.py
methods/06_cnn_v4_v41/apply_cnn_scale_aware_classifier_v41.py
```

Use `weights/sam/`, `checkpoints/cnn_v41/cnn_scale_aware_model.pt`, `data/example.bmp`, `work/example_cnn_v41/`, and `outputs/example_cnn_v41/` consistently.

- [ ] **Step 3: Present methods without sequence numbering**

Use display names `SAM candidate generation`, `Area and shape filtering`, `Seed heuristic filtering`, `Learned-seed filtering`, `CNN v3`, and `CNN v4/v4.1`. Add a sentence stating that these are alternative or composable approaches, not mandatory sequential stages. Preserve existing numbered directory links.

- [ ] **Step 4: Add direct engineering links**

Link the official SAM model, shared CNN weights, representative examples, every method README, `docs/reproduction.md`, `docs/resources.md`, `docs/method_history.md`, and `annotation/README.md`.

### Task 2: Verify and Publish

**Files:**
- Verify: `README.md`

- [ ] **Step 1: Check Markdown and references**

Run:

```powershell
git diff --check
rg -n "huggingface.co/facebook/sam-vit-huge|1YxNJngJzs4wbboNhpb_cbjycBOqLfMAu|methods/01_sam_generation|methods/06_cnn_v4_v41|docs/reproduction.md|docs/resources.md|annotation/README.md" README.md
```

Expected: no whitespace errors and all required links/paths are present.

- [ ] **Step 2: Confirm method headings are not numbered**

Run:

```powershell
rg -n "^#{2,3} [0-9]+[.)]" README.md
```

Expected: no matches.

- [ ] **Step 3: Commit only the README**

```powershell
git add README.md
git commit -m "Reorganize README for engineering users"
```

- [ ] **Step 4: Synchronize GitHub**

```powershell
git fetch origin
git push origin main
git rev-parse main
git rev-parse origin/main
```

Expected: local and remote SHAs match.
