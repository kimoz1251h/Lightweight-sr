# Lightweight Real-Time Super-Resolution for Edge/Laptop Deployment

CSE 681 -- Advanced Digital Image and Video Processing (team project, catalog entry C16)

Team: Abdelhakeem Osama Abdelhakeem (2600322), Mohamed Mostafa Mahmoud Mohamed (2400066),
Mohammed Mahmoud Mohammed Saad (2400043)

This repository contains a small PyTorch pipeline that trains and benchmarks lightweight
x2/x4 single-image super-resolution (SR) models and measures PSNR/SSIM against parameters,
GMACs and CPU latency.

Models:

| Name | Description |
|---|---|
| `bicubic` | Non-learning baseline (PyTorch bicubic interpolation) |
| `fsrcnn` | FSRCNN baseline (d=56, s=12, m=4), RGB input |
| `espcn` | ESPCN baseline (implemented, not yet trained) |
| `efficient` | **EfficientSR** (ours): 3x3 head, N residual blocks (depthwise 3x3 + pointwise 1x1 + PReLU + pointwise 1x1), 3x3 tail, PixelShuffle, optional global bicubic skip |

## 1. Setup

Python 3.10+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(On Linux/macOS: `source .venv/bin/activate`.) A CUDA GPU is optional: training uses it
automatically when available; the CPU benchmark always runs on the CPU.

## 2. Data

Datasets are **not** included in this repository (see licenses below). Create this layout
next to the script:

```
sr_data/
  train/            DIV2K training HR images (800 images)
  val/              DIV2K validation HR images (100 images; the first 50 are used by default)
  test_sets/
    Set5/  Set14/  BSD100/  Urban100/  Manga109/     (HR images directly inside each folder)
```

- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/ (academic research use)
- Set5, Set14, BSD100, Urban100, Manga109: obtained via the BasicSR dataset guide,
  https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md
- Manga109 is for academic use only and must not be redistributed; do not commit it.
- If a download keeps HR images in an `original/` subfolder, copy them up one level.
  Files ending in `_LR` or `_bicubic` are ignored automatically.

LR images are generated on the fly by PIL bicubic downscaling of the HR images.

## 3. Usage

All commands are run from the repository folder. Replace `initial.py` with the script name if it differs.

Train (checkpoints are saved next to the script; training resumes automatically):

```powershell
# FSRCNN baseline
python initial.py train --model fsrcnn --scale 2 --epochs 65 --batch-size 32 --val-limit 50
python initial.py train --model fsrcnn --scale 4 --epochs 70 --batch-size 32 --val-limit 50

# EfficientSR (C=48 channels, N=6 blocks)
python initial.py train --model efficient --scale 2 --channels 48 --blocks 6 --epochs 65 --batch-size 32 --val-limit 50
python initial.py train --model efficient --scale 4 --channels 48 --blocks 6 --epochs 70 --batch-size 32 --val-limit 50

# ablation: remove the global bicubic skip
python initial.py train --model efficient --scale 2 --no-skip --epochs 30 --batch-size 32 --val-limit 50
```

Useful flags: `--fresh` (ignore the saved resume file), `--seed` (default 0), `--lr` (default 1e-3).

Benchmark (bicubic is always included; writes `results_x{scale}.md`, `results_x{scale}.csv`
and `sample_*.png`):

```powershell
python initial.py benchmark --scale 2 --ckpts ckpt_fsrcnn_x2.pth ckpt_efficient_c48b6_x2.pth
python initial.py benchmark --scale 4 --ckpts ckpt_fsrcnn_x4.pth ckpt_efficient_c48b6_x4.pth
```

For clean CPU timings, close other applications and make sure no training is running.

Enlarge any image (no ground truth needed):

```powershell
python initial.py predict --ckpt ckpt_efficient_c48b6_x2.pth --image my_lr_image.png
```

## 4. Evaluation protocol

- PSNR/SSIM on the Y channel (ITU-R BT.601), outputs rounded to 8 bit, border of `scale` pixels shaved.
  "Avg." is the unweighted mean over the five test sets.
- Parameters: number of trainable parameters.
- GMACs: multiply-accumulates of all conv / transposed-conv layers for a 1280x720 output
  (FLOPs ~ 2 x MACs; the bicubic skip is not counted).
- CPU latency: median of 20 runs after 2 warm-up runs, 4 threads (`--threads`), end to end for a
  1280x720 output.
- Training: L1 loss, Adam (lr 1e-3, halved every floor(E/3) epochs), 96x96 HR patches,
  20 patches per image per epoch, flips and 90-degree rotations, seed 0, batch size 32,
  validation on the first 50 DIV2K validation images.

## 5. Results (Milestone 2)

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU (training); Intel Core i5-11400H, 4 threads (CPU timing).

x2:

| Method | Avg. PSNR | Avg. SSIM | Params | GMACs | CPU (ms) |
|---|---|---|---|---|---|
| Bicubic | 30.52 | 0.8883 | 0 | 0.000 | 15 |
| FSRCNN | 31.62 | 0.9082 | 24,683 | 5.607 | 159 |
| EfficientSR (C48B6) | 33.65 | 0.9273 | 37,932 | 8.460 | 448 |

x4:

| Method | Avg. PSNR | Avg. SSIM | Params | GMACs | CPU (ms) |
|---|---|---|---|---|---|
| Bicubic | 25.84 | 0.7293 | 0 | 0.000 | 15 |
| FSRCNN | 26.65 | 0.7592 | 24,683 | 1.402 | 40 |
| EfficientSR (C48B6) | 27.76 | 0.7927 | 53,520 | 3.011 | 122 |

Notes: EfficientSR is inside the project budget (<= 100,000 parameters, <= 10 GMACs) but is not
budget-matched to FSRCNN; matched-budget variants and ablations are planned for Milestone 3.
Absolute values differ from published tables because LR images are generated with PIL (not MATLAB
imresize), models use RGB input and only DIV2K for training; compare methods only within this protocol.

## 6. Repository contents

- `initial.py` -- data pipeline, models, metrics, `train` / `benchmark` / `predict` commands
- `requirements.txt` -- Python dependencies
- `results_x2.md`, `results_x4.md` (and `.csv`) -- benchmark outputs
- `final_comparison_x2.png`, `final_comparison_x4.png` -- qualitative comparisons used in the report
- `report/` -- LaTeX sources of the milestone reports

## 7. AI-use disclosure

AI assistance (ChatGPT and Claude) was used for LaTeX drafting, clarification of technical concepts,
and code debugging. All technical decisions, experiments, results, and final content were reviewed
and verified by the team.
