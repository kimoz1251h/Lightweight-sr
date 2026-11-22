
import argparse
import copy
import csv
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
HERE = Path(__file__).resolve().parent     # every default path is relative to this file,
                                           # so F5 / terminal / any working folder behave the same
BICUBIC = Image.Resampling.BICUBIC


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(folder):
    folder = Path(folder)
    assert folder.is_dir(), f"Folder not found: {folder}"
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
 
    hr = [p for p in files if p.stem.lower().endswith("_hr")]
    files = hr if hr else [p for p in files if not p.stem.lower().endswith(("_lr", "_bicubic"))]
    assert files, f"No images found in {folder}"
    return files


def crop_to_multiple(img, scale):
    """Trim width/height so they divide by `scale` (LR x scale == HR exactly)."""
    w, h = img.size
    return img.crop((0, 0, w - w % scale, h - h % scale))


def make_lr(hr_img, scale):
    """Shrink an HR PIL image by `scale` (bicubic) -> LR image."""
    w, h = hr_img.size
    return hr_img.resize((w // scale, h // scale), BICUBIC)


class TrainPatches(Dataset):
    def __init__(self, root, scale, patch=96, per_image=20):
        self.files = list_images(root)
        self.scale = scale
        self.patch = patch
        self.per_image = per_image

    def __len__(self):
        return len(self.files) * self.per_image

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx % len(self.files)]).convert("RGB")
        p, s = self.patch, self.scale

        # Crop the HR patch FIRST (aligned to the scale grid), then shrink only that patch.
        # Much cheaper than resizing the whole ~2K image for every sample.
        x = random.randint(0, max(0, (hr.width - p) // s)) * s
        y = random.randint(0, max(0, (hr.height - p) // s)) * s
        hr = hr.crop((x, y, x + p, y + p))
        lr = hr.resize((p // s, p // s), BICUBIC)

        # Same augmentation for both (patches are square, so rot90 is safe)
        if random.random() < 0.5:
            lr = lr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lr = lr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            lr = lr.transpose(Image.Transpose.ROTATE_90)
            hr = hr.transpose(Image.Transpose.ROTATE_90)

        return to_tensor(lr), to_tensor(hr)


class FullImages(Dataset):
    """Whole HR images for validation / test (batch size 1: sizes differ)."""

    def __init__(self, folder, scale, limit=None):
        files = list_images(folder)
        tagged = [f for f in files if f"srf_{scale}_" in f.stem.lower()]   # e.g. img_001_SRF_2_HR
        self.files = (tagged or files)[:limit or None]                     # limit 0 = all
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        hr = crop_to_multiple(Image.open(self.files[i]).convert("RGB"), self.scale)
        return to_tensor(make_lr(hr, self.scale)), to_tensor(hr), self.files[i].name


# ==========================================================================
# 2) MODELS
# NOTE: outputs are clamped to [0, 1] only in eval mode. Clamping while training
# kills the gradient for out-of-range pixels (they can get stuck).
# ==========================================================================
def finish(x, training):
    return x if training else x.clamp(0, 1)


class ESPCN(nn.Module):
    """Classic: 3 convs on the LR image, then PixelShuffle."""

    def __init__(self, scale):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3 * scale * scale, 3, padding=1),
        )
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return finish(self.shuffle(self.body(x)), self.training)


class FSRCNN(nn.Module):
    """Classic: feature extraction, shrink, mapping, expand, transposed-conv upsampling."""

    def __init__(self, scale, d=56, s=12, m=4):
        super().__init__()
        layers = [nn.Conv2d(3, d, 5, padding=2), nn.PReLU(d),
                  nn.Conv2d(d, s, 1), nn.PReLU(s)]
        for _ in range(m):
            layers += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
        layers += [nn.Conv2d(s, d, 1), nn.PReLU(d),
                   nn.ConvTranspose2d(d, 3, 9, stride=scale, padding=4,
                                      output_padding=scale - 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return finish(self.net(x), self.training)


class Block(nn.Module):
    

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.pw1 = nn.Conv2d(c, c, 1)
        self.act = nn.PReLU(c)
        self.pw2 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.dw(x))))


class EfficientSR(nn.Module):
    

    def __init__(self, scale, channels=48, blocks=6, skip=True):
        super().__init__()
        self.scale = scale
        self.skip = skip
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, 3 * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        h = self.head(x)
        f = self.blocks(h) + h
        out = self.shuffle(self.tail(f))
        if self.skip:
            out = out + F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                                      align_corners=False)
        return finish(out, self.training)


def build_model(name, scale, channels=48, blocks=6, skip=True):
    if name == "espcn":
        return ESPCN(scale)
    if name == "fsrcnn":
        return FSRCNN(scale)
    if name == "efficient":
        return EfficientSR(scale, channels, blocks, skip)
    raise ValueError(f"unknown model {name}")


def ckpt_path(name, scale, channels, blocks, skip=True):
    tag = f"efficient_c{channels}b{blocks}" + ("" if skip else "_noskip") if name == "efficient" else name
    return str(HERE / f"ckpt_{tag}_x{scale}.pth")


def load_ckpt(path):
    if not Path(path).exists():                       # allow just the file name
        path = HERE / path
    ck = torch.load(path, map_location=DEVICE)
    model = build_model(ck["name"], ck["scale"], ck["channels"], ck["blocks"],
                        ck.get("skip", True)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["scale"], Path(path).stem.replace("ckpt_", "")



def rgb_to_y(img):                       # (1,3,H,W) in [0,1]
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def _gauss(size=11, sigma=1.5):
    c = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a, b):                          # (1,1,H,W)
    w = _gauss().to(a.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    s_aa = F.conv2d(a * a, w) - mu_a ** 2
    s_bb = F.conv2d(b * b, w) - mu_b ** 2
    s_ab = F.conv2d(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean().item()


def score(sr, hr, scale):
    y_sr, y_hr = rgb_to_y(sr), rgb_to_y(hr)
    if min(y_hr.shape[-2:]) > 2 * scale + 11:
        y_sr = y_sr[..., scale:-scale, scale:-scale]
        y_hr = y_hr[..., scale:-scale, scale:-scale]
    return psnr(y_sr, y_hr), ssim(y_sr, y_hr)


def bicubic_up(lr, size):
    return F.interpolate(lr, size=size, mode="bicubic", align_corners=False).clamp(0, 1)


def to_8bit(x):
    """Round to 8-bit levels, like saving the SR image as a PNG (standard before PSNR)."""
    return (x.clamp(0, 1) * 255.0).round() / 255.0


def evaluate(model, loader, scale):
    """Average PSNR / SSIM of `model` (None = bicubic) over a loader."""
    if model is not None:
        model.eval()
    ps, ss = [], []
    with torch.no_grad():
        for lr, hr, _ in loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = bicubic_up(lr, hr.shape[-2:]) if model is None else model(lr)
            assert sr.shape == hr.shape, f"size mismatch {sr.shape} vs {hr.shape}"
            p, s = score(to_8bit(sr), hr, scale)
            ps.append(p)
            ss.append(s)
    return sum(ps) / len(ps), sum(ss) / len(ss)


# ==========================================================================
# 4) COST: parameters, GMACs (FLOPs/2) and CPU runtime
# ==========================================================================
def count_params(model):
    return 0 if model is None else sum(p.numel() for p in model.parameters())


def count_gmacs(model, scale, out_h=720, out_w=1280):
    """Multiply-accumulates to produce a 1280x720 output (FLOPs ~ 2 x MACs).
    Counts conv / transposed-conv layers; the bicubic skip is not counted."""
    if model is None:
        return 0.0
    model = copy.deepcopy(model).cpu().eval()
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def deconv_hook(m, inp, out):
        total[0] += inp[0][0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.ConvTranspose2d):
            hooks.append(mod.register_forward_hook(deconv_hook))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
    with torch.no_grad():
        model(torch.rand(1, 3, out_h // scale, out_w // scale))
    for h in hooks:
        h.remove()
    return total[0] / 1e9


def cpu_ms(model, scale, out_h=720, out_w=1280, runs=20, warmup=2):
    """MEDIAN CPU milliseconds to produce a 1280x720 output (thread count is set by the caller)."""
    model = None if model is None else copy.deepcopy(model).cpu().eval()
    x = torch.rand(1, 3, out_h // scale, out_w // scale)
    run = (lambda: bicubic_up(x, (out_h, out_w))) if model is None else (lambda: model(x))
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            run()
        for _ in range(runs):
            t = time.perf_counter()
            run()
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


# ==========================================================================
# 5) TRAIN (+ validate after every epoch)
# ==========================================================================
def save_atomic(obj, path):
    """Write to a temp file, then rename. If the laptop dies mid-save, the old
    checkpoint is still intact instead of a half-written (corrupt) file."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def train(a):
    set_seed(a.seed)
    scale = a.scale
    skip = not a.no_skip
    train_loader = DataLoader(TrainPatches(Path(a.data) / "train", scale),
                              batch_size=a.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(FullImages(Path(a.data) / "val", scale, a.val_limit), batch_size=1)

    model = build_model(a.model, scale, a.channels, a.blocks, skip).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    step_size = max(1, a.epochs // 3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.5)
    path = ckpt_path(a.model, scale, a.channels, a.blocks, skip)   # best model (for benchmark)
    last_path = path.replace(".pth", "_last.pth")                  # resume point (every epoch)
    print(f"Device {DEVICE} | {a.model} x{scale} | params {count_params(model):,} | "
          f"{count_gmacs(model, scale):.3f} GMACs @720p")

    meta = {"name": a.model, "scale": scale, "channels": a.channels,
            "blocks": a.blocks, "skip": skip}

    # ---- RESUME: continue from the last finished epoch if a resume file exists ----
    start_epoch, best = 1, 0.0
    if Path(last_path).exists() and not a.fresh:
        st = torch.load(last_path, map_location=DEVICE)
        old = (st["name"], st["scale"], st["channels"], st["blocks"], st.get("skip", True))
        if old != (a.model, scale, a.channels, a.blocks, skip):
            raise SystemExit(f"{last_path} was saved with a different model/scale/size. "
                             f"Delete it or run with --fresh to start over.")
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        sched.step_size = step_size            # loaded state keeps the OLD step_size; use the new --epochs
        start_epoch, best = st["epoch"] + 1, st["best"]
        print(f"Resuming from epoch {st['epoch']} (best val PSNR so far {best:.2f} dB)")
        if start_epoch > a.epochs:
            print(f"Already trained for {st['epoch']} epochs (>= --epochs {a.epochs}). "
                  f"Increase --epochs to train longer, or use --fresh to restart.")
            return

    for epoch in range(start_epoch, a.epochs + 1):
        t0 = time.perf_counter()                        # start timer
        model.train()                                   # --- learning ---
        total = 0.0
        for lr, hr in train_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            loss = F.l1_loss(model(lr), hr)             # SR patch vs real HR patch
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        t_train = time.perf_counter() - t0              # training time only

        v_psnr, v_ssim = evaluate(model, val_loader, scale)   # --- score only ---
        dt = time.perf_counter() - t0                   # training + validation
        print(f"Epoch {epoch:02d}/{a.epochs} | loss {total / len(train_loader):.4f} | "
              f"val PSNR {v_psnr:.2f} dB  SSIM {v_ssim:.4f} | "
              f"train {t_train:.0f} s, total {dt:.0f} s")
        if v_psnr > best:
            best = v_psnr
            save_atomic({**meta, "state_dict": model.state_dict()}, path)
            print(f"  -> saved {path}")

        # resume point: everything needed to continue exactly where we stopped
        save_atomic({**meta, "state_dict": model.state_dict(),
                     "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "epoch": epoch, "best": best}, last_path)
    print(f"Done. Best val PSNR {best:.2f} dB")


# ==========================================================================
# 6) BENCHMARK: all models x all test sets -> table + CSV + markdown
# ==========================================================================
def benchmark(a):
    torch.set_num_threads(a.threads)                    # fixed thread count -> reproducible runtime
    root = Path(a.test_root)
    sets = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert sets, f"No test-set folders in {root} (expected Set5/, Set14/, ...)"
    loaders = {s: DataLoader(FullImages(root / s, a.scale), batch_size=1) for s in sets}

    entries = [("bicubic", None)]
    for p in a.ckpts:
        model, ck_scale, label = load_ckpt(p)
        assert ck_scale == a.scale, f"{p} is x{ck_scale}, but --scale is {a.scale}"
        entries.append((label, model))

    header = ["model", "params", "GMACs@720p (learned convs)",
              f"CPU ms@720p (end-to-end, {a.threads}T, median)"] + \
             [f"{s} PSNR/SSIM" for s in sets] + ["avg PSNR", "avg SSIM"]
    rows = []
    for label, model in entries:
        print(f"Evaluating {label} ...")
        res = [evaluate(model, loaders[s], a.scale) for s in sets]
        row = [label, f"{count_params(model):,}", f"{count_gmacs(model, a.scale):.3f}",
               f"{cpu_ms(model, a.scale):.0f}"]
        row += [f"{p:.2f} / {s:.4f}" for p, s in res]
        row += [f"{sum(p for p, _ in res) / len(res):.2f}",
                f"{sum(s for _, s in res) / len(res):.4f}"]
        rows.append(row)

    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    table = [fmt(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [fmt(r) for r in rows]
    print(f"\nResults x{a.scale} (PSNR in dB on Y channel, 8-bit rounded; GMACs ~ FLOPs/2; "
          f"CPU threads = {a.threads})\n")
    print("\n".join(table))

    note = ("Note: GMACs count only the learned conv / transposed-conv layers. The bicubic skip "
            "(and the bicubic baseline's interpolation) is NOT counted in GMACs, but IS included "
            "in the end-to-end CPU runtime, so runtime can differ between models with equal GMACs.")
    print("\n" + note)

    (HERE / f"results_x{a.scale}.md").write_text("\n".join(table) + "\n\n" + note + "\n")
    with open(HERE / f"results_x{a.scale}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nSaved results_x{a.scale}.md and results_x{a.scale}.csv")

    # visual sample: LR / bicubic / each model / HR for the first image of the first set
    from torchvision.utils import save_image
    lr, hr, name = next(iter(loaders[sets[0]]))
    lr_d = lr.to(DEVICE)
    save_image(lr, str(HERE / "sample_lr_input.png"))
    save_image(hr, str(HERE / "sample_original_hr.png"))
    save_image(bicubic_up(lr_d, hr.shape[-2:]).cpu(), str(HERE / "sample_bicubic.png"))
    with torch.no_grad():
        for label, model in entries[1:]:
            save_image(model(lr_d).cpu(), str(HERE / f"sample_{label}.png"))
    print(f"Saved sample images for {name[0]}")


# ==========================================================================
# 7) PREDICT: enlarge any image (no HR needed -> no score)
# ==========================================================================
def predict(a):
    model, scale, label = load_ckpt(a.ckpt)
    lr = to_tensor(Image.open(a.image).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sr = model(lr).squeeze(0).cpu()
    out = f"{Path(a.image).stem}_x{scale}_{label}.png"
    to_pil_image(sr).save(out)
    print(f"{lr.shape[-1]}x{lr.shape[-2]} -> {sr.shape[-1]}x{sr.shape[-2]}  saved {out}")


# ==========================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", choices=["espcn", "fsrcnn", "efficient"], required=True)
    t.add_argument("--scale", type=int, choices=[2, 4], default=4)
    t.add_argument("--data", default=str(Path(__file__).resolve().parent / "sr_data"))
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--channels", type=int, default=48, help="efficient model width")
    t.add_argument("--blocks", type=int, default=6, help="efficient model depth")
    t.add_argument("--no-skip", action="store_true",
                   help="efficient model: remove the bicubic global skip (ablation)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--val-limit", type=int, default=20,
                   help="use only the first N validation images (faster on CPU)")
    t.add_argument("--fresh", action="store_true",
                   help="ignore any saved resume file and start from epoch 1")

    b = sub.add_parser("benchmark")
    b.add_argument("--scale", type=int, choices=[2, 4], default=4)
    b.add_argument("--test-root", default=str(HERE / "sr_data" / "test_sets"))
    b.add_argument("--threads", type=int, default=4, help="CPU threads for the runtime measurement")
    b.add_argument("--ckpts", nargs="+", required=True)

    r = sub.add_parser("predict")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--image", required=True)

    if len(sys.argv) > 1:                       # terminal: python sr_benchmark.py <command> ...
        args = p.parse_args()
    else:                                       # F5 in VS Code: this default job (edit it freely)
        args = p.parse_args(["train", "--model", "fsrcnn", "--scale", "2",
                             "--epochs", "65", "--batch-size", "32",
                             "--val-limit", "50"])
    {"train": train, "benchmark": benchmark, "predict": predict}[args.cmd](args)