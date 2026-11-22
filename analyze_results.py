
import csv

FILES = {
    "x2": "results_x2.csv",
    "x4": "results_x4.csv"
}

def load_results(filename):
    with open(filename, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    return {row["model"]: row for row in rows}


def analyze_scale(scale, filename):
    results = load_results(filename)

    fs = results[f"fsrcnn_{scale}"]
    es = results[f"efficient_c48b6_{scale}"]

    def number(row, key):
        return float(row[key].replace(",", ""))

    # Quality metrics
    fs_psnr = number(fs, "avg PSNR")
    es_psnr = number(es, "avg PSNR")

    fs_ssim = number(fs, "avg SSIM")
    es_ssim = number(es, "avg SSIM")

    # Computational metrics
    fs_params = number(fs, "params")
    es_params = number(es, "params")

    fs_gmacs = number(fs, "GMACs@720p (learned convs)")
    es_gmacs = number(es, "GMACs@720p (learned convs)")

    cpu_column = "CPU ms@720p (end-to-end, 4T, median)"
    fs_cpu = number(fs, cpu_column)
    es_cpu = number(es, cpu_column)

    print("\n" + "=" * 55)
    print(f"ANALYSIS: {scale.upper()} SUPER-RESOLUTION")
    print("=" * 55)

    print("\n1. IMAGE QUALITY")
    print(f"PSNR improvement: {es_psnr - fs_psnr:.2f} dB")
    print(f"PSNR relative improvement: {(es_psnr/fs_psnr - 1)*100:.2f}%")
    print(f"SSIM improvement: {es_ssim - fs_ssim:.4f}")
    print(f"SSIM relative improvement: {(es_ssim/fs_ssim - 1)*100:.2f}%")

    print("\n2. MODEL COMPLEXITY")
    print(f"Additional parameters: {es_params-fs_params:,.0f}")
    print(f"Parameter increase: {(es_params/fs_params - 1)*100:.2f}%")
    print(f"Additional GMACs: {es_gmacs-fs_gmacs:.3f}")
    print(f"GMAC increase: {(es_gmacs/fs_gmacs - 1)*100:.2f}%")

    print("\n3. CPU INFERENCE TIME")
    print(f"FSRCNN: {fs_cpu:.0f} ms")
    print(f"EfficientSR: {es_cpu:.0f} ms")
    print(f"Additional time: {es_cpu-fs_cpu:.0f} ms")
    print(f"Time increase: {(es_cpu/fs_cpu - 1)*100:.2f}%")
    print(f"EfficientSR is {es_cpu/fs_cpu:.2f}x slower")

    print("\n4. QUALITY VS COMPUTATIONAL COST")
    print(f"PSNR gain per additional GMAC: "
          f"{(es_psnr-fs_psnr)/(es_gmacs-fs_gmacs):.3f} dB/GMAC")
    print(f"PSNR gain per additional 100 ms: "
          f"{(es_psnr-fs_psnr)/((es_cpu-fs_cpu)/100):.3f} dB/100ms")


for scale, filename in FILES.items():
    analyze_scale(scale, filename)

print("\n" + "=" * 55)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 55)
