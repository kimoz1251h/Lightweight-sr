
from PIL import Image, ImageDraw

# Load ×4 images
image_files = {
    "Ground Truth": "sample_original_hr.png",
    "Bicubic": "sample_bicubic.png",
    "FSRCNN x4": "sample_fsrcnn_x4.png",
    "EfficientSR x4": "sample_efficient_c48b6_x4.png"
}

images = {}

for name, filename in image_files.items():
    img = Image.open(filename).convert("RGB")
    print(f"{name}: {img.size}")

    if name == "Ground Truth":
        img = img.resize((320, 480), Image.Resampling.LANCZOS)

    images[name] = img

# Same crop coordinates
crop_box = (130, 25, 225, 270)

zoom_factor = 3
label_height = 45
gap = 15

cropped_images = []

for name, img in images.items():
    crop = img.crop(crop_box)

    zoomed = crop.resize(
        (crop.width * zoom_factor, crop.height * zoom_factor),
        Image.Resampling.NEAREST
    )

    cropped_images.append((name, zoomed))

# Create comparison canvas
crop_width, crop_height = cropped_images[0][1].size

canvas_width = crop_width * 4 + gap * 5
canvas_height = crop_height + label_height + 20

comparison = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(comparison)

for i, (name, img) in enumerate(cropped_images):
    x = gap + i * (crop_width + gap)
    y = label_height

    comparison.paste(img, (x, y))
    draw.text((x + 5, 12), name, fill="black")

output_file = "final_comparison_x4.png"
comparison.save(output_file)

print("\nComparison saved:", output_file)
print("Crop size:", crop_width // zoom_factor, "x", crop_height // zoom_factor)
print("Display zoom:", zoom_factor, "x")
