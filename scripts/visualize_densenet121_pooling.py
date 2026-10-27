
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\lung_xray_clean_512"
)

CHECKPOINT_PATH = (
    PROJECT_DIR / "saved_models" / "lung" / "densenet121_512_best.pth"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "densenet121_intermediate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a test image by default. Change this to a full image path
# if you want to inspect a particular X-ray.
TEST_DIR = DATA_DIR / "test"


# --------------------------------------------------
# 2. Check the checkpoint and find an X-ray
# --------------------------------------------------
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}\n\n"
        "Train DenseNet121 first, or correct CHECKPOINT_PATH."
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted(
    p for p in TEST_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
)

if not image_paths:
    raise FileNotFoundError(f"No X-ray images found inside {TEST_DIR}")

image_path = image_paths[0]

print("Device:", DEVICE)
print("Input X-ray:", image_path)
print("Checkpoint:", CHECKPOINT_PATH)


# --------------------------------------------------
# 3. Use the same preprocessing as training
# --------------------------------------------------
weights = DenseNet121_Weights.DEFAULT
weight_transform = weights.transforms()

mean = weight_transform.mean
std = weight_transform.std

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

original_image = Image.open(image_path).convert("L")
input_tensor = preprocess(original_image).unsqueeze(0).to(DEVICE)


# --------------------------------------------------
# 4. Rebuild DenseNet121 and load trained weights
# --------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

classes = checkpoint["classes"]

model = models.densenet121(weights=None)
model.classifier = nn.Linear(
    model.classifier.in_features,
    len(classes),
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("Classes:", classes)
print("Checkpoint validation F1:", checkpoint.get("best_val_f1_cancer"))


# --------------------------------------------------
# 5. Capture intermediate outputs using hooks
# --------------------------------------------------
captured = {}

layers_to_capture = {
    "01_initial_max_pool": model.features.pool0,
    "02_transition1_average_pool": model.features.transition1.pool,
    "03_transition2_average_pool": model.features.transition2.pool,
    "04_transition3_average_pool": model.features.transition3.pool,
}

def save_output(name):
    def hook(module, inputs, output):
        captured[name] = output.detach().float().cpu()
    return hook

handles = []

for name, layer in layers_to_capture.items():
    handles.append(layer.register_forward_hook(save_output(name)))


# --------------------------------------------------
# 6. Run one X-ray through the trained network
# --------------------------------------------------
with torch.inference_mode():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

for handle in handles:
    handle.remove()

predicted_index = int(probabilities.argmax())
predicted_class = classes[predicted_index]
confidence = float(probabilities[predicted_index]) * 100

print("\nPrediction:", predicted_class)
print(f"Model confidence: {confidence:.2f}%")

for i, class_name in enumerate(classes):
    print(f"{class_name}: {probabilities[i].item() * 100:.2f}%")


# --------------------------------------------------
# 7. Save the original X-ray
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(original_image, cmap="gray")
ax.set_title("Original chest X-ray")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "00_original_xray.png", dpi=180)
plt.close(fig)


# --------------------------------------------------
# 8. Visualize representative feature maps
# --------------------------------------------------
for layer_name, activation in captured.items():
    # Shape: batch, channels, height, width
    feature_maps = activation[0]
    channel_count = feature_maps.shape[0]

    # Select channels with the highest average absolute activation.
    scores = feature_maps.abs().mean(dim=(1, 2))
    selected_channels = torch.argsort(scores, descending=True)[:8]

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    axes = axes.flatten()

    for ax, channel_idx in zip(axes, selected_channels):
        feature_map = feature_maps[channel_idx].numpy()

        # Normalize each map independently for display.
        low = feature_map.min()
        high = feature_map.max()

        if high > low:
            feature_map = (feature_map - low) / (high - low)
        else:
            feature_map = feature_map * 0

        ax.imshow(feature_map, cmap="viridis")
        ax.set_title(f"Channel {channel_idx.item()}")
        ax.axis("off")

    fig.suptitle(
        f"{layer_name}\n"
        f"Output shape: {tuple(activation.shape)}",
        fontsize=12,
    )
    fig.tight_layout()

    save_path = OUTPUT_DIR / f"{layer_name}.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")
    print("  Tensor shape:", tuple(activation.shape))


print("\nVisualization complete.")
print("Images saved in:", OUTPUT_DIR)
