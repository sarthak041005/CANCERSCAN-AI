
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn import CancerCNN


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
MODEL_PATH = Path("saved_models/lung/lung_resnet18_clean_best.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Match the validation preprocessing used during training.
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------
print(f"Using device: {DEVICE}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint.get("classes", ["Cancer", "NORMAL"])

model = CancerCNN(
    num_classes=len(classes),
    pretrained=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Trained ResNet18 checkpoint loaded.")
print("Class mapping:", {i: name for i, name in enumerate(classes)})


# --------------------------------------------------
# 3. CAPTURE REAL INTERMEDIATE ACTIVATIONS
# --------------------------------------------------
activations = {}

layers = {
    "Conv1": model.backbone.conv1,
    "Layer 1": model.backbone.layer1,
    "Layer 2": model.backbone.layer2,
    "Layer 3": model.backbone.layer3,
    "Layer 4": model.backbone.layer4,
}

hooks = []

def save_activation(name):
    def hook(module, inputs, output):
        activations[name] = output.detach().cpu()
    return hook

for name, layer in layers.items():
    hooks.append(
        layer.register_forward_hook(save_activation(name))
    )


# --------------------------------------------------
# 4. LOAD THE CHEST X-RAY
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Visualize intermediate activations of the trained lung ResNet18."
)
parser.add_argument(
    "image",
    help="Path to a chest X-ray image"
)
args = parser.parse_args()

image_path = Path(args.image)

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

original_image = Image.open(image_path).convert("L")
input_tensor = preprocess(original_image).unsqueeze(0).to(DEVICE)


# --------------------------------------------------
# 5. RUN INFERENCE
# --------------------------------------------------
with torch.inference_mode():
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1)[0]

predicted_index = int(probabilities.argmax().item())
predicted_class = classes[predicted_index]
confidence = float(probabilities[predicted_index].item())

print("\n----- MODEL PREDICTION -----")
print(f"Predicted class: {predicted_class}")
print(f"Model score for predicted class: {confidence:.2%}")

for index, name in enumerate(classes):
    print(f"{name}: {probabilities[index].item():.2%}")


# --------------------------------------------------
# 6. PLOT THE ORIGINAL IMAGE AND FEATURE MAPS
# --------------------------------------------------
columns = 5
feature_maps_per_layer = 5
rows = 1 + len(activations)

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(15, 3.2 * rows)
)

fig.suptitle(
    "CancerScan AI — ResNet18 Intermediate Activations\n"
    f"Prediction: {predicted_class} | Score: {confidence:.1%}",
    fontsize=16,
    fontweight="bold"
)

# Original input
axes[0, 0].imshow(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)),
                  cmap="gray")
axes[0, 0].set_title("Original X-ray")
axes[0, 0].axis("off")

for col in range(1, columns):
    axes[0, col].axis("off")

# Show selected channels from each actual layer output
for row, (layer_name, activation) in enumerate(
    activations.items(), start=1
):
    # activation shape: [batch, channels, height, width]
    feature_tensor = activation[0]
    channel_count = min(feature_maps_per_layer, feature_tensor.shape[0])

    for col in range(columns):
        ax = axes[row, col]

        if col < channel_count:
            feature_map = feature_tensor[col].numpy()

            # Scale each map for display only; model values are unchanged.
            low = feature_map.min()
            high = feature_map.max()

            if high > low:
                feature_map = (feature_map - low) / (high - low)

            ax.imshow(feature_map, cmap="viridis")
            ax.set_title(f"{layer_name}\nChannel {col + 1}")
        else:
            ax.axis("off")

        ax.axis("off")

    print(
        f"{layer_name} activation shape: "
        f"{tuple(activation.shape)}"
    )

plt.tight_layout(rect=[0, 0, 1, 0.95])

output_path = OUTPUT_DIR / "lung_intermediate_activations.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"\nVisualization saved to: {output_path.resolve()}")

plt.show()

for hook in hooks:
    hook.remove()

