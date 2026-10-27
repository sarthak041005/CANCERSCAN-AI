from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

import train_breast_resnet50_ivaf as training
from compare_breast_checkpoints import ValidationDataset

checkpoint_path = Path(
    "saved_models/breast_resnet50_ivaf_imagenet_best.pth"
)
saved_path = Path(
    "reports/breast_resnet50_ivaf_imagenet_predictions.csv"
)

checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=True
)

device = training.DEVICE
model = training.ResNet50IVAF(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model = model.to(device).eval()

rows = training.prepare_rows(training.VAL_CSV, "validation")
pairs, _ = training.build_pairs(rows, "validation")
dataset = ValidationDataset(pairs, use_imagenet_normalization=True)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

outputs = {}

for mode in ["fp32", "autocast"]:
    probabilities = []
    labels_all = []
    ids_all = []

    with torch.inference_mode():
        for cc, mlo, labels, patient_ids, cc_paths, mlo_paths in loader:
            cc = cc.to(device)
            mlo = mlo.to(device)

            with torch.autocast(
                device_type="cuda",
                enabled=(mode == "autocast" and device.type == "cuda"),
            ):
                logits = model(cc, mlo)

            probabilities.extend(
                torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
            )
            labels_all.extend(labels.tolist())
            ids_all.extend(patient_ids)

    outputs[mode] = pd.DataFrame({
        "patient_id": ids_all,
        "y_true": labels_all,
        f"prob_{mode}": probabilities,
    })

result = outputs["fp32"].merge(
    outputs["autocast"], on=["patient_id", "y_true"]
)

result["class_fp32"] = (result["prob_fp32"] >= 0.5).astype(int)
result["class_autocast"] = (result["prob_autocast"] >= 0.5).astype(int)

print("FP32 vs autocast")
print("Mean probability difference:", (
    result["prob_fp32"] - result["prob_autocast"]
).abs().mean())
print("Max probability difference:", (
    result["prob_fp32"] - result["prob_autocast"]
).abs().max())
print("Changed predictions:", (
    result["class_fp32"] != result["class_autocast"]
).sum())

if saved_path.exists():
    saved = pd.read_csv(saved_path)
    result = result.merge(
        saved[["patient_id", "y_true", "malignant_probability"]].rename(
            columns={"malignant_probability": "prob_saved"}
        ),
        on=["patient_id", "y_true"],
        how="left",
        validate="one_to_one",
    )
    print("\nSaved predictions vs FP32")
    print("Mean probability difference:", (
        result["prob_saved"] - result["prob_fp32"]
    ).abs().mean())
    print("Max probability difference:", (
        result["prob_saved"] - result["prob_fp32"]
    ).abs().max())

    print("\nSaved predictions vs autocast")
    print("Mean probability difference:", (
        result["prob_saved"] - result["prob_autocast"]
    ).abs().mean())
    print("Max probability difference:", (
        result["prob_saved"] - result["prob_autocast"]
    ).abs().max())

print("\nDiagnostic complete. No checkpoint or dataset modified.")
