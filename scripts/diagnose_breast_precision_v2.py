from pathlib import Path
import pandas as pd
import torch

import train_breast_resnet50_ivaf as training
from compare_breast_checkpoints import ValidationDataset

checkpoint_path = Path(
    "saved_models/breast_resnet50_ivaf_imagenet_best.pth"
)
saved_path = Path(
    "reports/breast_resnet50_ivaf_imagenet_predictions.csv"
)

device = training.DEVICE
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=True
)

model = training.ResNet50IVAF(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model = model.to(device).eval()

rows = training.prepare_rows(training.VAL_CSV, "validation")
pairs, _ = training.build_pairs(rows, "validation")
dataset = ValidationDataset(pairs, use_imagenet_normalization=True)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

outputs = {}

for mode in ["fp32", "autocast"]:
    records = []

    with torch.inference_mode():
        for cc, mlo, labels, patient_ids, cc_paths, mlo_paths in loader:
            cc, mlo = cc.to(device), mlo.to(device)

            with torch.autocast(
                device_type="cuda",
                enabled=(mode == "autocast" and device.type == "cuda"),
            ):
                logits = model(cc, mlo)

            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()

            for i, probability in enumerate(probs):
                records.append({
                    "patient_id": patient_ids[i],
                    "cc_path": cc_paths[i],
                    "mlo_path": mlo_paths[i],
                    "y_true": int(labels[i]),
                    f"prob_{mode}": probability,
                })

    outputs[mode] = pd.DataFrame(records)

keys = ["cc_path", "mlo_path", "y_true"]

merged = outputs["fp32"].merge(
    outputs["autocast"],
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

print("\nFP32 vs autocast")
print("Matched pairs:", int((merged["_merge"] == "both").sum()))
print("Unmatched rows:", int((merged["_merge"] != "both").sum()))

matched = merged[merged["_merge"] == "both"].copy()
matched["abs_diff"] = (
    matched["prob_fp32"] - matched["prob_autocast"]
).abs()
matched["fp32_pred"] = (matched["prob_fp32"] >= 0.5).astype(int)
matched["autocast_pred"] = (matched["prob_autocast"] >= 0.5).astype(int)

print("Mean absolute probability difference:", matched["abs_diff"].mean())
print("Maximum absolute probability difference:", matched["abs_diff"].max())
print("Changed predictions:", int(
    (matched["fp32_pred"] != matched["autocast_pred"]).sum()
))

if saved_path.exists():
    saved = pd.read_csv(saved_path)
    saved = saved[keys + ["malignant_probability"]].rename(
        columns={"malignant_probability": "prob_saved"}
    )

    compared = matched.merge(
        saved,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="saved_match",
    )

    print("\nSaved predictions vs fresh FP32")
    print("Matched saved rows:", int((compared["saved_match"] == "both").sum()))

    valid = compared[compared["saved_match"] == "both"]
    if len(valid):
        print("Mean difference:", (
            valid["prob_saved"] - valid["prob_fp32"]
        ).abs().mean())
        print("Maximum difference:", (
            valid["prob_saved"] - valid["prob_fp32"]
        ).abs().max())

        print("\nSaved predictions vs fresh autocast")
        print("Mean difference:", (
            valid["prob_saved"] - valid["prob_autocast"]
        ).abs().mean())
        print("Maximum difference:", (
            valid["prob_saved"] - valid["prob_autocast"]
        ).abs().max())

print("\nDiagnostic complete. No model or dataset files modified.")
