import pandas as pd
from pathlib import Path

KEYS = [
    "patient_id", "breast", "abnormality_type",
    "pathology", "assessment", "subtlety", "label"
]

for filename in ["datasets/train.csv", "datasets/val.csv"]:
    df = pd.read_csv(filename)
    df["view"] = df["view"].astype(str).str.upper()
    df["pathology"] = df["pathology"].astype(str).str.upper()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    print("\n" + "=" * 60)
    print(filename)

    # Verify pathology-to-label mapping.
    print("\nPathology vs label:")
    print(pd.crosstab(df["pathology"], df["label"], dropna=False))

    expected = df["pathology"].map({
        "MALIGNANT": 1,
        "BENIGN": 0,
        "BENIGN_WITHOUT_CALLBACK": 0
    })
    conflicts = df[expected.notna() & df["label"].ne(expected)]
    print("Pathology-label conflicts:", len(conflicts))

    # Count each CC/MLO view for every clinical key.
    counts = df.pivot_table(
        index=KEYS,
        columns="view",
        values="image_path",
        aggfunc="count",
        fill_value=0
    )

    for view in ["CC", "MLO"]:
        if view not in counts.columns:
            counts[view] = 0

    paired = counts[(counts["CC"] == 1) & (counts["MLO"] == 1)]
    duplicate_view_keys = counts[
        (counts["CC"] > 1) | (counts["MLO"] > 1)
    ]
    unmatched = counts[
        (counts["CC"] == 0) | (counts["MLO"] == 0)
    ]

    print("\nExact one-CC/one-MLO pairs:", len(paired))
    print("Keys with repeated views:", len(duplicate_view_keys))
    print("Keys missing a view:", len(unmatched))

    if len(duplicate_view_keys):
        print("\nRepeated-view examples:")
        print(duplicate_view_keys.head(10).to_string())

print("\nAudit complete. No files were modified.")
