import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 15

TRAIN_CSV = "mnt/data/CancerScan_AI/datasets/roi_train.csv"
VAL_CSV = "mnt/data/CancerScan_AI/datasets/roi_val.csv"

MODEL_DIR = "mnt/data/CancerScan_AI/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Train ROI images:", len(train_df))
print("Validation ROI images:", len(val_df))

print("\nTrain labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(val_df["label"].value_counts())

# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"]
)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("\nClass weights:")
print(class_weights)

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Grayscale -> RGB
    image = tf.image.grayscale_to_rgb(image)

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, tf.cast(label, tf.float32)


# ============================================================
# DATASET
# ============================================================

def create_dataset(df, shuffle=False):

    paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_df,
    shuffle=True
)

val_ds = create_dataset(
    val_df
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        mode="horizontal"
    ),

    layers.RandomRotation(
        0.03
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )
])

# ============================================================
# EFFICIENTNETB0
# ============================================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ============================================================
# CLASSIFICATION MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.4
)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.3
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.AUC(
            name="auc"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=os.path.join(
            MODEL_DIR,
            "roi_best_model.keras"
        ),

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=4,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING ROI MODEL TRAINING")
print("========================================\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_path = os.path.join(
    MODEL_DIR,
    "roi_final_model.keras"
)

model.save(final_path)

print("\n========================================")
print("ROI TRAINING COMPLETED")
print("========================================")

print("\nBest model:")
print(
    os.path.join(
        MODEL_DIR,
        "roi_best_model.keras"
    )
)

print("\nFinal model:")
print(final_path)