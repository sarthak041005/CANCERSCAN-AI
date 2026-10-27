from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / 'datasets'
MODEL_DIR = BASE_DIR / 'saved_models'
REPORT_DIR = BASE_DIR / 'reports'

IMAGE_SIZE = 512

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

NUM_WORKERS = 0

SEED = 42

BREAST_CLASSES = ['BENIGN', 'MALIGNANT']

LUNG_CLASSES = ['NORMAL', 'Cancer']

CONFIGS = {
    'breast': {
        'dataset_dir': DATASET_DIR / 'breast_expanded',
        'classes': BREAST_CLASSES,
        'model_path': MODEL_DIR / 'breast_cnn_expanded.pth'
    },

    'lung': {
        'dataset_dir': DATASET_DIR / 'lung',
        'classes': LUNG_CLASSES,
        'model_path': MODEL_DIR / 'lung_cnn.pth'
    }
}

MODEL_DIR.mkdir(
    exist_ok=True
)

REPORT_DIR.mkdir(
    exist_ok=True
)   