from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / 'datasets'
MODEL_DIR = BASE_DIR / 'saved_models'
REPORT_DIR = BASE_DIR / 'reports'

IMAGE_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_WORKERS = 0
SEED = 42

BREAST_CLASSES = ['BENIGN', 'MALIGNANT']
LUNG_CLASSES = ['NORMAL', 'Cancer']

BREAST_MODELS = {
    'resnet50_ivaf': MODEL_DIR / 'breast_resnet50_ivaf_stage3_improved_best.pth',
    'densenet121_multiview': MODEL_DIR / 'breast_densenet121_multiview_512_best.pth',
    'focal_lesion': MODEL_DIR / 'breast_roi_lesion_classifier_best.pth',
}

LUNG_MODELS = {
    'densenet121': MODEL_DIR / 'lung' / 'densenet121_512_best.pth',
    'resnet18': MODEL_DIR / 'lung' / 'lung_resnet18_clean_best.pth',
}

CONFIGS = {
    'breast': {
        'dataset_dir': DATASET_DIR / 'breast_expanded',
        'classes': BREAST_CLASSES,
        'model_path': MODEL_DIR / 'breast_resnet50_ivaf_stage3_improved_best.pth',
    },
    'lung': {
        'dataset_dir': DATASET_DIR / 'lung',
        'classes': LUNG_CLASSES,
        'model_path': MODEL_DIR / 'lung' / 'lung_resnet18_clean_best.pth',
    }
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
