# CancerScan AI

Breast and lung cancer image classification/detection prototype using PyTorch, transfer learning (ResNet18), OpenCV-compatible image processing, evaluation metrics and Streamlit.

## Dataset folders

Breast (example 3-class dataset): `datasets/breast/train|val|test/{benign,malignant,normal}`

Lung (example binary dataset): `datasets/lung/train|val|test/{normal,cancer}`

**Change the folder names and `config.py` classes if your actual dataset has different labels.** Do not claim cancer staging unless your dataset contains reliable stage labels.

## Setup

`python -m venv venv`

Windows: `venv\\Scripts\\activate`

`pip install -r requirements.txt`

## Train

`python train.py --cancer breast --epochs 20 --batch-size 16`

`python train.py --cancer lung --epochs 20 --batch-size 16`

## Evaluate

`python evaluate.py --cancer breast`

`python evaluate.py --cancer lung`


The system is a research/educational CAD prototype and must not be presented as a clinical diagnostic device.


important commands for running training and testing files

go into project file:
cd "PS C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete"
cd "C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI"

Breast cancer ssl fine tuning: (3000 images)
python ".\train.py" --cancer breast --ssl --epochs 20 --batch-size 16 --lr 0.00005

Breast cancer — baseline training(complete model)
python ".\train.py" --cancer breast --epochs 20 --batch-size 16 --lr 0.0001

breast cancer raw dataset:
C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\jpeg

Breast text/marker detection:
C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\mammo_text_detection

BREAST CLEANED DATASET:
C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\breast_cleaned_v2

Breast SSL dataset:
C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\scripts\create_ssl_dataset.py
 
Lung cancer raw dataset:
C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\chest_xray_lung

Clean/preprocessed lung dataset:
C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\lung_xray_clean_512

Classification report for both models:
cd "C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI"
python ".\evaluate_breast_expanded.py"
python ".\evaluate_lung_clean.py"

go to project:
cd "C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI"

1. Preprocessing
python ".\scripts\prepare_breast_dataset.py"

2. Mammogram inspection
python ".\scripts\inspect_mammograms.py"    

3. Text/marker detection
python ".\scripts\detect_mammo_text.py"

4. Dataset split

python ".\scripts\split_breast_dataset.py"

5. classification report for lung cancer model
python .\evaluate_lung_clean.py