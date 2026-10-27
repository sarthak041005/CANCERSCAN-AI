import streamlit as st
import torch
import numpy as np
from PIL import Image

from config import CONFIGS, DEVICE, IMAGE_SIZE, MODEL_DIR
from models.cnn import CancerCNN
from utils import preprocess_pil
from models.breast_models import load_breast_models, predict_single_breast

st.set_page_config(page_title='CancerScan AI', page_icon='🩺', layout='wide')
st.title('🩺 CancerScan AI')
st.subheader('Breast Cancer and Lung Cancer Detection using Deep Learning')
st.warning('⚠️ Research/educational prototype only. This is not a medical diagnosis.')

choice = st.sidebar.selectbox('Select model', ['Breast Cancer', 'Lung Cancer'])
cancer = 'breast' if choice == 'Breast Cancer' else 'lung'

st.sidebar.markdown('---')
st.sidebar.markdown(f'**Compute Device:** `{DEVICE}`')

# -----------------------------------------------------------------------------
# LUNG CANCER MODULE (EXACT UNMODIFIED LOGIC)
# -----------------------------------------------------------------------------
if cancer == 'lung':
    @st.cache_resource
    def load_lung():
        ck_path = CONFIGS['lung']['model_path']
        if not ck_path.is_file():
            ck_path = MODEL_DIR / 'lung' / 'lung_resnet18_clean_best.pth'
        if not ck_path.is_file():
            raise FileNotFoundError(f'Trained lung cancer model not found at {ck_path}')

        ck = torch.load(ck_path, map_location=DEVICE)
        classes = ck.get('classes', CONFIGS['lung']['classes'])
        state = ck.get('model_state_dict', ck.get('model_state', ck))

        m = CancerCNN(len(classes), False).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        return m, classes

    up = st.file_uploader('Upload medical image', type=['jpg', 'jpeg', 'png', 'bmp'])
    if up:
        img = Image.open(up).convert('RGB')
        st.image(img, caption='Uploaded image', width=500)

        try:
            m, classes = load_lung()
        except FileNotFoundError:
            st.error('Trained model not found. Train this model first.')
            st.stop()

        x = preprocess_pil(img, IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(m(x), 1)[0].cpu().numpy()

        i = int(np.argmax(probs))
        st.success(f'Prediction: {classes[i].upper()}')
        st.metric('Confidence', f'{probs[i]*100:.2f}%')

        st.markdown('### Class probabilities')
        for label, p in zip(classes, probs):
            st.write(f'**{label}** — {p*100:.2f}%')
            st.progress(float(p))

    st.caption('CancerScan AI | PyTorch | CNN | Streamlit')

# -----------------------------------------------------------------------------
# CHAMPION BREAST CANCER MODULE (SINGLE INPUT INFERENCE)
# -----------------------------------------------------------------------------
else:
    @st.cache_resource
    def load_breast_champion():
        models_dict = load_breast_models(MODEL_DIR, DEVICE)
        return models_dict

    with st.sidebar.expander('📊 Validated Test Benchmark', expanded=False):
        st.markdown(
            """
            | Metric | Validated Score |
            | :--- | :--- |
            | **Test Accuracy** | **85.56%** |
            | **Malignant Recall (Sensitivity)** | **90.91%** |
            | **Benign Precision (NPV)** | **94.00%** |
            | **Specificity** | **82.46%** |
            | **ROC-AUC** | **0.8958** |
            """
        )

    up = st.file_uploader('Upload medical image', type=['jpg', 'jpeg', 'png', 'bmp'])
    if up:
        img = Image.open(up).convert('RGB')
        st.image(img, caption='Uploaded image', width=500)

        try:
            breast_models = load_breast_champion()
        except FileNotFoundError:
            st.error('Trained breast model not found. Train this model first.')
            st.stop()

        with st.spinner('Analyzing mammogram with champion deep learning ensemble...'):
            res = predict_single_breast(
                image=img,
                models_dict=breast_models,
                device=DEVICE,
                use_tta=True,
                optimal_threshold=0.4910
            )

        classes = res['classes']  # ['BENIGN', 'MALIGNANT']
        probs = res['probs']      # [benign_prob, malignant_prob]
        pred = res['prediction']

        if pred == 'MALIGNANT':
            st.error(f'🚨 Prediction: {pred}')
        else:
            st.success(f'🛡️ Prediction: {pred}')

        st.metric('Confidence', f"{res['confidence']:.2f}%")

        st.markdown('### Class probabilities')
        for label, p in zip(classes, probs):
            st.write(f'**{label}** — {p*100:.2f}%')
            st.progress(float(p))

        st.markdown(f"**Risk Level:** `{res['risk_level']}`")
        st.markdown(f"> **Clinical Recommendation:** {res['recommendation']}")

        with st.expander('🔍 Model Ensemble Breakdown', expanded=False):
            for name, p in res['component_probs'].items():
                st.write(f"**{name}** (Malignancy Risk): `{p*100:.2f}%`")
                st.progress(float(p))

    st.caption('CancerScan AI | Champion Breast Cancer Ensemble | PyTorch | Streamlit')
