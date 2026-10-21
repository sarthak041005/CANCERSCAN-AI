import streamlit as st, torch, numpy as np
from PIL import Image
from config import CONFIGS,DEVICE,IMAGE_SIZE
from models.cnn import CancerCNN
from utils import preprocess_pil

st.set_page_config(page_title='CancerScan AI',page_icon='🩺',layout='wide')
st.title('🩺 CancerScan AI');st.subheader('Breast Cancer and Lung Cancer Detection using Deep Learning')
st.warning('Research/educational prototype only. This is not a medical diagnosis.')
choice=st.sidebar.selectbox('Select model',['Breast Cancer','Lung Cancer']); cancer='breast' if choice=='Breast Cancer' else 'lung'
@st.cache_resource
def load(c):
    ck=torch.load(CONFIGS[c]['model_path'],map_location=DEVICE); m=CancerCNN(len(ck['classes']),False).to(DEVICE);m.load_state_dict(ck['model_state']);m.eval();return m,ck['classes']
up=st.file_uploader('Upload medical image',type=['jpg','jpeg','png','bmp'])
if up:
    img=Image.open(up).convert('RGB');st.image(img,caption='Uploaded image',width=500)
    try:m,classes=load(cancer)
    except FileNotFoundError:st.error('Trained model not found. Train this model first.');st.stop()
    x=preprocess_pil(img,IMAGE_SIZE).to(DEVICE)
    with torch.no_grad(): probs=torch.softmax(m(x),1)[0].cpu().numpy()
    i=int(np.argmax(probs));st.success(f'Prediction: {classes[i].upper()}');st.metric('Confidence',f'{probs[i]*100:.2f}%')
    st.markdown('### Class probabilities')
    for label,p in zip(classes,probs):st.write(f'**{label}** — {p*100:.2f}%');st.progress(float(p))
st.caption('CancerScan AI | PyTorch | CNN | Streamlit')
