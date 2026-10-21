import argparse, torch
from PIL import Image
from config import CONFIGS,DEVICE,IMAGE_SIZE
from models.cnn import CancerCNN
from utils import preprocess_pil

def predict(cancer,path):
    ck=torch.load(CONFIGS[cancer]['model_path'],map_location=DEVICE); classes=ck['classes']; model=CancerCNN(len(classes),False).to(DEVICE);model.load_state_dict(ck['model_state']);model.eval(); x=preprocess_pil(Image.open(path),IMAGE_SIZE).to(DEVICE)
    with torch.no_grad(): p=torch.softmax(model(x),1)[0];i=int(p.argmax())
    return classes[i],float(p[i]),{classes[j]:float(p[j]) for j in range(len(classes))}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cancer',choices=['breast','lung'],required=True);p.add_argument('--image',required=True);a=p.parse_args();print(predict(a.cancer,a.image))
