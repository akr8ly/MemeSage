# meme_analyzer/ml/inference.py
import torch
from torchvision import transforms
from PIL import Image
from model import MemeClassifierNet

def load_model(weights_path, num_classes):
    model = MemeClassifierNet(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict(img_path, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    return pred
