import torch
from torchvision import transforms
from PIL import Image
from model import MemeClassifierNet  

num_classes = 3  # Use the same as training
model_path = 'memesage/meme_analyzer/ml/classifier.pt'

model = MemeClassifierNet(num_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


def load_model(weights_path, num_classes):
    model = MemeClassifierNet(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(img_path, model):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    return pred

# Example usage:
result = predict('memesage/meme_analyzer/datasets/data/img/01235.png', model)

# Define label-to-class map (adjust according to your dataset)
label_map = {
    0: "Non-hateful",
    1: "Hateful"
}

# Print human-readable label
print("Predicted label:", label_map.get(result, "Unknown"))
