# meme_analyzer/ml/train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MemeDataset, MemeClassifierNet

train_jsonl = 'memesage/meme_analyzer/datasets/data/train.jsonl'
img_dir = 'memesage/meme_analyzer/datasets/data'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = MemeDataset(
    jsonl_file=train_jsonl,
    img_dir=img_dir,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

num_classes = 3  # Set the actual number of classes here
model = MemeClassifierNet(num_classes)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

if __name__ == "__main__":
    for epoch in range(5):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")
    torch.save(model.state_dict(), 'memesage/meme_analyzer/ml/classifier.pt')
