# meme_analyzer/ml/model.py
import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class MemeDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, transform=None):
        self.items = [json.loads(line) for line in open(jsonl_file, 'r')]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.img_dir, item['file_name'])
        image = Image.open(img_path).convert('RGB')
        label = item['label']  # map to int outside if needed
        if self.transform:
            image = self.transform(image)
        return image, label

class MemeClassifierNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(54080, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
