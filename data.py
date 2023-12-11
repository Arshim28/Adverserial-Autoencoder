import os
import json

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

config_path = './config.json'
with open(config_path) as config_file:
    config = json.load(config_file)

#Create a custom dataset using MNIST
os.makedirs('./images', exist_ok=True)
img_path = './images'

# MNIST dataset

transform = transforms.Compose([transforms.Resize(config['image_size']), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root=img_path, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=img_path, train=False, transform=transform, download=True)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)  


