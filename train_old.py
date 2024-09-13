import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np
from PIL import Image

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SegmentationClassifier(nn.Module):
    def __init__(self):
        super(SegmentationClassifier, self).__init__()
        self.unet = UNet()
        self.fc = nn.Sequential(
            nn.Linear(128 * 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Assuming binary classification
        )

    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)  # Ensure output is in the range [0, 1]
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# Custom transform for both image and mask
class ToTensorAndNormalize:
    def __init__(self, size):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, img, mask):
        img = self.transform(img)
        mask = transforms.Resize(self.size)(mask)
        mask = transforms.ToTensor()(mask)
        return img, mask


def main():
    # Data loading and transformation
    transform = ToTensorAndNormalize((128, 128))

    train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transforms=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    # Check for GPU availability and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the segmentation classifier
    model = SegmentationClassifier().to(device)

    # Loss and optimizer
    segmentation_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(1):
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Dummy labels for binary classification
            labels = torch.randint(0, 2, (images.size(0),)).to(device)

            # Forward pass
            segmentation_outputs = model.unet(images)
            classification_outputs = model(images)

            # Loss calculation
            segmentation_loss = segmentation_criterion(segmentation_outputs, masks)
            classification_loss = classification_criterion(classification_outputs, labels)
            loss = segmentation_loss + classification_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/5], Segmentation Loss: {segmentation_loss.item():.4f}, Classification Loss: {classification_loss.item():.4f}')

    print('Training complete')

if __name__ == '__main__':
    main()