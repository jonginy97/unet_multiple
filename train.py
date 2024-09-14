import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from models import UNet
from datasets import build

# parse 함수
parser = argparse.ArgumentParser(description="U-Net Segmentation")
parser.add_argument('--data-path', type=str, default="./data/jaxa_dataset", help='Path to dataset')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
args = parser.parse_args()

print(args)

# 데이터셋 로드 및 DataLoader 설정
data_path = args.data_path
dataset = build(data_path, return_masks=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# 모델, 손실 함수, 옵티마이저 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=2)
model.to(device)

# 모델 구조 요약
summary(model, (3, 256, 256))  # 이미지 크기를 적절하게 수정

criterion = nn.CrossEntropyLoss()  # 다중 클래스 세그멘테이션이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 학습
num_epochs = args.epochs
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        masks = targets['masks'].to(device)

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 모델 출력과 손실 계산
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 역전파와 최적화
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        break

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")