import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from models import UNet
from datasets import build



# 데이터셋 로드 및 DataLoader 설정
data_path = "C:/Users/YOON JONGIN/Desktop/workspace/unet_multiple/data/jaxa_dataset"
dataset = build(data_path, return_masks=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델, 손실 함수, 옵티마이저 정의
model = UNet(in_channels=3, out_channels=2)
criterion = nn.CrossEntropyLoss()  # 다중 클래스 세그멘테이션이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)



model.to(device)
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

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")