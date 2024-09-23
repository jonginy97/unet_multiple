import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from pathlib import Path
import torchvision
from pycocotools import mask as Jaxa_mask
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from datasets import build, collate_fn
from models import UNet
from unet_trans import UNet_tr



# 학습 스크립트
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Segmentation")
    parser.add_argument('--data-path', type=str, default="./data/jaxa_dataset", help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    # 데이터셋 로드 및 DataLoader 설정
    data_path = args.data_path
    dataset = build(data_path, return_masks=True)
    # DataLoader에 custom collate function 적용
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델, 손실 함수, 옵티마이저 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_tr(in_channels=3, out_channels=3)  # output 채널을 2로 유지
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # 손실 함수를 CrossEntropyLoss로 유지
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습
    num_epochs = args.epochs
    model.train()

    # 학습 루프에서 마스크 결합 부분 수정
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            masks = [tgt['masks'].to(device) for tgt in targets]

            # 이미지들을 스택하여 배치 텐서로 변환
            images = torch.stack(images)

            # 각 객체의 마스크를 해당 객체의 클래스 인덱스 값으로 변환
            masks_combined = []
            for mask, target in zip(masks, targets):
                combined_mask = torch.zeros(mask.shape[1:], dtype=torch.long, device=device)  # 배경은 0으로 초기화
                for idx, m in enumerate(mask):
                    combined_mask[m.bool()] = idx + 1  # 각 객체에 대해 1부터 시작하는 클래스 인덱스 할당
                masks_combined.append(combined_mask)


            # 마스크들을 스택하여 배치 텐서로 변환
            masks = torch.stack(masks_combined)  # [batch_size, height, width]

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 모델 출력과 손실 계산 (raw logits 반환)
            outputs = model(images)  # outputs 크기: [batch_size, num_classes, height, width]

            # 출력과 마스크의 크기를 맞춰 손실 계산
            if outputs.size()[2:] != masks.size()[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)  # CrossEntropyLoss 사용

            # 역전파와 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")


    # 훈련된 모델로 이미지 세그멘테이션 수행
    model.eval()

    # 이미지와 마스크 시각화
    img, target = dataset[0]
    img = img.unsqueeze(0).to(device)
    output = model(img)
    output = F.softmax(output, dim=1)
    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 마스크 시각화
    plt.imshow(output)
    plt.show()
    