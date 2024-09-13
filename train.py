import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

# 모델 정의
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=64):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = self._encoder_block(in_channels, features)
        self.encoder2 = self._encoder_block(features, features * 2)
        self.encoder3 = self._encoder_block(features * 2, features * 4)
        self.encoder4 = self._encoder_block(features * 4, features * 8)
        self.bottleneck = self._encoder_block(features * 8, features * 16)

        self.upconv4 = self._upconv_block(features * 16, features * 8)
        self.decoder4 = self._decoder_block(features * 16, features * 8)
        self.upconv3 = self._upconv_block(features * 8, features * 4)
        self.decoder3 = self._decoder_block(features * 8, features * 4)
        self.upconv2 = self._upconv_block(features * 4, features * 2)
        self.decoder2 = self._decoder_block(features * 4, features * 2)
        self.upconv1 = self._upconv_block(features * 2, features)
        self.decoder1 = self._decoder_block(features * 2, features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def _encoder_block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def _decoder_block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, features):
        return nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(kernel_size=2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(kernel_size=2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(kernel_size=2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))

        dec4 = torch.cat((self.upconv4(bottleneck), enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = torch.cat((self.upconv3(dec4), enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = torch.cat((self.upconv2(dec3), enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = torch.cat((self.upconv1(dec2), enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)  # 2개의 클래스로 예측하므로 softmax 사용

# 훈련 함수 정의
def train_model(model, dataloader, criterion, optimizer, num_epochs=25, device='cuda'):
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

# 데이터셋 로드 및 DataLoader 설정
data_path = "C:/Users/YOON JONGIN/Desktop/workspace/unet_multiple/data/jaxa_dataset"
dataset = build(data_path, return_masks=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델, 손실 함수, 옵티마이저 정의
model = UNet(in_channels=3, out_channels=2)
criterion = nn.CrossEntropyLoss()  # 다중 클래스 세그멘테이션이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
train_model(model, dataloader, criterion, optimizer, num_epochs=25, device='cuda' if torch.cuda.is_available() else 'cpu')
