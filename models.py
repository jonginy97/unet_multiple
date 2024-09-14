import torch
import torch.nn as nn

from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
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
        """ 인코더 블록: Conv2D + BatchNorm2D + ReLU 두 번 반복 """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def _decoder_block(self, in_channels, features):
        """ 디코더 블록: Conv2D + BatchNorm2D + ReLU 두 번 반복 """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, features):
        """ 업샘플링을 위한 ConvTranspose2D """
        return nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2)

    def forward(self, x):
        # 인코딩 경로
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(kernel_size=2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(kernel_size=2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(kernel_size=2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))

        # 디코딩 경로 및 스킵 연결
        dec4 = torch.cat((self.upconv4(bottleneck), enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = torch.cat((self.upconv3(dec4), enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = torch.cat((self.upconv2(dec3), enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = torch.cat((self.upconv1(dec2), enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

# 모델 요약 출력 예시
if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1, init_features=64)
    # print(model)
    summary(model, (3, 256, 256), device='cpu')