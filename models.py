import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=64):  # out_channels=2로 유지
        super(UNet, self).__init__()
        
        features = init_features
        # Encoder blocks
        self.encoder1 = self.create_encoder_block(in_channels, features)
        self.encoder2 = self.create_encoder_block(features, features * 2)
        self.encoder3 = self.create_encoder_block(features * 2, features * 4)
        self.encoder4 = self.create_encoder_block(features * 4, features * 8)
        self.bottleneck = self.create_encoder_block(features * 8, features * 16)

        # Decoder blocks with upsampling
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self.create_decoder_block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self.create_decoder_block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self.create_decoder_block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self.create_decoder_block(features * 2, features)

        # Final output layer (No activation function here)
        self.final_conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def create_encoder_block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def create_decoder_block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 인코딩 경로
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        upconv4_out = self.upconv4(bottleneck)
        upconv4_out = F.interpolate(upconv4_out, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((upconv4_out, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        upconv3_out = self.upconv3(dec4)
        upconv3_out = F.interpolate(upconv3_out, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((upconv3_out, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        upconv2_out = self.upconv2(dec3)
        upconv2_out = F.interpolate(upconv2_out, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((upconv2_out, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        upconv1_out = self.upconv1(dec2)
        upconv1_out = F.interpolate(upconv1_out, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((upconv1_out, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # No activation function, raw logits
        return self.final_conv(dec1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3)  # output 채널을 2로 유지
    model.to(device)
    summary(model, (3, 299, 299))  # 이미지 크기에 맞게 수정


    