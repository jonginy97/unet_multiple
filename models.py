import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        self.height = height
        self.width = width

        # Create a grid for position encoding
        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).repeat(height, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        
        # Sine for even indices in the channel dimension, Cosine for odd indices
        pe[0::2, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe.unsqueeze(0)  # Add positional encoding to input

    
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        key = self.key(x).view(batch_size, -1, width * height)  # B x C x N
        attention = torch.bmm(query, key)  # B x N x N
        attention = torch.softmax(attention, dim=-1)  # B x N x N

        value = self.value(x).view(batch_size, -1, width * height)  # B x C x N
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, channels, width, height)  # B x C x W x H

        out = self.gamma * out + x
        return out

class UNet_tr(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=64):
        super(UNet_tr, self).__init__()
        
        features = init_features
        # Encoder blocks
        self.encoder1 = self.create_encoder_block(in_channels, features)
        self.encoder2 = self.create_encoder_block(features, features * 2)
        self.encoder3 = self.create_encoder_block(features * 2, features * 4)
        self.encoder4 = self.create_encoder_block(features * 4, features * 8)

        # Bottleneck block with Self-Attention
        self.bottleneck = nn.Sequential(
            self.create_encoder_block(features * 8, features * 16),
            SelfAttention(features * 16)
        )

        # Positional encoding for the bottleneck
        self.positional_encoding = PositionalEncoding(features * 16, 18, 18)  # Height and width should match bottleneck output size

        # Decoder blocks with upsampling
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self.create_decoder_block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self.create_decoder_block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self.create_decoder_block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self.create_decoder_block(features * 2, features)

        # Final output layer
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
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck with Self-Attention and Positional Encoding
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        bottleneck = self.positional_encoding(bottleneck)

        # Decoding path
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

        # Final output
        return self.final_conv(dec1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_tr(in_channels=3, out_channels=3)  # output 채널을 2로 유지
    model.to(device)
    summary(model, (3, 299, 299))  # 이미지 크기에 맞게 수정


    