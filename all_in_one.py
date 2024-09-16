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

        # 디코딩 경로 및 스킵 연결
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

# JaxaDataset 정의
class JaxaDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(JaxaDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertJaxaPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(JaxaDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # transforms는 이미지만 변환하고 타겟은 그대로 유지
        if self._transforms is not None:
            img = self._transforms(img)  # 이미지만 변환

        return img, target

def convert_Jaxa_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = Jaxa_mask.frPyObjects(polygons, height, width)
        mask = Jaxa_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    
    # 마스크가 없을 경우 기본 빈 마스크 생성
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((1, height, width), dtype=torch.uint8)
    return masks

class ConvertJaxaPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            if segmentations:
                masks = convert_Jaxa_poly_to_mask(segmentations, h, w)
            else:
                masks = torch.zeros((1, h, w), dtype=torch.uint8)  # 마스크가 없을 경우 빈 마스크 생성

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        poses = [obj["pose"] for obj in anno if "pose" in obj]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep] if len(masks) > 0 else torch.zeros((1, h, w), dtype=torch.uint8)
        if keypoints is not None:
            keypoints = keypoints[keep]
        if poses:
            poses = [poses[i] for i in range(len(poses)) if keep[i]]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        if poses:
            target["poses"] = poses

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_Jaxa_transforms():
    transforms = T.Compose([
        T.Resize((299, 299)),  # 이미지와 마스크를 299x299 크기로 통일
        T.ToTensor(),
    ])
    return transforms

def build(data_path, return_masks=False):
    root = Path(data_path)
    assert root.exists(), f'provided Jaxa path {root} does not exist'
    img_folder = root / "images"
    ann_file = root / "annotations.json"

    dataset = JaxaDataset(img_folder, ann_file, transforms=make_Jaxa_transforms(), return_masks=return_masks)
    return dataset

# Custom collate function 정의
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return images, targets

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
    model = UNet(in_channels=3, out_channels=3)  # output 채널을 2로 유지
    model.to(device)

    # 모델 구조 요약
    summary(model, (3, 299, 299))  # 이미지 크기에 맞게 수정

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
    