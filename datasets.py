from pathlib import Path
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as Jaxa_mask
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    normalize = T.Compose([
        T.ToTensor(),
    ])
    return normalize


def build(data_path, return_masks=False):
    root = Path(data_path)
    assert root.exists(), f'provided Jaxa path {root} does not exist'
    img_folder = root / "images"
    ann_file = root / "annotations.json"

    dataset = JaxaDataset(img_folder, ann_file, transforms=make_Jaxa_transforms(), return_masks=return_masks)
    return dataset


# # 메인 실행 부분
# if __name__ == "__main__":
#     data_path = "./data/jaxa_dataset/"
#     dataset = build(data_path, return_masks=True)

#     # DataLoader 설정
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#     # 데이터 불러오기 예시
#     n = 0
#     for images, targets in dataloader:
#         print("Images shape:", [img.shape for img in images])
#         # print("Targets:", targets)
#         # image shape torch.Size([1, 3, 299, 299])
#         # 보여주기


#         img = images[0].permute(1, 2, 0).numpy()
#         plt.imshow(img)
#         plt.show()

#         masks = targets['masks']
#         print("Masks shape:", masks.shape)
#         # Masks shape: torch.Size([1, 2, 299, 299])
#         # 보여주기
#         if len(masks) == 
#         mask = masks[0, 0].numpy()
#         plt.imshow(mask)
#         plt.show()

#         mask = masks[0, 1].numpy()
#         plt.imshow(mask)
#         plt.show()


#         n =+ 1 
#         if n == 3: 
#             break  # 첫 번째 배치만 확인
