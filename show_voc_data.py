import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np
import torch

# Custom transform for both image and mask
class ToTensor:
    def __call__(self, img, mask):
        img = transforms.ToTensor()(img)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        return img, mask

# Function to visualize image, mask, and class information
def show_image_and_mask(dataset, index):
    img, mask = dataset[index]
    
    img_np = img.numpy().transpose((1, 2, 0))
    mask_np = mask.numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # axs[0].imshow(img_np)
    # axs[0].set_title('Image')
    # axs[0].axis('off')
    
    # axs[1].imshow(mask_np, cmap='gray')
    # axs[1].set_title('Mask')
    # axs[1].axis('off')
    
    unique_labels = np.unique(mask_np) # unique means unique values in the mask
    filtered_labels = [label for label in unique_labels if label < len(VOC_CLASSES)]
    class_labels = ', '.join([VOC_CLASSES[label] for label in filtered_labels])
    # axs[2].text(0.5, 0.5, f'Classes: {class_labels}', fontsize=12, ha='center')
    # axs[2].set_title('Classes')
    # axs[2].axis('off')
    
    # plt.tight_layout()
    # plt.show()

    print(f'Image shape: {img_np.shape}')
    print(f'Mask shape: {mask_np.shape}')
    print(f'Unique labels in mask: {unique_labels}')
    print(f'Filtered labels in mask: {filtered_labels}')


if __name__ == '__main__':


# Define VOC2012 class names
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
    ]
    
    # Data loading and transformation
    transform = ToTensor()

    # Load the dataset
    dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transforms=transform)
    
    # Show the first sample
    show_image_and_mask(dataset, 0)