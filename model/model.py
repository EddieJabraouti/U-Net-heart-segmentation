import os
from PIL import Image, ImageFile
import nibabel as nib

import torch 
import torch.nn as net
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, models 
from sklearn.metrics import accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALLOWED_EXTENSIONS = {'.jpg', '.png', '.nii', ',jpeg', '.tif', '.tiff', '.webp'}

class HeartSegmentationDataset(Dataset): #processing data for model input
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for patient_id in sorted(os.listdir(root_dir)):
            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for fname in os.listdir(patient_dir):
                #Select only ED/ES image (not 4d or gt)
                if(
                    "frame" in fname
                    and not fname.endswith("_gt.nii")
                    and not fname.endswith("_gt.nii.gz")
                    and not fname.endswith("_4d.nii")
                ):
                    img_path = os.path.join(patient_dir, fname)

                    if fname.endswith(".nii.gz"):
                        gt_path = img_path.replace(".nii.gz", "_gt.nii.gz")
                    else:
                        gt_path = img_path.replace(".nii", "_gt.nii")

                    if os.path.exists(gt_path):
                        self.samples.append((img_path, gt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        image = nib.load(img_path).get_fdata()
        mask = nib.load(gt_path).get_fdata()

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        image = (image - image.mean()) / (image.std() + 1e-8) #Z score normalization - commonality in Cardiac Imagery

        #(H, W, S) -> (S, H, W)
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

         #add channel dims (1, S, H, W)
        image = image.unsqueeze(0)

        return image, mask

    #need to use encoder decoder architecture
    #using Dice loss function

#Convolution architecture to be applied to each layer
class DoubleConv3D(net.module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = net.Sequential(
            net.Conv3d(in_channels, out_channels, 3, padding=1), 
            net.BatchNorm3d(out_channels), 
            net.ReLU(inplace=True),

            net.Conv3d(out_channels, in_channels, 3, padding=1), 
            net.BatchNorm3d(out_channels),
            net.ReLU(inplace=True)
            )


class DiceLoss(net.module):
    def __init__(self, smooth=1e-6):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets = (torch.net.functional.one_hot(
            targets, num_classes=probs.shape[1]
        ).permute(0,4,1,2,3).float())

        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()



class UNet3D(net.module):
    def __init__(self, in_channels=1, num_classes = 4, base_filters=32):
        super().__init__()




#Model:
model = UNet3D(in_channels=1, num_classes=4, base_filters=32).to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#Training loop:
num_epochs = 16

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in DataLoader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs} - loss: {running_loss / len(train_load):.4f}")





