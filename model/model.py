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

            for subdir in os.listdir(patient_dir):
                subdir_path = os.path.join(patient_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                for fname in os.listdir(subdir_path):
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


#Questions i must ask myself
"""
-Is my netowork following the downsampling of 2 3x3 convolutions followed by ReLu and a 2x2 max pooling operation with stride 2?
- Are the number of feature channels being doubled per each downsampling step?
-Is the upsampling step a 2x2 convolution  that halves the number of feature channels?
- am i concatentating with the correspondingly cropped feature map, following 2 3x3 convolutions with ReLu for a total of 23 convolutional layers?
-are we using the input images and their corresponding segmentation masks to train? do i need SGD as part of my optimization
- Is softmax being used? 
- Should i introduce a weight map, pre computed(before training) for each ground truth segmentation to compensate different frequency of pixels from a certain class 
and force the netowrk to learn small seperation borders that we introduce between touching parts of the heart
-Although it isnt entirely from scratch are my weight being initalized from a gausian distribution with sd: root(2/N), N= #incoming noder per neuron

"""
#Convolution architecture to be applied to each layer
class DoubleConv3D(net.Module):
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


class DiceLoss(net.Module):
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


#Actual UNet encode/decode architecture implemented in this class
class UNet3D(net.Module):
    def __init__(self, in_channels=1, num_classes = 4, base_filters=32):
        super().__init__()
        #encode - Multiple Convolutional layers to downsize image and extract features

        self.enc1 = DoubleConv3D(in_channels, base_filters)
        self.enc2 = DoubleConv3D(in_channels, base_filters * 2)
        self.enc3 = DoubleConv3D(in_channels * 2, base_filters * 4)
        self.enc4 = DoubleConv3D(in_channels * 4, base_filters * 8)

        self.pool = net.MaxPool3d(2)

        #bottleneck
        self.bottleneck = DoubleConv3D(base_filters  * 8, base_filters * 16)



#sanity check
ds = HeartSegmentationDataset('database/training')
print(len(ds))
img, msk = ds[0]
print(img.shape, msk.shape) #Should be (1, D, H, W)  (D, H, W)

#loading data for test, val, test
train_set = HeartSegmentationDataset('database/training')
test_set = HeartSegmentationDataset('database/testing')


train_load = DataLoader(
    train_set,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory = torch.cuda.is_available()
)
test_load = DataLoader(
    test_set,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory = torch.cuda.is_available()
)

#Model:
model = UNet3D(in_channels=1, num_classes=4, base_filters=32).to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#Training loop:
num_epochs = 16

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_load:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs} - loss: {running_loss / len(train_load):.4f}")

    """""
    model.eval()
    test_masks, test_seg = [], []
    with torch.no_grad(): #with no gradients to save mem
        for images, masks in test_load:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            #get generated segmentation
            #get Ground truth segmentation
        accuracy = accuracy_score(ground truth seg, generated seg)
        print(f"Test Accuracy: {accuracy: .4f}")
    """""








