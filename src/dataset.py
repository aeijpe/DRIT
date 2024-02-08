import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import glob

from monai.transforms import (
    LoadImage,
    LoadImaged,
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    SqueezeDim,
    SqueezeDimd,
    MapLabelValued,
)

# Dataset for testing
class dataset_single(data.Dataset):
  def __init__(self, args, input_dim, case):
    self.dataroot = args.data_dir1
    self.img = sorted(glob.glob(os.path.join(os.path.join(os.path.join(self.dataroot, "images"), case), "slice_*.nii.gz")))
    self.seg = sorted(glob.glob(os.path.join(os.path.join(os.path.join(self.dataroot, "labels"), case), "slice_*.nii.gz")))
    
    # self.img = sorted(glob.glob(os.path.join(self.dataroot, "images/case_1001/slice_*.nii.gz")))
    # self.seg = sorted(glob.glob(os.path.join(self.dataroot, "labels/case_1001/slice_*.nii.gz")))


    self.dataset = [{"img": img, "seg": seg} for img, seg in zip(self.img, self.seg)]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    self.transforms = Compose(
            [
              LoadImage(),
              # EnsureChannelFirst(),
              # SqueezeDim(dim=-1),
              Normalize(mean=[0.5], std=[0.5]),
            ])
  
    self.transforms_seg = Compose(
            [
              LoadImage(),
              EnsureChannelFirst(),
              SqueezeDim(dim=-1),
            ])

  def __getitem__(self, index):
    img = self.transforms(self.img[index])
    seg = LoadImage()(self.seg[index])
    return img, seg

  def load_img(self, img_name):
    img = self.transforms(img_name)
    return img

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, args):
    # preprocessed/ folder
    self.data_1 = args.data_dir1
    self.data_2 = args.data_dir2

    # A --> take only 18 cases for training unet
    train_images_A1 = sorted(glob.glob(os.path.join(self.data_1, "images/case_100*/slice_*.nii.gz"))) # "images/case_100*/slice_*.nii.gz"))) 
    train_images_A2 = sorted(glob.glob(os.path.join(self.data_1, "images/case_101[0-8]/slice_*.nii.gz"))) 
    
    self.A = train_images_A1 + train_images_A2

    # B --> take only 18 cases for training unet
    train_images_B1 = sorted(glob.glob(os.path.join(self.data_2, "images/case_100*/slice_*.nii.gz"))) #"images/case_100*/slice_*.nii.gz"))) 
    train_images_B2 = sorted(glob.glob(os.path.join(self.data_2, "images/case_101[0-8]/slice_*.nii.gz"))) 

    self.B = train_images_B1 + train_images_B2

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = args.input_dim_a
    self.input_dim_B = args.input_dim_b

    self.transforms = Compose(
            [
              LoadImage(),
              Normalize(mean=[0.5], std=[0.5]),
            ])


  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index])
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)])
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)])
      data_B = self.load_img(self.B[index])
    return data_A, data_B

  def load_img(self, img_name):
    img = self.transforms(img_name)    
    return img

  def __len__(self):
    return self.dataset_size
