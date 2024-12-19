import os
import torch.utils.data as data
from torchvision.transforms import Compose
import random
import itertools
import glob
import re
import numpy as np

from monai.transforms import (
    LoadImage,
    Compose,
    EnsureChannelFirst,
    SqueezeDim,
    Resize,
    Transpose,
)

# Dataset for testing --> one modality!
class dataset_single(data.Dataset):
  def __init__(self, args, input_dim, case):
    self.dataroot = args.data_dir1
    self.data_set_type = args.data_type
    self.img = sorted(glob.glob(os.path.join(os.path.join(os.path.join(self.dataroot, "images"), case), "slice_*.nii.gz")))
    self.seg = sorted(glob.glob(os.path.join(os.path.join(os.path.join(self.dataroot, "labels"), case), "slice_*.nii.gz")))
    
    self.dataset = [{"img": img, "seg": seg} for img, seg in zip(self.img, self.seg)]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    self.transforms = Compose(
            [
              LoadImage(),
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
    if self.data_set_type == "CHAOS":
      img = img.unsqueeze(0)
      seg = seg.unsqueeze(0)
    return img, seg

  def load_img(self, img_name):
    img = self.transforms(img_name)
    return img

  def __len__(self):
    return self.size

# Dataset for testing --> one modality! for nnUNet baseline!!
class dataset_single_nn_pre(data.Dataset):
  def __init__(self, args, case):
    self.dataroot = args.data_dir1
    self.data_set_type = args.data_type
    self.img = sorted(glob.glob(os.path.join(self.dataroot, f"nnUNetPlans_2d/*_{case}x*[0-9].npy")))
    self.seg = sorted(glob.glob(os.path.join(self.dataroot, f"nnUNetPlans_2d/*_{case}x*_seg.npy")))
    # OR TAKE gt_segmentations??
    
    self.dataset = [{"img": img, "seg": seg} for img, seg in zip(self.img, self.seg)]

    # CHECK statement
    for item in zip(self.dataset):
      img = item["img"]
      seg = item["seg"]
      new_img = re.sub("_seg", "", seg)
      assert new_img == img


    self.size = len(self.img)

    # setup image transformation
    self.transforms = Compose(
            [
              LoadImage(),
              SqueezeDim(dim=0),
              Resize(spatial_size=(256, 256), mode="bilinear"),
              Transpose(indices=(0, 2, 1))
            ])
  
    self.transforms_seg = Compose(
            [
              LoadImage(),
              SqueezeDim(dim=0),
              Resize(spatial_size=(256, 256), mode="nearest"),
              Transpose(indices=(0, 2, 1))
            ])

  def __getitem__(self, index):
    img = self.transforms(self.img[index])
    seg = self.transforms_seg(self.seg[index])
    return img, seg

  def __len__(self):
    return self.size



# Dataset for two modalities without segmentation masks
class dataset_unpair(data.Dataset):
  def __init__(self, args, train_cases):
    # preprocessed/ folder
    print("DATSET: ", args.data_type)
    self.data_1 = args.data_dir1
    self.data_2 = args.data_dir2
    self.cases_1 = train_cases[0]
    self.cases_2 = train_cases[1]
    self.data_set_type = args.data_type

    self.all_images_1 = sorted(glob.glob(os.path.join(self.data_1, "images/case_*")))
    images1 = [glob.glob(self.all_images_1[idx]+ "/*.nii.gz") for idx in self.cases_1]
    self.A = sorted(list(itertools.chain.from_iterable(images1)))
    
    self.all_images_2 = sorted(glob.glob(os.path.join(self.data_2, "images/case_*")))
    images2 = [glob.glob(self.all_images_2[idx]+ "/*.nii.gz") for idx in self.cases_2]
    self.B = sorted(list(itertools.chain.from_iterable(images2)))

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = args.input_dim_a
    self.input_dim_B = args.input_dim_b

    self.transforms = Compose(
            [
              LoadImage(),
            ])


  def __getitem__(self, index):
    if index > (self.B_size - 1):
      data_A = self.load_img(self.A[index])
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)])
    elif index > (self.A_size - 1): 
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)])
      data_B = self.load_img(self.B[index])
    else:
      data_A = self.load_img(self.A[index])
      data_B = self.load_img(self.B[index])

    if self.data_set_type == "CHAOS":
      data_A = data_A.unsqueeze(0)
      data_B = data_B.unsqueeze(0)
    return data_A, data_B

  def load_img(self, img_name):
    img = self.transforms(img_name)    
    return img

  def __len__(self):
    return self.dataset_size


# Dataset for two modalities without segmentation masks --> For images processed by nnUNet!!
class dataset_unpair_nn_pre(data.Dataset):
  def __init__(self, args, train_cases):
    self.data_1 = args.data_dir1
    self.data_2 = args.data_dir2
    self.cases_1 = train_cases[0]
    self.cases_2 = train_cases[1]

    images1 = [glob.glob(os.path.join(self.data_1, f"nnUNetPlans_2d/{idx}.npz")) for idx in self.cases_1]
    self.A = sorted(list(itertools.chain.from_iterable(images1)))
    
    images2 = [glob.glob(os.path.join(self.data_2, f"nnUNetPlans_2d/{idx}.npz")) for idx in self.cases_2]
    self.B = sorted(list(itertools.chain.from_iterable(images2)))

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = args.input_dim_a
    self.input_dim_B = args.input_dim_b

    self.transforms = Compose(
            [
              SqueezeDim(dim=0),
              Resize(spatial_size=(256, 256), mode="bilinear"),
              Transpose(indices=(0, 2, 1))
            ])


  def __getitem__(self, index):
    if index > (self.B_size - 1):
      data_A = self.load_img(self.A[index])
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)])
    elif index > (self.A_size - 1): 
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)])
      data_B = self.load_img(self.B[index])
    else:
      data_A = self.load_img(self.A[index])
      data_B = self.load_img(self.B[index])

    return data_A, data_B

  def load_img(self, img_name):
    data_file = np.load(img_name)['data']
    img = self.transforms(data_file)    
    return img

  def __len__(self):
    return self.dataset_size