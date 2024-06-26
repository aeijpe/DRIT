import torch
from options import TestOptions
from dataset import dataset_single
import os
from saver import save_imgs, save_imgs_mmwhs
from model import DRIT
from monai.transforms import ScaleIntensity
from saver import Saver
import glob

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # Get the folds for the dataset on which the model is trained, to translate those source images to the target domain
  if opts.cases_folds == 0:
    train_cases = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19]
  elif opts.cases_folds == 1:
    train_cases = [0,1,2,4,6,7,9,10,12,13,14,15,16,17,18,19]
  elif opts.cases_folds == 2:
    train_cases = [0,1,3,4,5,6,7,8,9,10,11,12,14,15,17,19]
  elif opts.cases_folds == 3:
    train_cases = [0,1,2,3,5,6,7,8,10,11,13,14,15,16,17,18]
  elif opts.cases_folds == 4:
    train_cases = [0,1,2,3,4,5,8,9,11,12,13,15,16,17,18,19]

  all_images = sorted(os.listdir(os.path.join(opts.data_dir1, "images")))
  nr_cases = [all_images[idx]for idx in train_cases]
  print("train cases: ", train_cases)
  print('Files of cases', nr_cases)
  slice_nr = 0

  result_dir = os.path.join(opts.result_dir, opts.name)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # For all patients
  for case in nr_cases:
    # data loader
    print(f'\n--- load dataset: {case} ---')
    dataset = dataset_single(opts, opts.input_dim_a, case)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

    # test
    print('\n--- testing ---')
    for idx1, (img1, seg) in enumerate(loader):
      print('{}/{}'.format(idx1, len(loader)))
      img1 = img1.cuda()
      orig = img1.detach().cpu()
      imgs = []
      origs = []
      names = []
      labels = []
      for idx2 in range(opts.num):
        with torch.no_grad():
          # get translated image
          img = model.test_forward(img1, a2b=opts.a2b)
          img = ScaleIntensity()(img)
        img = img.detach().cpu()
        imgs.append(img)
        origs.append(orig)
        labels.append(seg)
        names.append(f'slice_{slice_nr}_{idx2}')
      slice_nr += 1

      # save image as nii.gz file
      if opts.data_type == "MMWHS":
        save_imgs_mmwhs(imgs, labels, origs, names, result_dir, case)
      else:
        save_imgs(imgs, labels, origs, names, result_dir, case)
     

if __name__ == '__main__':
  main()