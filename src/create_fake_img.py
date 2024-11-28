import torch
from options import TestOptions
from dataset import dataset_single, dataset_single_nn_pre
import os
from saver import save_imgs, save_imgs_mmwhs, save_imgs_nnUNet
from model import DRIT
from monai.transforms import ScaleIntensity
from saver import Saver
import glob
from utils import set_seed

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()
  set_seed(1)

  result_dir = os.path.join(opts.result_dir, f'fold_{opts.cases_folds}')

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # For all patients
  for case in range(20):
    # data loader
    print(f'\n--- load dataset: {case} ---')
    if opts.data_type == 'nnUNet':
        dataset = dataset_single_nn_pre(opts, case)
    else:
        dataset = dataset_single(opts, opts.input_dim_a, case)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

    

    # test
    print('\n--- testing ---')
    slice_nr = 0
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

        img = img.detach().cpu()
        imgs.append(img)
        origs.append(orig)
        labels.append(seg)
        names.append(f'{opts.name_new_ds}_{case}x{slice_nr}')
      slice_nr += 1

      # save image as nii.gz file
      if opts.data_type == "MMWHS":
        save_imgs_mmwhs(imgs, labels, origs, names, result_dir, case)
      elif opts.data_type == "nnUNet":
        # CHECK if this goes correctly!
        save_imgs_nnUNet(imgs, labels, origs, names, result_dir)
      else:
        save_imgs(imgs, labels, origs, names, result_dir, case)
     

if __name__ == '__main__':
  main()