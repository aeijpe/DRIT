import torch
from options import TestOptions
from dataset import dataset_single
import os
from saver import save_imgs
from model import DRIT
from monai.transforms import ScaleIntensity, AsChannelLast
from saver import Saver

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  nr_cases = sorted(os.listdir(os.path.join(opts.data_dir1, "images")))[0:18]
  print(nr_cases)
  slice_nr = 0

  result_dir = os.path.join(opts.result_dir, opts.name)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

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
          img = model.test_forward(img1, a2b=opts.a2b)
          #img = ScaleIntensity()(img)
        img = img.detach().cpu()
        imgs.append(img)
        origs.append(orig)
        labels.append(seg)
        names.append(f'slice_{slice_nr}_{idx2}')
      slice_nr += 1
      save_imgs(imgs, labels, origs, names, result_dir, case)
    
    # for now only check case 1001
    #return
     

if __name__ == '__main__':
  main()