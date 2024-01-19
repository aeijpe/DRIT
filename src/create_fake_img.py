import torch
from options import TestOptions
from dataset import dataset_single
import os
from saver import save_imgs
from model import DRIT

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  nr_cases = os.listdir(os.path.join(opts.data_dir1, "images"))
  slice_nr = 0

  result_dir = os.path.join(opts.result_dir, opts.name)

  for case in nr_cases:
    # data loader
    print('\n--- load dataset ---')
    dataset = dataset_single(opts, opts.input_dim_a)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # directory
    result_dir = os.path.join(result_dir, case)
    os.makedirs(result_dir, exist_ok=True)

    # test
    print('\n--- testing ---')
    for idx1, (img1, seg) in enumerate(loader):
      print('{}/{}'.format(idx1, len(loader)))
      img1 = img1.cuda()
      imgs = []
      names = []
      labels = []
      for idx2 in range(opts.num):
        with torch.no_grad():
          img = model.test_forward(img1, a2b=opts.a2b)
        imgs.append(img)
        labels.append(seg)
        names.append(f'slice_{slice_nr}_{idx2}')
      slice_nr += 1
      save_imgs(imgs, labels, names, result_dir)

  return
     

if __name__ == '__main__':
  main()