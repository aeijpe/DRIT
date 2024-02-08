import torch
from options import TestOptions
from dataset import dataset_unpair
import os
from saver import save_imgs
from model import DRIT
from monai.transforms import ScaleIntensity, AsChannelLast
from saver import Saver
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
from torchmetrics.image.kid import KernelInceptionDistance
import glob

def evaluate(opts, loader):

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # Evaluation metrics
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    kid_MRI = KernelInceptionDistance(subsets=10, subset_size=60, normalize=True)
    kid_CT = KernelInceptionDistance(subsets=10, subset_size=60, normalize=True)

    
    lpips_list_MRI = []
    lpips_list_CT = []
    # test
    print('\n--- Evaluating ---')
    for idx1, (img_ct, img_mri) in enumerate(loader):
        img_ct = img_ct.cuda()
        img_mri = img_mri.cuda()
        for idx2 in range(opts.num):
            with torch.no_grad():
                img_fake_MRI = model.test_forward(img_ct, a2b=1)
                img_fake_CT = model.test_forward(img_mri, a2b=0) 
                img_fake_MRI = ScaleIntensity()(img_fake_MRI).detach().cpu()
                img_fake_CT = ScaleIntensity()(img_fake_CT).detach().cpu()
            
            img_mri_copy = ScaleIntensity()(img_mri).detach().cpu()
            img_mri_3 = torch.cat((img_mri_copy, img_mri_copy, img_mri_copy), dim=1)
            img_ct_copy = ScaleIntensity()(img_ct).detach().cpu()
            img_ct_3 = torch.cat((img_ct_copy, img_ct_copy, img_ct_copy), dim=1)

            img_mri_fake_3 = torch.cat((img_fake_MRI, img_fake_MRI, img_fake_MRI), dim=1)
            img_ct_fake_3 = torch.cat((img_fake_CT, img_fake_CT, img_fake_CT), dim=1)            

            lpips_list_MRI.append(lpips(img_mri_fake_3, img_mri_3))
            lpips_list_CT.append(lpips(img_ct_fake_3, img_ct_3))

            kid_MRI.update(img_mri_fake_3, real=False)
            kid_MRI.update(img_mri_3, real=True)

            kid_CT.update(img_ct_fake_3, real=False)
            kid_CT.update(img_ct_3, real=True)
    
    print(f"Model: {opts.resume[-9:]}")
    print(f'lpips_MRI: {torch.mean(torch.stack(lpips_list_MRI))}')
    print(f'lpips_CT: {torch.mean(torch.stack(lpips_list_CT))}')
    print(f'kid_MRI: {kid_MRI.compute()}')
    print(f'kid_CT: {kid_CT.compute()}')
    print('\n--- Evaluation Done ---')


def main():
    parser = TestOptions()
    opts = parser.parse()
    dir_model = "../results/reproduce_all_concat/"
    model_list = sorted(glob.glob(os.path.join(dir_model, "*.pth")))

    # data loader
    print(f'\n--- load dataset: ---')
    dataset = dataset_unpair(opts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=20, num_workers=opts.nThreads)

    for model in model_list:
        opts.resume = model
        evaluate(opts, loader)

    
if __name__ == '__main__':
  main()