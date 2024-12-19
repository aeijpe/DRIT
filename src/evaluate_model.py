import torch
from options import TestOptions
from dataset import dataset_unpair, dataset_unpair_nn_pre
import os
from model import DRIT
from monai.transforms import ScaleIntensity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
import glob
from utils import set_seed, get_train_cases

# Evaluate model
def evaluate(opts, loader):

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # Validation metric
    lpips_alex_CTT_MRF = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    lpips_alex_CT = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    lpips_alex_MRT_CTF = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    lpips_alex_MRI = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)

    lpips_list_CTT_MRF = []
    lpips_list_CT = []
    lpips_list_MRT_CTF = []
    lpips_list_MRI = []

    print('\n--- Evaluating ---')
    for idx1, (img_ct, img_mri) in enumerate(loader):
        # print("in outer loop")
        img_ct = img_ct.cuda()
        img_mri = img_mri.cuda()
        for idx2 in range(opts.num):
            # print("in inner loop")
            with torch.no_grad():
                # Get translated images from both domains
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

            # Compute LPIPS Metric
            # lpips_list_CTT_MRF.append(lpips_alex_CTT_MRF(img_ct_3, img_mri_fake_3))
            # lpips_list_MRT_CTF.append(lpips_alex_MRT_CTF(img_mri_3, img_ct_fake_3))
            lpips_list_CT.append(lpips_alex_CT(img_ct_3, img_ct_fake_3))
            lpips_list_MRI.append(lpips_alex_MRI(img_mri_3, img_mri_fake_3))
    
    print(f"Model: {opts.resume[-9:]}")
    print('\n--- Evaluation Done ---')

    # Return LPIPS for both directions
    return {"Model": opts.resume[-9:], 
            # "lpips_CTT_MRF": torch.mean(torch.stack(lpips_list_CTT_MRF)).item(), 
            # "lpips_MRT_CTF": torch.mean(torch.stack(lpips_list_MRT_CTF)).item(), 
            "lpips_CT": torch.mean(torch.stack(lpips_list_CT)).item(), 
            "lpips_MRI": torch.mean(torch.stack(lpips_list_MRI)).item()
            }


def main():
    parser = TestOptions()
    opts = parser.parse()
    set_seed(1)
    # CHANGE accordingly
    dir_model = f"../results/{opts.name}/"
    
    model_list = sorted(glob.glob(os.path.join(dir_model, "*.pth")))

    # data loader
    print('\n--- load dataset ---')
    train_cases = get_train_cases(opts)
    print("TRAIN cases: ", train_cases)

    if opts.data_type == 'nnUNet':
        print("Getting nnunet data")
        dataset = dataset_unpair_nn_pre(opts, train_cases)
    else:
        dataset = dataset_unpair(opts, train_cases)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)
    print("len loader: ", loader.__len__())
   
    all_metrics_models = []
    all_metrics_lpips_CTT_MRF = []
    all_metrics_lpips_MRT_CTF = []
    all_metrics_lpips_CT = []
    all_metrics_lpips_MRI = []

    # COmpute metric for all saved models
    for model in model_list:
        opts.resume = model
        res = evaluate(opts, loader)
        all_metrics_models.append(res["Model"])
        # all_metrics_lpips_CTT_MRF.append(res["lpips_CTT_MRF"])
        # all_metrics_lpips_MRT_CTF.append(res["lpips_MRT_CTF"])
        all_metrics_lpips_CT.append(res["lpips_CT"])
        all_metrics_lpips_MRI.append(res["lpips_MRI"])

    # Print the results 
    # These results are used in the `results folder`!
    print("Models:", all_metrics_models)
    print("squeeze lpips_CTT_MRF:", all_metrics_lpips_CTT_MRF)
    print("squeeze lpips_MRT_CTF:", all_metrics_lpips_MRT_CTF) 
    print("squeeze lpips_CT:", all_metrics_lpips_CT)
    print("squeeze lpips_MRI:", all_metrics_lpips_MRI) 
        

    
if __name__ == '__main__':
  main()