import torch
from options import TestOptions
from dataset import dataset_unpair
import os
from model import DRIT
from monai.transforms import ScaleIntensity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
import glob

# Evaluate model
def evaluate(opts, loader):

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # Validation metric
    lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)

    lpips_list_CTT_MRF = []
    lpips_list_MRT_CTF = []

    print('\n--- Evaluating ---')
    for idx1, (img_ct, img_mri) in enumerate(loader):
        img_ct = img_ct.cuda()
        img_mri = img_mri.cuda()
        for idx2 in range(opts.num):
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
            lpips_list_CTT_MRF.append(lpips_alex(img_ct_3, img_mri_fake_3))
            lpips_list_MRT_CTF.append(lpips_alex(img_mri_3, img_ct_fake_3))
    
    print(f"Model: {opts.resume[-9:]}")
    print('\n--- Evaluation Done ---')

    # Return LPIPS for both directions
    return {"Model": opts.resume[-9:], "lpips_CTT_MRF": torch.mean(torch.stack(lpips_list_CTT_MRF)).item(), "lpips_MRT_CTF": torch.mean(torch.stack(lpips_list_MRT_CTF)).item()}


def main():
    parser = TestOptions()
    opts = parser.parse()
    # CHANGE accordingly
    dir_model = f"../results/{opts.data_type}_run_fold_{opts.cases_folds}/"
    
    model_list = sorted(glob.glob(os.path.join(dir_model, "*.pth")))

    # Get the folds for the dataset on which the model is trained, to evaluate on
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

    # data loader
    print(f'\n--- load dataset: ---')
    dataset = dataset_unpair(opts, train_cases)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)
   
    all_metrics_models = []
    all_metrics_lpips_CTT_MRF = []
    all_metrics_lpips_MRT_CTF = []

    # COmpute metric for all saved models
    for model in model_list:
        opts.resume = model
        res = evaluate(opts, loader)
        all_metrics_models.append(res["Model"])
        all_metrics_lpips_CTT_MRF.append(res["lpips_CTT_MRF"])
        all_metrics_lpips_MRT_CTF.append(res["lpips_MRT_CTF"])

    # Print the results 
    # These results are used in the `results folder`!
    print("Models:", all_metrics_models)
    print("squeeze lpips_CTT_MRF:", all_metrics_lpips_CTT_MRF)
    print("squeeze lpips_MRT_CTF:", all_metrics_lpips_MRT_CTF) 
        

    
if __name__ == '__main__':
  main()