import torch
import torch.optim as optim
from skimage import io
import time
from Fun.Add_Noise import batch_add_noise1, batch_add_noise2
from Fun.Rescale_OTF import process_otf_tensor
import os
from Fun.load_data import ReconsDataset2D_single
from Fun.Model_RCAN import RCAN

####### User parameters #######
corrected_sOTF = r'E:\Prj70_DeepIso_submit\sOTF\WF_sOTF.tif'
XOY_path = r'E:\Prj70_DeepIso_submit\Data\dataset\XOY'
XOZ_path = r'E:\Prj70_DeepIso_submit\Data\dataset\XOZ'
###############################

if __name__ == '__main__':
    start = time.time()
    # parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    minipatch = 128

    # dataset
    ZS3D_dataset_initial = ReconsDataset2D_single(train_gt_path=XOY_path,
                                                  train_ref_path=XOZ_path,
                                                  img_type='tif')
    ZS3D_dataset = torch.utils.data.DataLoader(ZS3D_dataset_initial, batch_size = batch_size, shuffle=True,
                                                   pin_memory=False, drop_last=True)

    model = RCAN()
    # model = Unet1_step(1,0)
    model = model.to(device)
    optimizer_denoise = optim.Adam(model.parameters(), lr=1e-3)

    OTF = io.imread(corrected_sOTF)
    OTF = OTF/OTF.max()
    OTF = torch.tensor(OTF)
    OTF = OTF.unsqueeze(dim=0)  # 在第0轴（batch）和第1轴（通道）扩展
    OTF = OTF.repeat(batch_size, 1, 1, 1)

    for epoch in range(2):
        for batch_idx, items in enumerate(ZS3D_dataset):
            gt = items['image_gt']
            ref = items['image_ref']

            OTF_rescale = process_otf_tensor(OTF, minipatch=minipatch, variance=0.01)
            up_clean = torch.abs(torch.fft.ifft2(torch.fft.fft2(gt) * torch.fft.fftshift(OTF_rescale)))
            up = batch_add_noise2(up_clean,torch.tensor(ref))

            model.train()
            pred_image = model(up.float().to(device))

            criterion = torch.nn.L1Loss(reduction='mean')
            loss = criterion(pred_image, gt.float().to(device))

            optimizer_denoise.zero_grad()
            loss.backward()
            optimizer_denoise.step()

            folder_path = "Step2_Train_Isotropic_branch"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            print("[Epoch %d] [Batch %d/%d] [loss1: %f]" % (epoch, batch_idx, len(ZS3D_dataset), loss.item()*100 ))
            if batch_idx % 200 == 1:
                io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_pred.tif',
                           pred_image[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_in.tif',
                           up[:, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_gt.tif',
                           gt[0, :, :, :].detach().cpu().numpy())

        end = time.time()
        print(f"Consuming：{end - start:.6f} s")
        torch.save(model, r"Step2_Train_Isotropic_branch\Model_isotropic_epoch_" + str(epoch + 1) + ".pth")