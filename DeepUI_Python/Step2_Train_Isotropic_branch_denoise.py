import torch
import torch.optim as optim
from skimage import io
import time
from Fun.Add_Noise import batch_add_noise2
from Fun.Rescale_OTF import process_otf_tensor
from Fun.SN2N import normalize2d, block2d_2
from Fun.load_data import ReconsDataset2D_single
from Fun.Model_RCAN import RCAN
import os

####### User parameters #######
corrected_sOTF = r'E:\Prj70_DeepIso_submit\sOTF\ISM_sOTF.tif'
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
    ZS3D_dataset_initial = ReconsDataset2D_single(train_gt_path = XOY_path,
                                                  train_ref_path = XOZ_path,
                                                  img_type = 'tif')
    ZS3D_dataset = torch.utils.data.DataLoader(ZS3D_dataset_initial, batch_size = batch_size, shuffle = True,
                                                   pin_memory = False, drop_last=True)

    model = RCAN()
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
            gt1, gt2, gt3, gt4 = block2d_2(gt)
            gt_1 = normalize2d(gt1)
            gt_2 = normalize2d(gt2)
            gt_3 = normalize2d(gt1)
            gt_4 = normalize2d(gt2)
            ref = normalize2d(ref)

            OTF_rescale = process_otf_tensor(OTF, minipatch=minipatch, variance=0.01)

            up_clean = torch.abs(torch.fft.ifft2(torch.fft.fft2(gt1) * torch.fft.fftshift(OTF_rescale)))
            up1 = batch_add_noise2(up_clean,torch.tensor(ref))
            up_clean = torch.abs(torch.fft.ifft2(torch.fft.fft2(gt2) * torch.fft.fftshift(OTF_rescale)))
            up2 = batch_add_noise2(up_clean,torch.tensor(ref))
            up_clean = torch.abs(torch.fft.ifft2(torch.fft.fft2(gt3) * torch.fft.fftshift(OTF_rescale)))
            up3 = batch_add_noise2(up_clean,torch.tensor(ref))
            up_clean = torch.abs(torch.fft.ifft2(torch.fft.fft2(gt4) * torch.fft.fftshift(OTF_rescale)))
            up4 = batch_add_noise2(up_clean,torch.tensor(ref))

            model.train()

            pred_image1 = model(up1.float().to(device))
            pred_image2 = model(up2.float().to(device))
            pred_image3 = model(up3.float().to(device))
            pred_image4 = model(up4.float().to(device))

            criterion = torch.nn.L1Loss(reduction='mean')
            loss1 = criterion(pred_image1, gt1.float().to(device)) + criterion(pred_image1, gt2.float().to(device))
            loss2 = criterion(pred_image2, gt1.float().to(device)) + criterion(pred_image2, gt2.float().to(device))
            loss3 = criterion(pred_image3, gt1.float().to(device)) + criterion(pred_image3, gt2.float().to(device))
            loss4 = criterion(pred_image4, gt1.float().to(device)) + criterion(pred_image4, gt2.float().to(device))
            loss5 = criterion(pred_image1, gt3.float().to(device)) + criterion(pred_image1, gt4.float().to(device))
            loss6 = criterion(pred_image2, gt3.float().to(device)) + criterion(pred_image2, gt4.float().to(device))
            loss7 = criterion(pred_image3, gt3.float().to(device)) + criterion(pred_image3, gt4.float().to(device))
            loss8 = criterion(pred_image4, gt3.float().to(device)) + criterion(pred_image4, gt4.float().to(device))

            loss9 = criterion(pred_image1, pred_image2) + criterion(pred_image1, pred_image3)+criterion(pred_image1, pred_image4)+criterion(pred_image2, pred_image3)+criterion(pred_image2, pred_image4)+criterion(pred_image3, pred_image4)
            optimizer_denoise.zero_grad()
            loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
            loss.backward()
            optimizer_denoise.step()

            folder_path = "Step2_Train_Isotropic_denoise_branch"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            print("[Epoch %d] [Batch %d/%d] [loss1: %f]" % (epoch, batch_idx, len(ZS3D_dataset), loss.item()*100 ))
            if batch_idx % 200 == 1:
                io.imsave(r'Step2_Train_Isotropic_denoise_branch\Train_denoise' + '_pred.tif',
                           pred_image2[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_denoise_branch\Train_denoise' + '_in_clean.tif',
                           up_clean[:, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_denoise_branch\Train_denoise' + '_in1.tif',
                           up1[:, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_denoise_branch\Train_denoise' + '_gt1.tif',
                           gt1[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_denoise_branch\Train_denoise' + '_in2.tif',
                           up2[:, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Isotropic_denoise_branch\Train_denoise' + '_gt2.tif',
                           gt2[0, :, :, :].detach().cpu().numpy())
        torch.save(model, r"Step2_Train_Isotropic_denoise_branch\Model_isotropic_denoise_epoch_" + str(epoch + 1) + ".pth")

    end = time.time()
    print(f"Comsuming time：{end - start:.6f}s")