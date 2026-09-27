import torch
import torch.optim as optim
import numpy as np
from skimage import io
import torch.nn.functional as F
import os
from Fun.Add_Noise import batch_add_noise1
from Fun.Rescale_OTF import process_otf_tensor
from Fun.load_data import ReconsDataset2D_single
from Fun.Model_RCAN import RCAN
import math

####### User parameters #######
corrected_sOTF = r'E:\Prj70_DeepIso_submit\sOTF\WF_sOTF.tif'
XOY_path = r'E:\Prj70_DeepIso_submit\Data\dataset\XOY'
XOZ_path = r'E:\Prj70_DeepIso_submit\Data\dataset\XOZ'
step = 6
###############################


def get_sequence(x):
     original_sequence = list ( range (1, 123, x ))
     if not original_sequence :
         return [], None, None
     first_item = original_sequence [0]
     last_item = original_sequence [-1]
     distance = last_item - first_item +1
     y = math.floor ((128- distance )/2)
     result = [num + y for num in original_sequence]
     return result [0], result [-1]


if __name__ == '__main__':

    # parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    minipatch = 128

    # dataset
    ZS3D_dataset_initial = ReconsDataset2D_single(train_gt_path = XOY_path,
                                                  train_ref_path = XOZ_path,
                                                  img_type='tif')
    ZS3D_dataset = torch.utils.data.DataLoader(ZS3D_dataset_initial, batch_size=batch_size, shuffle=True,
                                                   pin_memory=False, drop_last=True)

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
            max_prc = 100
            min_prc = 0
            ref = (ref - np.percentile(ref, min_prc)) / (np.percentile(ref, max_prc) - np.percentile(ref, min_prc) + 1e-7)
            gt = (gt - np.percentile(gt, min_prc)) / (np.percentile(gt, max_prc) - np.percentile(gt, min_prc) + 1e-7)
            ref[ref > 1] = 1
            gt[gt < 0] = 0
            gt[gt > 1] = 1
            gt[gt < 0] = 0
            OTF_rescale = process_otf_tensor(OTF, minipatch=minipatch, variance=0.01)
            up_clean = torch.abs(torch.fft.ifft2(torch.fft.fft2(gt) * torch.fft.fftshift(OTF_rescale)))
            up_noise = batch_add_noise1(up_clean,torch.tensor(ref))

            x, y = get_sequence(step)
            up_tmp = up_noise[:,:,x:y:step,x:y]
            up_tmp_final = F.interpolate(
                up_tmp,
                size=(y-x, y-x),  # 目标尺寸（高度保持128，宽度扩展到128）
                mode='bilinear',  # 双线性插值（适用于图像）
                align_corners=True
            )

            up = torch.zeros(up_noise.size())
            up[:,:,x:y,x:y] = up_tmp_final

            model.train()
            pred_image2 = model(up.float().to(device))

            criterion = torch.nn.L1Loss(reduction='mean')
            loss = criterion(pred_image2, up_clean.float().to(device))

            folder_path = "Step2_Train_Interplotation_branch"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            optimizer_denoise.zero_grad()
            loss.backward()
            optimizer_denoise.step()

            print("[Epoch %d] [Batch %d/%d] [loss1: %f]" % (epoch, batch_idx, len(ZS3D_dataset), loss.item()*100 ))
            # print("[%f]" % (loss.item()*100))
            if batch_idx % 200 == 1:
                io.imsave(r'Step2_Train_Interplotation_branch\Train' + '_pred.tif',
                           pred_image2[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Interplotation_branch\Train' + '_up.tif',
                           up[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Interplotation_branch\Train' + '_up_clean.tif',
                           up_clean[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step2_Train_Interplotation_branch\Train' + '_up_tmp.tif',
                          up_tmp.detach().cpu().numpy())

        torch.save(model, r"Step2_Train_Interplotation_branch\Model_interplotation_epoch_" + str(epoch + 1) + ".pth")