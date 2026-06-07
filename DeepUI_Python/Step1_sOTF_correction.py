import torch
from torch.optim import Adam
import torch.nn.functional as F
from skimage import io
from Fun.Model_Unet import Unet1_step
from Fun.SSIM import SSIM
from Fun.load_data import create_paired_loaders
import os

####### User parameters #######
contrast = 0.30
sOTF_name = r'E:\Prj70_DeepIso_submit\sOTF\ISM_sOTF_simulation.tif'
XOY_path = r'E:\Prj70_DeepIso_submit\Data\dataset\XOY'
XOZ_path = r'E:\Prj70_DeepIso_submit\Data\dataset\XOZ'
corrected_sOTF = r'E:\Prj70_DeepIso_submit\sOTF\ISM_sOTF.tif'
###############################

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    loader = create_paired_loaders(
        path_A = XOY_path,
        path_B = XOZ_path,
        batch_size = batch_size
    )

    model = Unet1_step(input_shape=1, upsample_flag=1).to(device)
    opt_g = Adam(model.parameters(), lr=1e-4)

    OTF = io.imread(sOTF_name)
    OTF = OTF/OTF.max()
    OTF = torch.tensor(OTF)
    OTF = OTF.unsqueeze(dim=0)  # 在第0轴（batch）和第1轴（通道）扩展
    OTF = OTF.unsqueeze(dim=0)
    OTF = OTF.repeat(batch_size, 1, 1, 1)

    for batch in loader:
        images_A, images_B = batch  # 直接解包
        random_index = 0  # 可以改为随机数，如 torch.randint(0, images_A.size(0), (1,)).item()
        real_A = batch['A'].float().to(device)  # 从批量A中取一张
        real_B = batch['B'].float().to(device) # 从批量B中取一张
        zero_rows = torch.all(real_B == 0, dim=(1, 3)).sum(dim=1)
        line = 1 if torch.sum(zero_rows)/16 > 2 else 0
        break

    for epoch in range(2):
        # 验证数据配对
        kkk = 0
        for batch in loader:

            real_A = batch['A'].float().to(device)
            real_B = batch['B'].float().to(device)

            OTF_ = model(OTF.float().to(device))

            OTF_ = torch.abs(OTF_ / OTF_.max())
            right_bottom = OTF_[..., 64:, 64:]
            left_bottom = torch.flip(right_bottom, dims=[-1])  # 水平翻转
            right_top = torch.flip(right_bottom, dims=[-2])  # 垂直翻转
            left_top = torch.flip(left_bottom, dims=[-2])  # 垂直翻转左上区域
            top = torch.cat([left_top, right_top], dim=-1)  # 上半部分
            bottom = torch.cat([left_bottom, right_bottom], dim=-1)  # 下半部分
            OTF_ = torch.cat([top, bottom], dim=-2)

            fake_A = torch.abs(torch.fft.ifft2(torch.fft.fft2(real_A) * torch.fft.fftshift(OTF_)))

            window = torch.hamming_window(128).unsqueeze(0) * torch.hamming_window(128).unsqueeze(1)
            window = window.unsqueeze(0).unsqueeze(0).float().to(device)
            fake_A_window = fake_A * window
            real_B_window = real_B * window

            tmp1 = 1 + torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(real_B_window-real_B_window.mean()))) + 1e-3)
            tmp2 = 1 + torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A_window-fake_A_window.mean()))) + 1e-3)
            summed_tmp10 = tmp1.sum(dim=0).unsqueeze(0)  # 形状变为 (1,128,128)
            summed_tmp20 = tmp2.sum(dim=0).unsqueeze(0)

            summed_tmp1 = summed_tmp10-summed_tmp10.min()
            summed_tmp2 = summed_tmp20-summed_tmp20.min()
            summed_tmp1 = summed_tmp1 /summed_tmp1.max()
            summed_tmp2 = summed_tmp2 /summed_tmp2.max()

            if line > 0:
                summed_tmp1[:, :, :, 64-line:64+line] = 0
                summed_tmp2[:, :, :, 64-line:64+line] = 0

            def gaussian_kernel(size=3, sigma=1.0):
                kernel = torch.exp(-torch.arange(-(size // 2), size // 2 + 1) ** 2 / (2 * sigma ** 2))
                kernel = kernel / kernel.sum()
                return kernel.outer(kernel).unsqueeze(0).unsqueeze(0)

            kernel = gaussian_kernel().to(device)
            summed_tmp1 = F.conv2d(summed_tmp1, kernel, padding='same')
            summed_tmp2 = F.conv2d(summed_tmp2, kernel, padding='same')
            summed_tmp1 = F.conv2d(summed_tmp1, kernel, padding='same')
            summed_tmp2 = F.conv2d(summed_tmp2, kernel, padding='same')

            summed_tmp1 = torch.clamp(summed_tmp1, min=contrast)
            summed_tmp2 = torch.clamp(summed_tmp2, min=contrast)

            # 生成器训练
            L1_loss = F.l1_loss(OTF_, OTF.float().to(device))
            ssim_loss = SSIM(window_size=4)
            SSIM_loss = 1-ssim_loss(summed_tmp1, summed_tmp2)
            #
            # image_loss2 = 1 - ssim_loss(summed_tmp1, summed_tmp2)
            loss_g = 2000*SSIM_loss + L1_loss #默认50：1
            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_g.step()

            print(f"Epoch {epoch}: L1_loss={L1_loss.item():.3f}, image_loss1={SSIM_loss.item():.3f} kkk={kkk:1f}")
            kkk = kkk + 1

            folder_path = "Step1_Degrade"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            if kkk % 15 == 0:
                io.imsave(r'Step1_Degrade\disc_real.tif',
                summed_tmp1.detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\disc_fake.tif',
                summed_tmp2.detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\disc_real1.tif',
                summed_tmp10.detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\disc_fake1.tif',
                summed_tmp20.detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\real_A.tif',
                real_A[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\real_B.tif',
                real_B[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\pred_LR.tif',
                fake_A[0, :, :, :].detach().cpu().numpy())
                io.imsave(corrected_sOTF,
                OTF_[0, :, :, :].detach().cpu().numpy())


