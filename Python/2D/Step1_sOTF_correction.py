import torch
from torch.optim import Adam
import torch.nn.functional as F
from skimage import io
from Fun.Model_Unet import Unet1_step
from Fun.SSIM import SSIM
from Fun.load_data import create_paired_loaders
import os

####### User parameters #######
contrast = 0.3
sOTF_name = r'sOTF\sOTF_2DSIM_simulation.tif'
XOY_path = r'Data\patch'
corrected_sOTF = r'sOTF\sOTF_2DSIM_Tubulin.tif'
###############################


def compute_isotropy_loss(spectrum, GT, n_angles=72):
    """
    spectrum: [B, H, W] 频谱幅度（已log+归一化）
    返回：各方向MTF的差异
    """
    B, H, W = spectrum.shape
    device = spectrum.device

    # 频率网格
    kx = torch.linspace(-0.5, 0.5, W, device=device)
    ky = torch.linspace(-0.5, 0.5, H, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing='xy')

    # 中心
    cy, cx = H // 2, W // 2

    # 提取径向MTF的函数
    def get_mtf(angle_deg):
        angle_rad = torch.deg2rad(torch.tensor(angle_deg, device=device))
        # 径向距离
        max_r = min(cy, cx)
        r = torch.arange(0, max_r, device=device).float()

        # 该角度上的坐标
        xq = cx + r * torch.cos(angle_rad)
        yq = cy + r * torch.sin(angle_rad)

        # 限制在图像范围内
        valid = (xq >= 0) & (xq < W) & (yq >= 0) & (yq < H)
        xq = xq[valid].long()
        yq = yq[valid].long()
        r_valid = r[valid]

        # 提取MTF
        mtfs = []
        for b in range(B):
            mtf = spectrum[b, yq, xq]  # 沿该角度的值
            # 插值到固定长度
            target_len = max_r
            if len(mtf) > 1:
                mtf_interp = F.interpolate(
                    mtf.unsqueeze(0).unsqueeze(0),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            else:
                mtf_interp = torch.zeros(target_len, device=device)
            mtfs.append(mtf_interp)

        return torch.stack(mtfs)  # [B, target_len]

    # 提取径向MTF的函数
    def get_mtf_GT(angle_deg):
        angle_rad = torch.deg2rad(torch.tensor(angle_deg, device=device))
        # 径向距离
        max_r = min(cy, cx)
        r = torch.arange(0, max_r, device=device).float()

        # 该角度上的坐标
        xq = cx + r * torch.cos(angle_rad)
        yq = cy + r * torch.sin(angle_rad)

        # 限制在图像范围内
        valid = (xq >= 0) & (xq < W) & (yq >= 0) & (yq < H)
        xq = xq[valid].long()
        yq = yq[valid].long()
        r_valid = r[valid]

        # 提取MTF
        mtfs = []
        for b in range(B):
            mtf = GT[b, yq, xq]  # 沿该角度的值
            # 插值到固定长度
            target_len = max_r
            if len(mtf) > 1:
                mtf_interp = F.interpolate(
                    mtf.unsqueeze(0).unsqueeze(0),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            else:
                mtf_interp = torch.zeros(target_len, device=device)
            mtfs.append(mtf_interp)

        return torch.stack(mtfs)  # [B, target_len]

    # 采样多个角度
    angles = torch.linspace(0, 360, n_angles, device=device)

    # 以0°为基准
    mtf_ref_gt = get_mtf(0)

    loss = 0
    for ang in angles[1:]:
        mtf = get_mtf(ang.item())
        # 只比较有效长度
        min_len = min(mtf_ref_gt.shape[1], mtf.shape[1])
        loss += F.mse_loss(mtf[:, :min_len], mtf_ref_gt[:, :min_len])

    return loss / (n_angles - 1)



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    minipatch=256

    loader = create_paired_loaders(
        path_A = XOY_path,
        batch_size = batch_size
    )

    model = Unet1_step(input_shape=1, upsample_flag=1).to(device)
    opt_g = Adam(model.parameters(), lr=5e-5)

    OTF = io.imread(sOTF_name)
    OTF = OTF/OTF.max()
    OTF = torch.tensor(OTF)
    OTF = OTF.unsqueeze(dim=0)  # 在第0轴（batch）和第1轴（通道）扩展
    OTF = OTF.unsqueeze(dim=0)
    OTF = OTF.repeat(batch_size, 1, 1, 1)

    for batch in loader:
        images_A = batch  # 直接解包
        random_index = 0  # 可以改为随机数，如 torch.randint(0, images_A.size(0), (1,)).item()
        real_A = batch['A'].float().to(device)  # 从批量A中取一张
        zero_rows = torch.all(real_A == 0, dim=(1, 3)).sum(dim=1)
        line = 0 if torch.sum(zero_rows)/16 > 2 else 0
        break

    for epoch in range(2):
        # 验证数据配对
        kkk = 0
        for batch_idx, items in enumerate(loader):

            real_A = items['A'].float().to(device)
            OTF_ = model(OTF.float().to(device))
            OTF_ = torch.abs(OTF_ / OTF_.max())
            # OTF_ = OTF.float().to(device)

            fake_A = torch.abs(torch.fft.ifft2(torch.fft.fft2(real_A) * torch.fft.fftshift(OTF_)))

            window = torch.hamming_window(minipatch).unsqueeze(0) * torch.hamming_window(minipatch).unsqueeze(1)
            window = window.unsqueeze(0).unsqueeze(0).float().to(device)
            fake_A_window = fake_A * window

            tmp2 = 1 + torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A_window-fake_A_window.mean()))) + 1e-3)
            summed_tmp20 = tmp2.sum(dim=0).unsqueeze(0)

            summed_tmp2 = summed_tmp20-summed_tmp20.min()
            summed_tmp2 = summed_tmp2 /summed_tmp2.max()

            if line > 0:
                summed_tmp2[:, :, :, 64-line:64+line] = 0

            def gaussian_kernel(size=3, sigma=1.0):
                kernel = torch.exp(-torch.arange(-(size // 2), size // 2 + 1) ** 2 / (2 * sigma ** 2))
                kernel = kernel / kernel.sum()
                return kernel.outer(kernel).unsqueeze(0).unsqueeze(0)

            kernel = gaussian_kernel().to(device)
            summed_tmp2 = F.conv2d(summed_tmp2, kernel, padding='same')
            summed_tmp2 = F.conv2d(summed_tmp2, kernel, padding='same')
            summed_tmp2 = torch.clamp(summed_tmp2, min=contrast)
            # summed_tmp2[:,:,126:129,126:129]=0
            loss_iso1 = compute_isotropy_loss(summed_tmp2.squeeze(1), real_A.sum(dim=0))
            summed_tmp2_clamp = torch.clamp(summed_tmp2, min=0.5)
            loss_iso2 = compute_isotropy_loss(summed_tmp2_clamp.squeeze(1), real_A.sum(dim=0))
            loss_prior = F.l1_loss(OTF_, OTF.float().to(device))

            if batch_idx % 10 == 1:
                io.imsave(r'Step1_Degrade\OTF_.tif', OTF_[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\OTF.tif', OTF[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\summed_tmp2.tif', summed_tmp2[0, :, :, :].detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\pred_LR.tif',fake_A[0, :, :, :].detach().cpu().numpy())
            loss_iso = loss_iso1 + loss_iso2
            loss_g = 20*loss_iso + loss_prior #10：1
            print(f"Epoch {epoch}: Iso_loss={loss_iso.item():.3f}, image_loss1={loss_prior.item():.3f} kkk={kkk:1f}")

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_g.step()


            kkk = kkk + 1

            folder_path = "Step1_Degrade"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            if kkk % 50 == 1:
                io.imsave(r'Step1_Degrade\disc_real.tif',
                summed_tmp2.detach().cpu().numpy())
                io.imsave(r'Step1_Degrade\real_A.tif',
                real_A[0, :, :, :].detach().cpu().numpy())
                io.imsave(corrected_sOTF,
                OTF_[0, :, :, :].detach().cpu().numpy())


