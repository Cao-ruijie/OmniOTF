from skimage import io
import numpy as np
import torch
from tifffile import tifffile
from scipy import ndimage
import torchvision.transforms.functional as TF
from Fun.Rescale_OTF import process_otf_tensor
import torch.nn.functional as F

####### User parameters #######
if_denoise = 1    # If denoise model is not used, please define this as 0
input_path = r'Data\Tubulin\Single.tif'
output_path = r'Step3_predict\Single-degrade.tif'
model1 = torch.load(r'Step2_Train_Isotropic_branch\Model_isotropic_epoch256_sOTF_2DSIM_tubulin_clean_new2.pth', weights_only=False)
corrected_sOTF = r'Step1_Degrade\OTF_.tif'
N = 1024
###############################

def center_pad_to_target(img_tensor: torch.Tensor, target_size: int = 1024) -> torch.Tensor:
    # 获取原图高、宽（最后两维）
    h, w = img_tensor.shape[-2], img_tensor.shape[-1]

    # 计算高度方向上下填充量
    pad_h_total = target_size - h
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top

    # 计算宽度方向左右填充量
    pad_w_total = target_size - w
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    # F.pad 填充顺序：(左, 右, 上, 下)
    padded_tensor = F.pad(
        img_tensor,
        pad=(pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0  # 填充0 = 纯黑背景
    )
    return padded_tensor

# ========== 新增：居中裁剪函数（pad的逆操作）==========
def center_crop_from_target(img_tensor: torch.Tensor, original_h: int, original_w: int) -> torch.Tensor:
    h_full, w_full = img_tensor.shape[-2], img_tensor.shape[-1]
    # 裁剪起始点 = 之前的填充偏移量，完全对齐
    start_y = (h_full - original_h) // 2
    start_x = (w_full - original_w) // 2
    # 保留批次、通道维度，只裁剪空间维度
    return img_tensor[..., start_y:start_y+original_h, start_x:start_x+original_w]


def save_3d_tiff(data, path):
    data = np.clip(data, 0, 1)
    data_16bit = (data * 65535).astype(np.uint16)
    with tifffile.TiffWriter(path) as tif:
        if data_16bit.ndim == 3:
            for i in range(data_16bit.shape[0]):
                tif.save(data_16bit[i], photometric='minisblack')
        else:
            tif.save(data_16bit, photometric='minisblack')
    print(f"Sucessfully save the tif file to: {path}")


def save_3d_tiff(data, path):
    data = np.clip(data, 0, 1)
    data_16bit = (data * 65535).astype(np.uint16)
    with tifffile.TiffWriter(path) as tif:
        if data_16bit.ndim == 3:
            for i in range(data_16bit.shape[0]):
                tif.save(data_16bit[i], photometric='minisblack')
        else:
            tif.save(data_16bit, photometric='minisblack')
    print(f"Sucessfully save the tif file to: {path}")

IN_total = io.imread(input_path)
original_h, original_w = IN_total.shape[-2], IN_total.shape[-1]  # 记录原始尺寸
max_prc = 100
min_prc = 0
IN_total = (IN_total - np.percentile(IN_total, min_prc)) / (np.percentile(IN_total, max_prc) - np.percentile(IN_total, min_prc) + 1e-7)
IN_total[IN_total > 1] = 1
IN_total[IN_total < 0] = 0
IN_total = np.expand_dims(IN_total, 0)
IN_total = np.expand_dims(IN_total, 0)
IN_total = torch.from_numpy(IN_total)
IN_total = center_pad_to_target(IN_total, target_size=N)

OTF = io.imread(corrected_sOTF)
OTF = OTF - OTF.min()
OTF = OTF/OTF.max()
OTF = torch.tensor(OTF)
OTF = OTF.unsqueeze(dim=0)   # 在第0轴（batch）和第1轴（通道）扩展
OTF_resized = torch.nn.functional.interpolate(OTF, size=(N, N), mode='bilinear', align_corners=False)


# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_OTF_resized.tif',
#           OTF_resized[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_OTF_rot60.tif',
#           OTF_rot60[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_OTF_rot.tif',
#           OTF_rot120[0, :, :, :].detach().cpu().numpy())



F_blur = torch.fft.fft2(IN_total) * torch.fft.fftshift(OTF_resized)
up_clean = torch.abs(torch.fft.ifft2(F_blur))
up_clean_cropped = center_crop_from_target(up_clean, original_h, original_w)
save_3d_tiff(up_clean_cropped.detach().cpu().numpy(), output_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)


# 总参数量
total_params = sum(p.numel() for p in model1.parameters())
# 可训练参数量
trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)

print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")


pred_image0 = model1(up_clean.float().to(device))

IN_total45 = TF.rotate(up_clean, angle=45, fill=0)
pred_image45 = model1(IN_total45.float().to(device))
pred_image45_back = TF.rotate(pred_image45, angle=-45, fill=0)

IN_total90 = TF.rotate(up_clean, angle=90, fill=0)
pred_image90 = model1(IN_total90.float().to(device))
pred_image90_back = TF.rotate(pred_image90, angle=-90, fill=0)

IN_total135 = TF.rotate(up_clean, angle=-45, fill=0)
pred_image135 = model1(IN_total135.float().to(device))
pred_image135_back = TF.rotate(pred_image135, angle=45, fill=0)


# pred_image0[pred_image0<0]=0
# pred_image45_back[pred_image45_back<0]=0
# pred_image135_back[pred_image135_back<0]=0
# pred_image90_back[pred_image90_back<0]=0


# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_IN_total135-1.tif',
#           IN_total135[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_IN_total90-1.tif',
#           IN_total90[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_IN_total0-1.tif',
#           up_clean[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_pred_image0-1.tif',
#           pred_image0[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_pred_image135-1.tif',
#           pred_image135[0, :, :, :].detach().cpu().numpy())
# io.imsave(r'Step2_Train_Isotropic_branch\Train' + '_pred_image90-1.tif',
#           pred_image90[0, :, :, :].detach().cpu().numpy())

def get_fft_shift(img):
    """输入 [B,C,H,W]，返回中心化复频谱 [B,C,H,W]"""
    fft_complex = torch.fft.fft2(img)
    fft_shift = torch.fft.fftshift(fft_complex)
    return fft_shift

# 得到三张中心化频谱

fft0 = get_fft_shift(pred_image0)
fft45 = get_fft_shift(pred_image45_back)
fft135 = get_fft_shift(pred_image135_back)
fft90 = get_fft_shift(pred_image90_back)

# 沿新增维度堆叠 [4, B, C, H, W]
fft_stack = torch.stack([fft0, fft45, fft135, fft90], dim=0)
# 计算每张频谱的幅值
amp_stack = torch.abs(fft_stack)
# 每个像素找到幅值最大的索引 argmax(0)
max_idx = torch.argmax(amp_stack, dim=0, keepdim=True)  # [1,B,C,H,W]


# 根据索引取出对应复数（幅度相位一体）
# expand匹配stack维度用于gather
max_idx_exp = max_idx.expand(-1, *fft_stack.shape[1:])
fft_fuse = torch.gather(fft_stack, dim=0, index=max_idx_exp).squeeze(0)

# 逆FFT还原图像
fft_fuse_ishift = torch.fft.ifftshift(fft_fuse)
img_fuse_complex = torch.fft.ifft2(fft_fuse_ishift)
img_fuse = torch.real(img_fuse_complex)  # 显微图像取实部更合理


img_fuse_cropped = center_crop_from_target(img_fuse, original_h, original_w)

# 高斯平滑去噪，kernel_size建议3/5，sigma越大模糊越强
img_fuse_filtered = TF.gaussian_blur(img_fuse_cropped, kernel_size=3, sigma=2.0)
# 所有小于0的像素截断为0
img_fuse_filtered = torch.clamp(img_fuse_filtered, min=0.0)

# 保存
io.imsave(r'Step3_predict\Train_img_fuse-tubulin.tif',
          img_fuse_filtered.detach().cpu().numpy()[0,0])



