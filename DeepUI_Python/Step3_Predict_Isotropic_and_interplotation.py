from skimage import io
import numpy as np
import torch
from tifffile import tifffile
from scipy import ndimage

####### User parameters #######
if_denoise = 0 # If denoise model is used, please define this as 1
input_path = r'E:\Prj70_DeepIso_submit\Data\raw_data\Mouse_actin_down\1_down_6.tif'
output_path = r'E:\Prj70_DeepIso_submit\Data\raw_data\Mouse_actin_down\1_down_6-pre.tif'
model_interplotation = r'E:\Prj70_DeepIso_submit\Step2_Train_Interplotation_branch\Model_interplotation_epoch_1_mouse_actin_down.pth'
model_isotropic = r'E:\Prj70_DeepIso_submit\Step2_Train_Isotropic_branch\Model_isotropic_epoch_1_mouse_actin_down.pth'
###############################

def save_3d_tiff(data, path):
    data = np.clip(data, 0, 1)
    data_16bit = (data * 65535).astype(np.uint16)
    with tifffile.TiffWriter(path) as tif:
        for i in range(data_16bit.shape[0]):
            tif.save(data_16bit[i], photometric='minisblack')
    print(f"Sucessfully save the tif file to: {path}")

IN_total = io.imread(input_path)
max_prc = 100
min_prc = 0
IN_total = (IN_total - np.percentile(IN_total, min_prc)) / (np.percentile(IN_total, max_prc) - np.percentile(IN_total, min_prc) + 1e-7)
IN_total[IN_total > 1] = 1
IN_total[IN_total < 0] = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = torch.load(model_interplotation, weights_only=False)
model1 = model1.to(device)
model2 = torch.load(model_isotropic, weights_only=False)
model2 = model2.to(device)

[Nz, Nx, Ny] = IN_total.shape
X_tmp = np.zeros([Nz, Nx, Ny])
Y_tmp = np.zeros([Nz, Nx, Ny])

for jy in range(Ny-1):
    print(f"Processing {jy} (Total Nx = {Nx})")
    tmp = np.zeros([Nz, Nx])
    IN = IN_total[:, :, jy].squeeze()
    tmp[0:Nz,:] = IN
    IN = np.expand_dims(tmp, 0)
    IN = np.expand_dims(IN, 0)
    IN = torch.from_numpy(IN)
    pred_image1 = model1(IN.float().to(device))
    pred_image2 = model2(pred_image1)
    Y_tmp[:, :, jy] = pred_image2[0,0,:,:].detach().cpu().numpy()
Input_total = Y_tmp

for jx in range(Nx-1):
    print(f"Processing jx = {jx} (Total Nx = {Nx})")
    print(jx)
    tmp = np.zeros([Nz, Ny])
    IN = IN_total[:, jx, :].squeeze()
    tmp[0:Nz,:] = IN
    IN = np.expand_dims(tmp, 0)
    IN = np.expand_dims(IN, 0)
    IN = torch.from_numpy(IN)
    pred_image1 = model1(IN.float().to(device))
    pred_image2 = model2(pred_image1)
    X_tmp[:, jx, :] = pred_image2[0,0,:,:].detach().cpu().numpy()

Input_total = X_tmp + Y_tmp
if if_denoise == 1:
    sigma_xy = 1
    blurred_volume = ndimage.gaussian_filter(Input_total, sigma=(0, sigma_xy, sigma_xy)) # 更安全的做法是指定所有轴的标准差
    save_3d_tiff(blurred_volume/2, output_path)
else:
    save_3d_tiff(Input_total/2, output_path)





