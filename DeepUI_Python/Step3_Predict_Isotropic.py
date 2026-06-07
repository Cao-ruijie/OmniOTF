from skimage import io
import numpy as np
import torch
from tifffile import tifffile
from scipy import ndimage

####### User parameters #######
if_denoise = 1 # If denoise model is not used, please define this as 0
input_path = r'E:\Prj70_DeepIso_submit\Data\raw_data\live_mito\1.tif'
output_path = r'E:\Prj70_DeepIso_submit\Data\output\liveMito.tif'
model1 = torch.load(r'E:\Prj70_DeepIso_submit\Step2_Train_Isotropic_denoise_branch\Model_isotropic_denoise_epoch_1_liveMito.pth', weights_only=False)
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
model1 = model1.to(device)

[Nz, Nx, Ny] = IN_total.shape
X_tmp = np.zeros([Nz, Nx, Ny])
Y_tmp = np.zeros([Nz, Nx, Ny])

for jx in range(Nx-1):
    tmp = np.zeros([Nz+20, Ny])
    IN = IN_total[:, jx, :].squeeze()
    tmp[10:Nz+10,:] = IN
    IN = np.expand_dims(tmp, 0)
    IN = np.expand_dims(IN, 0)
    IN = torch.from_numpy(IN)
    pred_image2 = model1(IN.float().to(device))
    X_tmp[:, jx, :] = pred_image2[0,0,10:Nz+10,:].detach().cpu().numpy()

for jy in range(Ny-1):
    tmp = np.zeros([Nz+20, Nx])
    IN = IN_total[:, :, jy].squeeze()
    tmp[10:Nz+10,:] = IN
    IN = np.expand_dims(tmp, 0)
    IN = np.expand_dims(IN, 0)
    IN = torch.from_numpy(IN)
    pred_image2 = model1(IN.float().to(device))
    Y_tmp[:, :, jy] = pred_image2[0,0,10:Nz+10,:].detach().cpu().numpy()

Input_total = X_tmp + Y_tmp
if if_denoise == 1:
    sigma_xy = 1
    blurred_volume = ndimage.gaussian_filter(Input_total, sigma=(0, sigma_xy, sigma_xy)) # 更安全的做法是指定所有轴的标准差
    save_3d_tiff(blurred_volume/2, output_path)
else:
    save_3d_tiff(Input_total/2, output_path)



