import torch
import pywt
import numpy as np
from skimage import io
from torchvision.transforms.functional import gaussian_blur

def batch_add_noise1(final_image: torch.Tensor,
                    image0: torch.Tensor,
                    use_batch_average: bool = True) -> torch.Tensor:

    blurred_image0 = gaussian_blur(image0, kernel_size=5, sigma=[1.0, 1.0])
    N2 = image0 - blurred_image0
    noisy = final_image + 1*N2 + 0.02 * torch.randn_like(final_image)
    noisy = noisy/noisy.max()
    noisy[noisy < 0] = 0
    return noisy

def batch_add_noise2(final_image: torch.Tensor,
                    image0: torch.Tensor,
                    use_batch_average: bool = True) -> torch.Tensor:

    blurred_image0 = gaussian_blur(image0, kernel_size=5, sigma=[3.0, 3.0])
    N2 = image0 - blurred_image0
    N2 = torch.roll(N2, shifts=-1, dims=0)
    # noisy = final_image + 0.5*N2 + 0.01 * torch.randn_like(final_image)
    noisy = final_image + 0.5*N2 + 0.01 * torch.randn_like(final_image)
    noisy = noisy/noisy.max()
    noisy[noisy < 0] = 0
    # io.imsave(r'E:\Prj65_SN2NItIs\Step3_Train_DeepIs\Train1' + '_image0.tif',
    #           image0[:, :, :, :].detach().cpu().numpy())
    # io.imsave(r'E:\Prj65_SN2NItIs\Step3_Train_DeepIs\Train1' + '_noisy.tif',
    #           noisy[:, :, :, :].detach().cpu().numpy())
    return noisy