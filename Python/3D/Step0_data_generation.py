import torch
import numpy as np
import os
import random
from pathlib import Path
import tifffile
import torchvision.transforms.functional as TF

####### User parameters #######
thres = 100
raw_data_dir = r'E:\Prj70_DeepIso_submit\Data\raw_data\live_mito'
data_dir0 = r'E:\Prj70_DeepIso_submit\Data\dataset'
###############################

def imstackread(filepath):
    try:
        # Read data
        image_stack = tifffile.imread(filepath)
        # The data should be single-channel
        if len(image_stack.shape) == 3:
            pass
        elif len(image_stack.shape) == 4:
            image_stack = image_stack[:, :, :, 0]
        return image_stack.astype(np.float32)
    except Exception as e:
        print(f"Error reading TIFF file {filepath}: {e}")
        return None


def myedgetaper(image):
    if isinstance(image, torch.Tensor):
        image_np = image.numpy()
    else:
        image_np = image

    if len(image_np.shape) == 2:
        h, w = image_np.shape
        taper_width = min(h, w) // 10
        mask = np.ones((h, w), dtype=np.float32)

        for i in range(taper_width):
            factor = 0.5 - 0.5 * np.cos(np.pi * i / taper_width)
            mask[i, :] *= factor
            mask[h - 1 - i, :] *= factor

        for j in range(taper_width):
            factor = 0.5 - 0.5 * np.cos(np.pi * j / taper_width)
            mask[:, j] *= factor
            mask[:, w - 1 - j] *= factor

        result = image_np * mask
        return torch.from_numpy(result) if isinstance(image, torch.Tensor) else result
    else:
        tapered_slices = []
        for i in range(image_np.shape[0]):
            tapered_slices.append(myedgetaper(image_np[i]))
        return np.stack(tapered_slices)


def place_center3(image, minipatch):
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image)
    else:
        image_tensor = image

    if len(image_tensor.shape) == 3:
        [N0, Nx, Ny] = image_tensor.shape
        output = torch.zeros((1, minipatch, minipatch), dtype=image_tensor.dtype)

        center = minipatch // 2
        start_x = center - Nx // 2
        start_y = center - Ny // 2

        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(minipatch, start_x + Nx)
        end_y = min(minipatch, start_y + Ny)

        src_start_x = max(0, -start_x)
        src_start_y = max(0, -start_y)
        src_end_x = min(Nx, minipatch - start_x)
        src_end_y = min(Ny, minipatch - start_y)

        output[0,start_x:end_x, start_y:end_y] = image_tensor[0,src_start_x:src_end_x, src_start_y:src_end_y]

        return output

    elif len(image_tensor.shape) == 3:
        Nz, Nx, Ny = image_tensor.shape
        output = torch.zeros((minipatch, minipatch, minipatch), dtype=image_tensor.dtype)

        center = minipatch // 2
        start_z = center - Nz // 2
        start_x = center - Nx // 2
        start_y = center - Ny // 2

        start_z = max(0, start_z)
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_z = min(minipatch, start_z + Nz)
        end_x = min(minipatch, start_x + Nx)
        end_y = min(minipatch, start_y + Ny)

        output[start_z:end_z, start_x:end_x, start_y:end_y] = image_tensor
        return output


def save_tiff_image(image, filepath):
    if isinstance(image, torch.Tensor):
        image_np = image.numpy().astype(np.uint8)
    else:
        image_np = image.astype(np.uint8)

    # 确保是2D数组
    if len(image_np.shape) == 3 and image_np.shape[0] == 1:
        image_np = image_np.squeeze()

    tifffile.imwrite(filepath, image_np)


def main():
    minipatch = 128
    num = 200
    batch = 8

    data_dir = data_dir0
    dir1 = data_dir + '//XOZ//'
    dir2 = data_dir + '//XOY//'

    Path(dir1).mkdir(parents=True, exist_ok=True)
    Path(dir2).mkdir(parents=True, exist_ok=True)

    kkk1, kkk2, kkk3 = 1, 1, 1

    file_num =0
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_num = file_num+1

    if file_num == 0:
        print(f"There is no images in the path")

    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                full_path = os.path.join(root, file)
                image_data = imstackread(full_path)
                if image_data is None:
                    print(f"Failed to read image: {full_path}")
                    continue
                image_tensor = torch.from_numpy(image_data)
                image_normalized = 255 * image_tensor / torch.max(image_tensor)
                if len(image_normalized.shape) == 3:
                    Nz, Nx, Ny = image_normalized.shape
                else:
                    print("Unexpected image dimensions")
                    continue
                print(f"3D Image shape: {image_normalized.shape} (Depth, Height, Width)")

                kkk3_1 = 1
                while kkk3_1 < batch * num/file_num:
                    x0 = random.randint(0, Nx - minipatch - 1)
                    y0 = random.randint(0, Ny - 1)

                    if Nz > minipatch:
                        z0 = random.randint(0, Nz - minipatch - 1)
                        tmp00 = image_normalized[z0:z0 + minipatch, x0:x0 + minipatch, y0]
                    else:
                        z0 = 0
                        Nrepeat = minipatch // Nz
                        startx_scale = minipatch - Nrepeat * Nz
                        startx = int(torch.floor(0 + startx_scale * torch.rand(1)).item())
                        tmp00 = torch.zeros((minipatch, minipatch))

                        for repeat in range(Nrepeat):
                            x0 = random.randint(0, Nx - minipatch - 1)
                            y0 = random.randint(0, Ny - 1)
                            tmp00_ = image_normalized[z0:z0 + Nz, x0:x0 + minipatch, y0]
                            tmp00_ = myedgetaper(tmp00_)
                            tmp00[startx + repeat * Nz:startx + (repeat+1) * Nz,:] = tmp00_

                    if torch.max(tmp00) > thres:
                        tmp = (255 * tmp00 / torch.max(tmp00)).byte()
                        tmp = tmp.unsqueeze(0)
                        augmentations = [
                            lambda x: x,  # 旋转90度
                            lambda x: TF.hflip(x),  # 水平翻转后旋转90度
                            lambda x: TF.vflip(x),  # 垂直翻转后旋转90度
                            lambda x: TF.rotate(x, 180)  # 旋转270度
                        ]

                        for aug_func in augmentations:
                            try:
                                augmented = aug_func(tmp)
                                output_img = place_center3(augmented, minipatch)

                                # 保存图像
                                save_path = os.path.join(dir1, f"{kkk3:05d}.tif")
                                save_tiff_image(output_img, save_path)

                                kkk3 += 1
                            except Exception as e:
                                print(f"Error in augmentation: {e}")
                                continue
                        kkk3_1 += 1


                kkk2_1 = 1
                while kkk2_1 < batch * num/file_num:
                    x0 = random.randint(0, Nx - minipatch - 1)
                    y0 = random.randint(0, Ny - minipatch - 1)
                    z0 = random.randint(0, Nz - 1)

                    # 提取XOY平面 (x-y平面在固定z位置)
                    tmp00_raw = image_normalized[z0, x0:x0 + minipatch, y0:y0 + minipatch]
                    tmp00 = myedgetaper(tmp00_raw)

                    if torch.max(tmp00) > thres:
                        tmp = (255 * tmp00 / torch.max(tmp00)).byte()
                        tmp = tmp.unsqueeze(0)
                        # XOY图像的不同增强组合
                        augmentations = [
                            lambda x: x,  # 原图
                            lambda x: TF.hflip(x),  # 水平翻转
                            lambda x: TF.hflip(TF.rotate(x, 180)),  # 水平翻转+旋转180度
                            lambda x: TF.rotate(x, 180)  # 旋转180度
                        ]

                        for aug_func in augmentations:
                            try:
                                augmented = aug_func(tmp)
                                output_img = place_center3(augmented, minipatch)
                                save_path = os.path.join(dir2, f"{kkk2:05d}.tif")
                                save_tiff_image(output_img, save_path)
                                kkk2 += 1

                            except Exception as e:
                                print(f"Error in XOY augmentation: {e}")
                                continue
                        kkk2_1 += 1



if __name__ == "__main__":
    main()
    print("3D single channel TIFF processing completed successfully!")