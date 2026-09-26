import numpy as np
import torch
import torch.nn.functional as F


def fourier_inter(image_stack):
    """
    SN2N tool: Fourier re-scale
    ------
    image_stack
        image TO Fourier interpolation

    Returns
    -------
    imgf1: image with 2x size
    """
    imsize = [128, 128]
    if image_stack.ndim == 2:
        image_stack = np.expand_dims(image_stack, 0)
    [t, xx, x, y] = image_stack.shape
    imgf1 = np.zeros((t, 1, imsize[0], imsize[1]))

    for slice in range(t):
        img = image_stack[slice,0, :, :]
        imgsz = np.array([x, y])
        tem1 = np.divide(imgsz, 2)
        tem2 = np.multiply(tem1, 2)
        tem3 = np.subtract(imgsz, tem2)
        b = (tem3 == np.array([0, 0]))
        if b[0] == True:
            sz = imgsz - 1
        else:
            sz = imgsz
        n = np.array([2, 2])
        ttem1 = np.add(np.ceil(np.divide(sz, 2)), 1)
        ttem2 = np.multiply(np.floor(np.divide(sz, 2)), np.subtract(n, 1))
        idx = np.add(ttem1, ttem2)
        padsize = np.array([x / 2, y / 2], dtype='int')
        pad_wid = np.ceil(padsize[0]).astype('int')
        img = np.pad(img, ((pad_wid, 0), (pad_wid, 0)), 'symmetric')
        img = np.pad(img, ((0, pad_wid), (0, pad_wid)), 'symmetric')
        imgsz1 = np.array(img.shape)
        tttem1 = np.multiply(n, imgsz1)
        tttem2 = np.subtract(n, 1)
        newsz = np.round(np.subtract(tttem1, tttem2))
        img1 = interpft(img, newsz[0], 0)
        img1 = interpft(img1, newsz[1], 1)
        idx = idx.astype('int')
        ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
        ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
        imgf1[slice, :, :] = img1[idx[0] - 1:idx[0] + ttttem1, idx[1] - 1:idx[1] + ttttem2]
        imgf1[imgf1 < 0] = 0
    return imgf1


def interpft(x, ny, dim=0):
    '''
    Function to interpolate using FT method, based on matlab interpft()
    ------
    x
        array for interpolation
    ny
        length of returned vector post-interpolation
    dim
        performs interpolation along dimension DIM
        {default: 0}

    Returns
    -------
    y: interpolated data
    '''

    if dim >= 1:
        # if interpolating along columns, dim = 1
        x = np.swapaxes(x, 0, dim)
    # temporarily swap axes so calculations are universal regardless of dim
    if len(x.shape) == 1:
        # interpolation should always happen along same axis ultimately
        x = np.expand_dims(x, axis=1)

    siz = x.shape
    [m, n] = x.shape

    a = np.fft.fft(x, m, 0)
    nyqst = int(np.ceil((m + 1) / 2))
    b = np.concatenate((a[0:nyqst, :], np.zeros(shape=(ny - m, n)), a[nyqst:m, :]), 0)

    if np.remainder(m, 2) == 0:
        b[nyqst, :] = b[nyqst, :] / 2
        b[nyqst + ny - m, :] = b[nyqst, :]

    y = np.fft.irfft(b, b.shape[0], 0)
    y = y * ny / m
    y = np.reshape(y, [y.shape[0], siz[1]])
    y = np.squeeze(y)

    if dim >= 1:
        # switches dimensions back here to get desired form
        y = np.swapaxes(y, 0, dim)

    return y


def normalize2d(image):
    max_prc = 100
    min_prc = 0
    image = (image - np.percentile(image, min_prc)) / (np.percentile(image, max_prc) - np.percentile(image, min_prc) + 1e-7)
    image[image > 1] = 1
    image[image < 0] = 0
    return image

def block2d(image_stack):
    """
    Self-supervised data generator in xy
    ------
    image_stack
        image TO generate

    Returns
    -------
    left, right: noisy data pair
    """
    [b, t, x, y] = image_stack.shape
    upleft = []
    upright = []
    downright = []
    downleft = []
    left = []
    right = []
    for i in range(b):
        ul = image_stack[i, :, 0::2, 0::2]
        ur = image_stack[i, :, 0::2, 1::2]
        dr = image_stack[i, :, 1::2, 1::2]
        dl = image_stack[i, :, 1::2, 0::2]
        upleft.append(ul)
        upright.append(ur)
        downright.append(dr)
        downleft.append(dl)
        left.append(ul / 2 + dr / 2)
        right.append(ur / 2 + dl / 2)
    left = torch.stack(left)
    right = torch.stack(right)

    left = torch.from_numpy(fourier_inter(left))
    right = torch.from_numpy(fourier_inter(right))


    return left, right


def block2d_2(image_stack):
    """
    Self-supervised data generator in xy
    ------
    image_stack
        image TO generate

    Returns
    -------
    left, right: noisy data pair
    """
    [b, t, x, y] = image_stack.shape
    upleft = []
    upright = []
    downright = []
    downleft = []
    left1 = []
    right1 = []
    left2 = []
    right2 = []
    for i in range(b):
        ul = image_stack[i, :, 0::2, 0::2]
        ur = image_stack[i, :, 0::2, 1::2]
        dr = image_stack[i, :, 1::2, 1::2]
        dl = image_stack[i, :, 1::2, 0::2]
        upleft.append(ul)
        upright.append(ur)
        downright.append(dr)
        downleft.append(dl)
        left1.append(ul)
        left2.append(ur)
        right1.append(dr)
        right2.append(dl)
    left1 = torch.stack(left1)
    right1 = torch.stack(right1)
    left2 = torch.stack(left2)
    right2 = torch.stack(right2)

    left1 = torch.from_numpy(fourier_inter(left1))
    right1 = torch.from_numpy(fourier_inter(right1))
    left2 = torch.from_numpy(fourier_inter(left2))
    right2 = torch.from_numpy(fourier_inter(right2))


    return left1, right1, left2, right2

