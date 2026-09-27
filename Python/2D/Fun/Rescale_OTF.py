import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

def generate_scales(batch_size: int,
                   min_scale: float,
                   max_scale: float) -> tuple:
    """
    生成批量随机缩放因子
    参数:
        batch_size: 批次大小
        min_scale: 最小缩放比例
        max_scale: 最大缩放比例
    返回:
        (x_scales, y_scales): 每个样本独立的缩放因子元组
    """
    scales = torch.rand(batch_size, 2) * (max_scale - min_scale) + min_scale
    return scales[:, 0], scales[:, 1]

def dynamic_resize(tensor: torch.Tensor,
                   minipatch: int,
                   x_scale: float,
                   y_scale: float,
                   mode: str = 'bilinear',) -> torch.Tensor:
    """
    动态调整四维张量尺寸 (支持batch维度)
    参数:
        tensor: 输入张量 (batch, 1, 128, 128)
        x_scale: 宽度缩放因子 (0.5 ≤ x_scale ≤ 1.5)
        y_scale: 高度缩放因子 (0.5 ≤ y_scale ≤ 1.5)
        mode: 插值模式 ('nearest', 'bilinear', 'bicubic')
    返回:
        resized_tensor: 缩放后张量 (batch, 1, new_H, new_W)
    """
    # 计算目标尺寸
    new_H = int(minipatch * y_scale)
    new_W = int(minipatch * x_scale)

    # 插值缩放 (保持通道维度)
    resized = F.interpolate(
        input=tensor,
        size=(new_H, new_W),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    return resized

def fit_grayscale_tensor(A: torch.Tensor, minipatch: int) -> torch.Tensor:
    """
    将任意尺寸的批次张量适配到指定大小（居中裁剪/填充）
    Args:
        A: 输入张量 (batch, 1, H, W)
        minipatch: 目标尺寸
    Returns:
        output: (batch, 1, minipatch, minipatch)
    """
    batch_size, _, h, w = A.shape

    # 创建目标张量（保持设备一致）
    output = torch.zeros((batch_size, 1, minipatch, minipatch),
                         dtype=A.dtype, device=A.device)

    # 计算源区域裁剪索引
    src_h_start = max(0, (h - minipatch) // 2)
    src_h_end = min(h, src_h_start + minipatch)
    src_w_start = max(0, (w - minipatch) // 2)
    src_w_end = min(w, src_w_start + minipatch)

    # 计算目标区域位置
    dest_h_start = max(0, (minipatch - h) // 2)
    dest_h_end = dest_h_start + (src_h_end - src_h_start)
    dest_w_start = max(0, (minipatch - w) // 2)
    dest_w_end = dest_w_start + (src_w_end - src_w_start)

    # 执行裁剪/填充操作
    output[:, :, dest_h_start:dest_h_end, dest_w_start:dest_w_end] = \
        A[:, :, src_h_start:src_h_end, src_w_start:src_w_end]

    return output


def process_otf_tensor(OTF0: torch.Tensor, minipatch: int, variance: float) -> torch.Tensor:
    """
    批次处理OTF张量的随机缩放与适配
    Args:
        OTF0: 输入张量 (batch, 1, 128, 128)
        minipatch: 目标基准尺寸（如512）
        variance: 缩放方差系数
    Returns:
        OTF_resize: (batch, 1, minipatch, minipatch)
    """

    x_scales, y_scales = generate_scales(batch_size=OTF0.size(0),min_scale = 1,max_scale = 1)

    outputs = []
    for i in range(OTF0.size(0)):
        resized = dynamic_resize(
            tensor=OTF0[i].unsqueeze(0),
            minipatch = minipatch,
            x_scale=x_scales[i].item(),
            y_scale=y_scales[i].item(),
            mode='bilinear',
        )
        outputs.append(resized)

    OTF = torch.zeros(OTF0.size())
    for i in range(OTF0.size(0)):
        OTF[i,0,:,:] = fit_grayscale_tensor(outputs[i], minipatch)
    # 适配到minipatch尺寸
    return OTF