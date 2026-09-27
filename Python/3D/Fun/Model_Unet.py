import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(conv_num):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return pool, conv


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num):
        super(ConcatBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv_rest = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    #nn.ReLU(inplace=True)
                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
                )
                for _ in range(conv_num - 1)
            ]
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.conv_rest(x)
        return x


class Unet1_step(nn.Module):
    def __init__(self, input_shape, upsample_flag, conv_block_num=3, conv_num=2):
        super(Unet1_step, self).__init__()
        self.input_shape = input_shape
        self.upsample_flag = upsample_flag
        self.conv_block_num = conv_block_num
        self.conv_num = conv_num

        self.encoder_blocks1 = nn.ModuleList()
        for n in range(conv_block_num):
            in_channels = input_shape if n == 0 else 2 ** (n + 4)
            out_channels = 2 ** (n + 5)
            self.encoder_blocks1.append(ConvBlock(in_channels, out_channels, conv_num))

        # self.mid_conv11 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1)
        # self.mid_conv12 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)

        self.mid_conv11 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels*2),
        )

        self.mid_conv12 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.decoder_blocks1 = nn.ModuleList()
        for n in range(conv_block_num):
            in_channels = out_channels
            out_channels = in_channels // 2
            self.decoder_blocks1.append(ConcatBlock(in_channels * 2, out_channels, conv_num))

        self.output1_conv = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)


    def forward(self, x):
        # Encoder
        pool = x
        conv_list = []
        for block in self.encoder_blocks1:
            pool, conv = block(pool)
            conv_list.append(conv)

        # Middle
        mid = F.leaky_relu(self.mid_conv11(pool), negative_slope=0.01)
        mid = F.leaky_relu(self.mid_conv12(mid), negative_slope=0.01)

        # Decoder
        conv = mid
        for i, block in enumerate(self.decoder_blocks1):
            conv = block(conv, conv_list[-(i + 1)])

        output1 = F.leaky_relu(self.output1_conv(conv), negative_slope=0.01)

        return output1