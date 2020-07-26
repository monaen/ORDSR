import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from torch.nn.modules.upsampling import Upsample


def zigzag(n):
    '''zigzag rows'''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    return {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}


def get_DCT_2D_Basis(N=8):
    B = np.zeros([N, N, N, N], dtype=np.float64)
    [I, J] = np.meshgrid(range(N), range(N))
    A = np.sqrt(2.0 / N) * np.cos(((2 * I + 1) * J * np.pi) / (N * 2))
    A[0, :] = A[0, :] / np.sqrt(2.0)
    A = A.T
    for i in range(N):
        for j in range(N):
            B[:, :, i, j] = np.outer(A[:, i], A[:, j])

    # rearrange the 64 basis functions in B to zigzag order

    Bzigzag = np.zeros([N * N, 1, N, N], dtype=np.float64)
    zigzag_dict = zigzag(N)

    for i in range(N):
        for j in range(N):
            Bzigzag[zigzag_dict[(i, j)], 0] = B[:, :, i, j]

    Bzigzag = torch.from_numpy(Bzigzag.astype(np.float32))

    return Bzigzag


def get_pixelshuffle_2D_Basis(N=8):
    Bzigzag = np.zeros([N * N, 1, N, N], dtype=np.float64)
    for i in range(N * N):
        h = i // N
        w = i % N
        Bzigzag[i, 0, h, w] = 1

    Bzigzag = torch.from_numpy(Bzigzag.astype(np.float32))

    return Bzigzag


# ======================================================================================================= #
#                                              ORDSRModel                                                 #
# ======================================================================================================= #

class DCTConv(nn.Module):
    '''
    DCT Convolution and IDCT Convolution with sharing weights
    '''
    def __init__(self, stride=1, padding=0, dilation=1, groups=1, blocksize=8):
        super(DCTConv, self).__init__()
        self.weights = nn.Parameter(self.get_DCT_2D_Basis(N=blocksize))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def get_DCT_2D_Basis(self, N=8):
        B = np.zeros([N, N, N, N], dtype=np.float64)
        [I, J] = np.meshgrid(range(N), range(N))
        A = np.sqrt(2.0/N)*np.cos(((2*I+1)*J*np.pi)/(N*2))
        A[0, :] = A[0, :] / np.sqrt(2.0)
        A = A.T
        for i in range(N):
            for j in range(N):
                B[:, :, i, j] = np.outer(A[:, i], A[:, j])

        # rearrange the 64 basis functions in B to zigzag order

        Bzigzag = np.zeros([N*N, 1, N, N], dtype=np.float64)
        zigzag_dict = zigzag(N)

        for i in range(N):
            for j in range(N):
                Bzigzag[zigzag_dict[(i, j)], 0] = B[:, :, i, j]

        Bzigzag = torch.from_numpy(Bzigzag.astype(np.float32))

        return Bzigzag

    def ConvDCT2d(self, x):
        out = F.conv2d(x, self.weights, stride=self.stride, padding=self.padding, dilation=self.dilation,
                       groups=self.groups, bias=None)
        return out

    def InverseConvDCT(self, x):
        out = F.conv_transpose2d(x, self.weights, stride=self.stride, padding=self.padding, output_padding=0,
                                 dilation=self.dilation, groups=self.groups, bias=None)
        return out


class ORDSRModel(nn.Module):
    ''' modified ATDSRModel

    Experiments 06: Setting S=8
    '''

    def __init__(self, in_nc=3, out_nc=3, N=8, S=8, upscale=4):
        super(ORDSRModel, self).__init__()
        self.upscale = upscale
        self.N = N
        self.S = S
        self.upsampling = Upsample(scale_factor=self.upscale, mode='bicubic')

        # ================================ Extract Shallow DCT Features ================================ #
        self.DCTTrans = DCTConv(stride=self.S, padding=0, blocksize=self.N)

        # ================================ Learn the High DCT Features ================================ #
        self.conv0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(in_channels=60, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu8 = nn.LeakyReLU(0.2, inplace=True)

        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu9 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu10 = nn.LeakyReLU(0.2, inplace=True)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu11 = nn.LeakyReLU(0.2, inplace=True)

        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu12 = nn.LeakyReLU(0.2, inplace=True)

        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu13 = nn.LeakyReLU(0.2, inplace=True)

        self.conv14 = nn.Conv2d(in_channels=64, out_channels=60, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu14 = nn.LeakyReLU(0.2, inplace=True)

    def get_DCT_2D_Basis(self, N=8):
        B = np.zeros([N, N, N, N], dtype=np.float64)
        [I, J] = np.meshgrid(range(N), range(N))
        A = np.sqrt(2.0/N)*np.cos(((2*I+1)*J*np.pi)/(N*2))
        A[0, :] = A[0, :] / np.sqrt(2.0)
        A = A.T
        for i in range(N):
            for j in range(N):
                B[:, :, i, j] = np.outer(A[:, i], A[:, j])

        # rearrange the 64 basis functions in B to zigzag order

        Bzigzag = np.zeros([N*N, 1, N, N], dtype=np.float64)
        zigzag_dict = zigzag(N)

        for i in range(N):
            for j in range(N):
                Bzigzag[zigzag_dict[(i, j)], 0] = B[:, :, i, j]

        Bzigzag = torch.from_numpy(Bzigzag.astype(np.float32))

        return Bzigzag

    def set_init_dct_weights(self, layer):
        # layer.weight.requires_grad = False
        dct_weights = self.get_DCT_2D_Basis(N=8)  # 8 blocks
        layer.weight.data = dct_weights

    def start_grad(self):
        self.CDCT.weight.requires_grad = True
        self.IDCT.weight.requires_grad = True

    def get_dct_weight(self):
        return self.CDCT.weight

    def get_conv_weights(self):
        weights = []
        for module in self.children():
            if isinstance(module, (nn.modules.conv.Conv2d, nn.modules.conv.ConvTranspose2d)):
                weights.append(module.weight)

        return weights


    def forward(self, x):

        # ======== Pre-Upsampling ======== #
        x = self.upsampling(x)

        # ============ Step 1 ============ #
        # DCT_block = self.conv0(self.CDCT(x))  # out
        DCT_block = self.conv0(self.DCTTrans.ConvDCT2d(x))

        # pdb.set_trace()
        # ============ Step 2 ============ #
        f_low = DCT_block[:, :4, :, :]   # shape = [16, 4,  16, 16]
        f_high = DCT_block[:, 4:, :, :]  # shape = [16, 60, 16 ,16]

        # ============ Step 3 ============ #
        tmp = self.relu1(self.conv1(f_high))
        tmp = self.relu2(self.conv2(tmp))
        tmp = self.relu3(self.conv3(tmp))
        tmp = self.relu4(self.conv4(tmp))
        tmp = self.relu5(self.conv5(tmp))
        tmp = self.relu6(self.conv6(tmp))
        tmp = self.relu7(self.conv7(tmp))
        tmp = self.relu8(self.conv8(tmp))
        tmp = self.relu9(self.conv9(tmp))
        tmp = self.relu10(self.conv10(tmp))
        tmp = self.relu11(self.conv11(tmp))
        tmp = self.relu12(self.conv12(tmp))
        tmp = self.relu13(self.conv13(tmp))
        f_high_recons = self.relu14(self.conv14(tmp)) + f_high

        # ============ Step 4 ============ #
        f_out = torch.cat([f_low, f_high_recons], 1)

        # ============ Step 5 ============ #
        # out = ((self.N / self.S) ** -2) * self.IDCT(f_out)
        out = ((self.N / self.S) ** -2) * self.DCTTrans.InverseConvDCT(f_out)

        return out
