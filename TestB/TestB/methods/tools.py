import numpy as np
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

import torch

START_EPS = 16 / 255


def calculate_energy_ratio(features):
    # 将张量从 CUDA 设备移动到 CPU（如果需要）
    if features.is_cuda:
        features = features.cpu()

    # 进行傅里叶变换
    f_transform = torch.fft.fft2(features)
    f_transform_shifted = torch.fft.fftshift(f_transform)

    # 计算幅度谱
    magnitude_spectrum = torch.abs(f_transform_shifted)

    # 计算总能量
    total_energy = torch.sum(magnitude_spectrum ** 2)

    # 设计低通滤波器和高通滤波器
    rows, cols = features.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask_low = torch.zeros((rows, cols), dtype=torch.uint8)
    mask_high = torch.ones((rows, cols), dtype=torch.uint8)

    # 低通滤波器
    mask_low[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 高通滤波器
    mask_high[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 应用滤波器
    low_freq_energy = torch.sum((magnitude_spectrum * mask_low) ** 2)
    high_freq_energy = torch.sum((magnitude_spectrum * mask_high) ** 2)

    # 计算能量占比
    low_freq_ratio = low_freq_energy / total_energy
    high_freq_ratio = high_freq_energy / total_energy

    return low_freq_ratio.item(), high_freq_ratio.item()


def loss_fn_example(phase):
    # 假设一个简单的正则化目标
    return torch.sum(phase ** 2)


# 1. 高频和低频分量提取
def extract_high_low_frequency_components(feature):
    """
    使用 DWT 前向变换提取高频和低频分量。

    参数：
    feature (Tensor): 输入特征图，形状为 (N, C, H, W)，N 为批量大小，C 为通道数，H 和 W 为特征图的高度和宽度。

    返回：
    low_freq (Tensor): 低频分量。
    high_freq (Tensor): 高频分量。
    """
    # 使用 DWT 前向变换，选择小波类型（例如 'haar'）和小波的级数（例如 1）
    dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()

    # 输入特征图进行小波变换，返回四个小波系数：LL、LH、HL、HH
    LL, HH = dwt(feature)

    # 返回低频和高频分量
    low_freq = LL  # 低频分量
    high_freq = HH[0]
    # print("high_freq 的类型:", type(high_freq))

    return low_freq, high_freq


# 2. 添加扰动
def add_perturbation(init_input, epsilon, data_grad):
    # print("init_input.shape",init_input.shape)
    # random start init_input
    init_input = init_input + torch.empty_like(init_input).uniform_(-START_EPS, START_EPS)

    sign_data_grad = data_grad.sign()
    adv_input = init_input + epsilon * sign_data_grad
    # print("adv_input.shape",adv_input.shape)
    return adv_input


# 2. 添加扰动
def add_perturbation_high(init_input, epsilon, data_grad):
    # print("init_input.shape",init_input.shape)
    # random start init_input
    init_input = init_input + torch.empty_like(init_input).uniform_(-START_EPS, START_EPS)

    sign_data_grad = data_grad.sign()
    adv_input = init_input + 0.2 * epsilon * sign_data_grad
    # print("adv_input.shape",adv_input.shape)
    return adv_input


# 3. 逆变换
def reconstruct_feature(low_freq, high_freq):
    # print("high_freq 的类型:", type(high_freq))

    idwt = DWTInverse(wave='haar', mode='zero').cuda()
    # 确保 high_freq 是列表，如果只是一个张量，需包装为列表
    if not isinstance(high_freq, list):
        high_freq = [high_freq]
    # 将高低频分量重建成特征
    reconstructed_feature = idwt((low_freq, high_freq))

    # 返回重构后的特征图的实部
    return reconstructed_feature


def mutual_attention(q, k):
    assert (q.size() == k.size())
    weight = q.mul(k)
    weight_sig = torch.sigmoid(weight)
    v = k.mul(weight_sig)
    return v


def consistency_loss(scoresM1, scoresM2, type='euclidean'):
    if (type == 'euclidean'):
        avg_pro = (scoresM1 + scoresM2) / 2.0
        matrix1 = torch.sqrt(torch.sum((scoresM1 - avg_pro) ** 2, dim=1))
        matrix2 = torch.sqrt(torch.sum((scoresM2 - avg_pro) ** 2, dim=1))
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1 + dis2) / 2.0
    elif (type == 'KL1'):
        avg_pro = (scoresM1 + scoresM2) / 2.0
        matrix1 = torch.sum(
            F.softmax(scoresM1, dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(avg_pro, dim=-1)), 1)
        matrix2 = torch.sum(
            F.softmax(scoresM2, dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(avg_pro, dim=-1)), 1)
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1 + dis2) / 2.0
    elif (type == 'KL2'):
        matrix = torch.sum(
            F.softmax(scoresM2, dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(scoresM1, dim=-1)), 1)
        dis = torch.mean(matrix)
    elif (type == 'KL3'):
        matrix = torch.sum(
            F.softmax(scoresM1, dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(scoresM2, dim=-1)), 1)
        dis = torch.mean(matrix)
    else:
        return
    return dis
