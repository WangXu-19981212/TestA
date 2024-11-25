import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import torch
import torch.nn.functional as F


def loss_fn_example(phase):
    # 假设一个简单的正则化目标
    return torch.sum(phase ** 2)


# 1. 相位谱提取
def extract_phase_spectrum(image):
    # 使用 PyTorch 的傅里叶变换
    f_transform = torch.fft.fft2(image)  # PyTorch 的 2D FFT
    phase_spectrum = torch.angle(f_transform).float()  # 提取相位谱
    return phase_spectrum


# 2. 相位谱扰动
def perturb_phase_spectrum(phase_spectrum, epsilon, k_rt, iterations, loss_fn):
    # 添加随机噪声
    perturbed_phase_spectrum = phase_spectrum + k_rt * torch.randn_like(phase_spectrum)

    # 计算当前迭代的扰动强度系数
    perturbation_strength = max(0.1 / (iterations + 1), 0.0005)
    # print('perturbation_strength',perturbation_strength)

    loss = loss_fn_example(perturbed_phase_spectrum)
    gradient = torch.autograd.grad(loss, perturbed_phase_spectrum, retain_graph=True)[0]
    perturbed_phase_spectrum += perturbation_strength * epsilon * torch.sign(gradient)

    # 约束扰动范围（可选）
    # perturbed_phase_spectrum = torch.clamp(perturbed_phase_spectrum, min=-torch.pi, max=torch.pi)

    return perturbed_phase_spectrum.detach()


# 3. 逆傅里叶变换

def reconstruct_image(amplitude_spectrum, perturbed_phase_spectrum):
    # Ensure that amplitude and phase are on the same device (e.g., GPU or CPU)
    device = amplitude_spectrum.device

    # Combine the amplitude spectrum and the perturbed phase spectrum to form the complex spectrum
    combined_spectrum = amplitude_spectrum * torch.exp(1j * perturbed_phase_spectrum)

    # Perform the inverse FFT to reconstruct the image
    reconstructed_image = torch.fft.ifft2(combined_spectrum)

    # Take the real part of the reconstructed image (since it should be real)
    reconstructed_image = torch.real(reconstructed_image)

    return reconstructed_image


# 4. 应用相位谱扰动
def change_new_phase_spectrum(image, perturbed_phase_spectrum, p_thred=0):
    """
    应用相位谱扰动，生成对抗样本。

    Args:
        image (torch.Tensor): 输入图像。
        perturbed_phase_spectrum (torch.Tensor): 扰动后的相位谱。
        p_thred (float): 扰动阈值。

    Returns:
        adv_image (torch.Tensor): 对抗样本。
    """
    # 提取幅度谱
    f_transform = torch.fft.fft2(image)
    amplitude_spectrum = torch.abs(f_transform)

    # 重建图像
    adv_image = reconstruct_image(amplitude_spectrum, perturbed_phase_spectrum)

    return adv_image


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
