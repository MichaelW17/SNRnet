# Plz refer to the training & test code for comments

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
from torch import fft, special
from tqdm import trange
import datetime
import pytz
import time
# from efficientnet_pytorch import EfficientNet
import numpy as np
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import atexit
import random
import gc
import kornia.geometry as T


cudnn.deterministic = True
cudnn.benchmark = True
cudnn.enabled = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.device_count()


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.autograd.set_detect_anomaly(True)


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(2)

t0 = time.time()

print('torch.cuda.is_available():', torch.cuda.is_available())
device = 'cpu'

N = 256
bs = 16
power = 1000  # 用于归一化光束能量
dualeB = 64.91

wvl = 0.632e-6
delta = 3e-3
Cn2 = 1e-14  # 湍流强度
delta_Z = 2e3  # 传输距离
pi = 3.1415926535
k = 2*pi/wvl

x1 = torch.linspace(-N*delta/2, (N/2-1)*delta, N, device=device)
xx, yy = torch.meshgrid(x1, x1, indexing='ij')
r2 = xx ** 2 + yy ** 2  # 注意这里是r的平方

fx = torch.linspace(-0.5/delta, 0.5/delta-1/(N*delta), N, device=device)
FX, FY = torch.meshgrid(fx, fx, indexing='ij')
f = torch.sqrt(FX**2 + FY**2)

phase_PSD = 2*pi * k**2 * delta_Z * 0.033 * Cn2 * (2*pi)**(-11/3) * (4*pi**2) / (f**(11/3))
phase_PSD[128, 128] = 0
phase_PSD = torch.torch.sqrt(phase_PSD) / (N*delta)
print(phase_PSD.dtype)
phasePSD = torch.tile(phase_PSD, dims=[bs, 1, 1])

H = torch.exp(torch.complex(torch.zeros_like(FX), -pi*wvl*delta_Z*(FX**2+FY**2)))
H = fft.fftshift(H)
H = torch.tile(H, dims=[bs, 1, 1])

circ_mask = torch.zeros(N, N, device=device)
index = torch.where(torch.sqrt(r2) < 3e-2)
circ_mask[index] = 1.0
circ_mask = torch.tile(circ_mask, dims=[bs, 1, 1])


def propTF(H, Uin):
    U1 = fft.fft2(fft.fftshift(Uin, dim=(-2, -1)), dim=(-2, -1))
    U2 = torch.mul(H, U1)
    Uout = fft.ifftshift(fft.ifft2(U2, dim=(-2, -1)), dim=(-2, -1))
    return Uout


total_steps = 1000

n_w0s = 21  # w0 value points
w0s = torch.linspace(0.005, 0.1, n_w0s, device=device)  # w0 search range

SNRs= torch.zeros(0, device=device)
SIs = torch.zeros(0, device=device)

for w0 in w0s:
    print(w0)
    with trange(0, total_steps, desc='Training', ncols=0) as pbar:
        fluxs = torch.zeros(0, device=device)  # 最终的长度为bs*total_steps
        Pas = torch.zeros(0, device=device)  # 最终的长度为total_steps
        for step in pbar:

            w0_opt = w0 * torch.ones(bs, device=device)
            amp3 = torch.exp(-(xx ** 2 + yy ** 2) / (w0_opt.unsqueeze(1).unsqueeze(2) ** 2))  # generate Gaussian beam
            output_power3 = torch.sum(amp3 ** 2, dim=[1, 2])
            power_ratio3 = torch.sqrt(power / output_power3)
            U03 = torch.mul(amp3, power_ratio3.unsqueeze(1).unsqueeze(2))  # power norm


            rand_real = torch.randn(bs, N, N, dtype=torch.float)
            rand_img = torch.randn(bs, N, N, dtype=torch.float)
            rander = torch.complex(rand_real, rand_img).to(device)
            cn = torch.mul(rander, phasePSD)
            screen = torch.real(fft.ifftshift(fft.ifft2(fft.ifftshift(cn, dim=(-2, -1)), dim=(-2, -1)),
                                              dim=(-2, -1)) * N ** 2)

            # apply phase screen
            Uin3 = U03 * torch.exp(torch.complex(torch.zeros_like(U03).to(device), screen))
            # free space prop
            U = propTF(H, Uin3)

            Irridiance = (U.abs() * circ_mask) ** 2

            flux = torch.sum(Irridiance, dim=[1, 2])
            e1b = torch.mean(flux ** 2)
            e2b = torch.mean(flux) ** 2
            sib = e1b / e2b - 1.0

            fluxs = torch.cat((fluxs, flux))

            Pab = torch.sum(flux) / bs
            Pas = torch.cat((Pas, Pab.unsqueeze(0)))

            e1 = torch.mean(fluxs ** 2)
            e2 = torch.mean(fluxs) ** 2
            si = e1 / e2 - 1.0
            Pa = torch.mean(Pas)

            SNR = 1 / torch.sqrt(dualeB / Pa ** 2 + si)  # SNR_1是平均信噪比<SNR>的倒数，越小越好
            SNRdB = 10 * (torch.log(SNR) / np.log(10))  # 通过换底公式计算dB单位的信噪比

            pbar.set_postfix({'w0': '{0:1.3f}'.format(w0), 'SNR': '{0:1.4f}'.format(SNRdB), 'SI': '{0:1.4f}'.format(si), 'Aver Power': '{0:1.4f}'.format(Pa)})

        e1 = torch.mean(fluxs ** 2)
        e2 = torch.mean(fluxs) ** 2
        si = e1 / e2 - 1.0

        Pa = torch.mean(Pas)

        SNR = 1 / torch.sqrt(dualeB / Pa**2 + si)
        SNRdB = 10 * (torch.log(SNR) / np.log(10))


        SNRs = torch.cat((SNRs, SNRdB.unsqueeze(0)))
        SIs = torch.cat((SIs, si.unsqueeze(0)))

plt.figure(num=1, figsize=(5, 5))
plt.plot(w0s.cpu().numpy(), SNRs.cpu().numpy())
plt.show(block=True)

print(SNRs, '\n', SIs)


print(time.time() - t0)
now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("finished at: ", now.strftime("%Y-%m-%d-%H-%M"))


