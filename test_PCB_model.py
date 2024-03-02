# Plz refer to the training code for comments

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
from efficientnet_pytorch import EfficientNet
import numpy as np
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import atexit
import random
from efficientnet_v2 import EfficientNetV2
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

set_seed(1)

t0 = time.time()

print('torch.cuda.is_available():', torch.cuda.is_available())
device = 'cuda'

N = 256
bs = 16
power = 1000
npcb = 10
dualeB = 64.91  #

wvl = 0.632e-6
delta = 3e-3
Cn2 = 1e-14
delta_Z = 2e3
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
phasePSD = torch.tile(phase_PSD, dims=[bs, 1, 1])

H = torch.exp(torch.complex(torch.zeros_like(FX), -pi*wvl*delta_Z*(FX**2+FY**2)))
H = fft.fftshift(H)
H = torch.tile(H, dims=[bs, npcb, 1, 1])

circ_mask = torch.zeros(N, N, device=device)
index = torch.where(torch.sqrt(r2) < 5e-2)
circ_mask[index] = 1.0
circ_mask = torch.tile(circ_mask, dims=[bs, npcb, 1, 1])


def propTF(H, Uin):
    U1 = fft.fft2(fft.fftshift(Uin, dim=(-2, -1)), dim=(-2, -1))
    U2 = torch.mul(H, U1)
    Uout = fft.ifftshift(fft.ifft2(U2, dim=(-2, -1)), dim=(-2, -1))
    return Uout


fie_real = torch.randn(npcb, N, N)
fie_img = torch.randn(npcb, N, N)
fie_rander = torch.complex(fie_real, fie_img).to(device)
def pcbsyn(w0, Lcr):
    U0 = torch.complex(torch.zeros(bs, npcb, N, N, device=device), torch.zeros(bs, npcb, N, N, device=device))
    for i in range(bs):
        w = w0[i]
        u0 = torch.exp(-r2 / (w**2))
        u0 = torch.tile(u0, dims=[npcb, 1, 1])

        sigma_f = Lcr[i]
        sigma_r = torch.sqrt(4 * pi * sigma_f ** 4 / Lcr[i] ** 2)
        Filter = torch.exp(-pi ** 2 * sigma_f ** 2 * f ** 2)
        Filter = torch.tile(Filter, dims=[npcb, 1, 1])
        FIE = Filter * fie_rander
        FIE = fft.fftshift(FIE, dim=(-2, -1))
        fie = fft.ifft2(FIE, dim=(-2, -1)) * sigma_r * (N/delta)
        fie = torch.real(fie) * 10
        phase = torch.exp(torch.complex(torch.zeros_like(fie), fie))
        U0[i] = u0 * phase  # torch.Size([npcb, 256, 256])

    return U0


net = EfficientNetV2('s', in_channels=1, n_classes=2, in_spatial_shape=256).to(device)
net.load_state_dict(torch.load('logs/2km_1e-14/pcb/2024-02-24-15-46_steps50000.pt')['net'])


total_steps = 10000  # test steps
max_w0 = 0.1
max_Lcr = 0.5
with torch.no_grad():
    net.eval()
    with trange(0, total_steps, desc='Training', ncols=0) as pbar:
        fluxs = torch.zeros(0, device=device)
        Pas = torch.zeros(0, device=device)
        input = 0.1*torch.ones(bs, 1, 256, 256, device=device)
        for step in pbar:

            output = net(input)
            w0, Lcr = output[:, 0], output[:, 1]

            w0 = max_w0 * F.sigmoid(w0) + 0.005
            Lcr = max_Lcr * F.sigmoid(Lcr) + 0.001

            U0 = pcbsyn(w0, Lcr)
            amp = U0.abs()
            phase = torch.angle(U0)
            output_power = torch.sum(amp ** 2, dim=[-2, -1])
            power_ratio = torch.sqrt(power / output_power)
            U0 = torch.mul(U0, power_ratio.unsqueeze(2).unsqueeze(3))

            rand_real = torch.randn(bs, N, N, dtype=torch.float)
            rand_img = torch.randn(bs, N, N, dtype=torch.float)
            rander = torch.complex(rand_real, rand_img).to(device)
            cn = torch.mul(rander, phasePSD)
            screen0 = torch.real(fft.ifftshift(fft.ifft2(fft.ifftshift(cn, dim=(-2, -1)), dim=(-2, -1)),
                                              dim=(-2, -1)) * N ** 2)
            screen0 = torch.unsqueeze(screen0, dim=1)
            screen0 = torch.tile(screen0, dims=[1, npcb, 1, 1])

            Uin = U0 * torch.exp(torch.complex(torch.zeros_like(amp), screen0))

            U = propTF(H, Uin)

            Irridiance = (U.abs() * circ_mask) ** 2
            flux = torch.sum(Irridiance, dim=[1, 2, 3]) / npcb

            e1b = torch.mean(flux ** 2)  # postfix "b" for the current batch
            e2b = torch.mean(flux) ** 2
            sib = e1b / e2b - 1.0  # batch SI

            fluxs = torch.cat((fluxs, flux))
            Pab = torch.sum(flux) / bs    # batch PIB
            Pas = torch.cat((Pas, Pab.unsqueeze(0)))

            SNRb = 1 / torch.sqrt(dualeB / Pab**2 + sib)  # batch SNR
            SNRdB = 10 * (torch.log(SNRb) / np.log(10))
            pbar.set_postfix(
                {'SNR': '{0:1.4f}'.format(SNRdB), 'SI': '{0:1.4f}'.format(sib), 'Aver Power': '{0:1.4f}'.format(Pab)})

        e1 = torch.mean(fluxs ** 2)
        e2 = torch.mean(fluxs) ** 2
        si = e1 / e2 - 1.0  # average SI over all steps

        Pa = torch.mean(Pas)  # average PIB over all steps

        SNR = 1 / torch.sqrt(dualeB / Pa**2 + si)   # average SNR over all steps
        SNRdB = 10 * (torch.log(SNR) / np.log(10))
        print('SNR(liner)', SNR.item(), 'SNR(dB): ', SNRdB.item(), 'Pa: ', Pa.item(), 'SI: ', si.item())


print(time.time() - t0)
now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("finished training at: ", now.strftime("%Y-%m-%d-%H-%M"))

