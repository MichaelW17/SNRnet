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
from efficientnet_v2_4OSB import EfficientNetV2
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
dualeB = 64.91

wvl = 0.632e-6
delta = 4e-3
Cn2 = 1e-15
delta_Z = 5e3
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


base = torch.load("data/base2cm.pth")

def mixfield3(pred, phases):
    eigens = pred
    U0 = torch.complex(torch.zeros(bs, npcb, N, N, device=device), torch.zeros(bs, npcb, N, N, device=device))
    for i in range(bs):
        eigenv = torch.unsqueeze(eigens[i], 2)
        eigenv = torch.unsqueeze(eigenv, 3)
        graph = base * eigenv

        phase = torch.exp(torch.complex(torch.zeros_like(phases[i]), phases[i]))
        complex_modes = graph * phase.unsqueeze(2).unsqueeze(3)
        U0[i] = torch.sum(complex_modes, dim=1)

    return U0


net = EfficientNetV2('s', in_channels=1, n_classes=128, in_spatial_shape=256, npcb=npcb).to(device)
net.load_state_dict(torch.load('logs/2km_1e-14/osb/2024-02-24-04-52_steps50000.pt')['net'])


total_steps = 10000

with torch.no_grad():
    net.eval()
    with trange(0, total_steps, desc='Training', ncols=0) as pbar:
        fluxs = torch.zeros(0, device=device)
        Pas = torch.zeros(0, device=device)
        input1 = 0.1*torch.ones(bs, 1, 256, 256, device=device)
        for step in pbar:

            output = net(input1)
            eigenvalues, sigma_s, phases = output[:, :, :64], output[:, :, 64], output[:, :, 65:]

            eigenvalues = F.relu(eigenvalues)
            phases = torch.cat((phases, torch.zeros(bs, npcb, 1, device=device)), dim=2)
            phases = F.tanh(phases) * pi

            U0 = mixfield3(eigenvalues, phases)
            amp = U0.abs()

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
            Pab = torch.sum(flux) / bs  # batch PIB
            Pas = torch.cat((Pas, Pab.unsqueeze(0)))

            SNRb = 1 / torch.sqrt(dualeB / Pab**2 + sib)  # batch SNR
            SNRdB = 10 * (torch.log(SNRb) / np.log(10))
            pbar.set_postfix(
                {'SNR': '{0:1.4f}'.format(SNRdB), 'SI': '{0:1.4f}'.format(sib), 'Aver Power': '{0:1.4f}'.format(Pab)})

        e1 = torch.mean(fluxs ** 2)
        e2 = torch.mean(fluxs) ** 2
        si = e1 / e2 - 1.0  # average SI over all steps

        Pa = torch.mean(Pas)  # average PIB over all steps

        SNR = 1 / torch.sqrt(dualeB / Pa**2 + si)  # average SNR over all steps
        SNRdB = 10 * (torch.log(SNR) / np.log(10))
        print('SNR(liner)', SNR.item(), 'SNR(dB): ', SNRdB.item(), 'Pa: ', Pa.item(), 'SI: ', si.item())


print(time.time() - t0)
now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("finished training at: ", now.strftime("%Y-%m-%d-%H-%M"))

