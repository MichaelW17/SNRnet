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
power = 1000  # normalized power
npcb = 10  # number of coherent modes for averaging
dualeB = 64.91  # Gamma in Eq. (6)

wvl = 0.632e-6  # wavelength
delta = 1.2e-3  # sampling interval
Cn2 = 2e-13  # refractive index structure constant
L0 = 1  # outer scale
l0 = 1e-2  # inner scale
delta_Z = 0.5e3  # prop distance
pi = 3.1415926535
k = 2*pi/wvl

x1 = torch.linspace(-N*delta/2, (N/2-1)*delta, N, device=device)
xx, yy = torch.meshgrid(x1, x1, indexing='ij')
r2 = xx ** 2 + yy ** 2

fx = torch.linspace(-0.5/delta, 0.5/delta-1/(N*delta), N, device=device)
FX, FY = torch.meshgrid(fx, fx, indexing='ij')
f = torch.sqrt(FX**2 + FY**2)

fm = 5.92/l0/(2*pi)
f0 = 1/L0
phase_PSD = 2*pi * k**2 * delta_Z * 0.033 * Cn2 * (2*pi)**(-11/3) * (4*pi**2) * \
            torch.exp(-(f/fm)**2) / (f**2+f0**2)**(11/6)  # 引入内外尺度
phase_PSD[128, 128] = 0
phase_PSD = torch.torch.sqrt(phase_PSD) / (N*delta)
phasePSD = torch.tile(phase_PSD, dims=[bs, 1, 1])


# transfer matrix in Fresnel diffraction
H = torch.exp(torch.complex(torch.zeros_like(FX), -pi*wvl*delta_Z*(FX**2+FY**2)))
H = fft.fftshift(H)
H = torch.tile(H, dims=[bs, npcb, 1, 1])

# Rx aperture
circ_mask = torch.zeros(N, N, device=device)
index = torch.where(torch.sqrt(r2) < 1.5e-2)
circ_mask[index] = 1.0
circ_mask = torch.tile(circ_mask, dims=[bs, npcb, 1, 1])


# single-step propagation
def propTF(H, Uin):
    U1 = fft.fft2(fft.fftshift(Uin, dim=(-2, -1)), dim=(-2, -1))
    U2 = torch.mul(H, U1)
    Uout = fft.ifftshift(fft.ifft2(U2, dim=(-2, -1)), dim=(-2, -1))
    return Uout


# PCB synthesis
fie_real = torch.randn(npcb, N, N)
fie_img = torch.randn(npcb, N, N)
fie_rander = torch.complex(fie_real, fie_img).to(device)
def pcbsyn(w0, Lcr):
    U0 = torch.complex(torch.zeros(bs, npcb, N, N, device=device), torch.zeros(bs, npcb, N, N, device=device))

    for i in range(bs):
        # w = 2e-2
        w = w0[i]
        u0 = torch.exp(-r2 / (w**2))  # amplitude
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


def save_model(logdir, net, step):
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    torch.save({'net': net.state_dict()},
               os.path.join(logdir, now.strftime("%Y-%m-%d-%H-%M") + "_steps" + str(step) + '.pt'))


net = EfficientNetV2('s', in_channels=1, n_classes=2, in_spatial_shape=256).to(device)
optim_G = torch.optim.SGD(net.parameters(), lr=1e-3)


logdir = 'logs/pcb_xxxx'
if not os.path.exists(logdir):
    os.mkdir(logdir)

total_steps = 20000
writer = SummaryWriter(logdir)

max_w0 = 0.1
max_Lcr = 0.5
def main():
    net.train()
    with trange(0, total_steps, desc='Training', ncols=0) as pbar:
        aversi = 0.0
        averPa = 0.0
        averSNR = 0.0
        input1 = 0.1 * torch.ones(bs, 1, 256, 256, device=device)  # placeholder
        for step in pbar:

            output = net(input1)
            w0, Lcr = output[:, 0], output[:, 1]

            w0 = max_w0 * F.sigmoid(w0) + 0.005

            Lcr = max_Lcr * F.sigmoid(Lcr) + 0.001

            U0 = pcbsyn(w0, Lcr)  # beam generation

            # power normalization
            amp = U0.abs()
            output_power = torch.sum(amp ** 2, dim=[-2, -1])
            power_ratio = torch.sqrt(power / output_power)
            U0 = torch.mul(U0, power_ratio.unsqueeze(2).unsqueeze(3))

            # dynamic phase screen generation
            rand_real = torch.randn(bs, N, N, dtype=torch.float)
            rand_img = torch.randn(bs, N, N, dtype=torch.float)
            rander = torch.complex(rand_real, rand_img).to(device)
            cn = torch.mul(rander, phasePSD)
            screen0 = torch.real(fft.ifftshift(fft.ifft2(fft.ifftshift(cn, dim=(-2, -1)), dim=(-2, -1)),
                                              dim=(-2, -1)) * N ** 2)

            screen0 = torch.unsqueeze(screen0, dim=1)
            screen0 = torch.tile(screen0, dims=[1, npcb, 1, 1])

            # screen imposition
            Uin = U0 * torch.exp(torch.complex(torch.zeros_like(amp), screen0))

            # free-space propagation
            U = propTF(H, Uin)

            # plt.figure(num=1, figsize=(5, 5))
            # plt.imshow(amp[0].detach().cpu().abs().numpy(), cmap='gray')
            # plt.show(block=True)

            I = (U.abs() * circ_mask) ** 2  # apply aperture bucket
            flux = torch.sum(I, dim=[1, 2, 3]) / npcb
            e1 = torch.mean(flux ** 2)
            e2 = torch.mean(flux) ** 2
            si = e1 / e2 - 1.0  # SI

            Pa = torch.sum(flux) / bs  # PIB


            SNR_1 = torch.sqrt(dualeB / Pa**2 + si)  # inverse of SNR

            SNRdB = 10 * (torch.log(1 / SNR_1) / np.log(10))

            loss = SNR_1

            aversi = aversi + si
            averPa = averPa + Pa
            averSNR = averSNR + SNRdB

            w0i = torch.mean(w0)
            Lcri = torch.mean(Lcr)
            if step % 10 == 0 and step > 0:  # 每10步保存loss均值
                aversi = aversi / 10
                averPa = averPa / 10
                averSNR = averSNR / 10
                writer.add_scalar('SNR', averSNR, step)
                writer.add_scalar('si', aversi, step)
                writer.add_scalar('Pa', averPa, step)
                writer.add_scalar('w0', w0i, step)
                writer.add_scalar('Lcr', Lcri, step)

                aversi = 0.0
                averPa = 0.0
                averSNR = 0.0

            if step % 10000 == 0:  # 每10000步保存模型
                if step != 0:
                    save_model(logdir, net, step)

            optim_G.zero_grad()

            loss.backward()

            pbar.set_postfix({'SNR (dB)': '{0:1.4f}'.format(SNRdB), 'SI': '{0:1.4f}'.format(si),
                              'Aver Power': '{0:1.4f}'.format(Pa), 'w0': '{0:1.4f}'.format(w0i),
                              'Lcr': '{0:1.4f}'.format(Lcri)})

            optim_G.step()

            gc.collect()  # 垃圾回收

    writer.close()
    save_model(logdir, net, step+1)


if __name__ == '__main__':
    main()



print(time.time() - t0)
now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("finished training at: ", now.strftime("%Y-%m-%d-%H-%M"))

