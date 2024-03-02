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
power = 1000  # normalized power
npcb = 10  # number of coherent modes for averaging
dualeB = 64.91  # Gamma in Eq. (6)

wvl = 0.632e-6
delta = 3e-3  # sampling interval, 1.2e-3 for 0.5km, 2e-3 for 1km, 3e-3 for 2km, 4e-3 for 5km
Cn2 = 1e-15  # refractive index structure const
L0 = 1  # outer scale
l0 = 1e-2  # inner scale
delta_Z = 5e3  # prop distance
pi = 3.1415926535
k = 2*pi/wvl

x1 = torch.linspace(-N*delta/2, (N/2-1)*delta, N, device=device)
xx, yy = torch.meshgrid(x1, x1, indexing='ij')
r2 = xx ** 2 + yy ** 2  # 注意这里是r的平方

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
index = torch.where(torch.sqrt(r2) < 5e-2)
circ_mask[index] = 1.0
circ_mask = torch.tile(circ_mask, dims=[bs, npcb, 1, 1])


# single-step propagation
def propTF(H, Uin):
    U1 = fft.fft2(fft.fftshift(Uin, dim=(-2, -1)), dim=(-2, -1))
    U2 = torch.mul(H, U1)
    Uout = fft.ifftshift(fft.ifft2(U2, dim=(-2, -1)), dim=(-2, -1))
    return Uout


base = torch.load("HGbasis.pth")  # HG eigenmodes
# OSB synthesis
def mixfield3(pred, phases):
    eigens = pred  # [bs, npcb, 64]
    U0 = torch.complex(torch.zeros(bs, npcb, N, N, device=device), torch.zeros(bs, npcb, N, N, device=device))
    for i in range(bs):
        eigenv = torch.unsqueeze(eigens[i], 2)
        eigenv = torch.unsqueeze(eigenv, 3)
        graph = base * eigenv
        phase = torch.exp(torch.complex(torch.zeros_like(phases[i]), phases[i]))
        complex_modes = graph * phase.unsqueeze(2).unsqueeze(3)
        U0[i] = torch.sum(complex_modes, dim=1)
    return U0


def save_model(logdir, net, step):
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    torch.save({'net': net.state_dict()},
               os.path.join(logdir, now.strftime("%Y-%m-%d-%H-%M") + "_steps" + str(step) + '.pt'))


net = EfficientNetV2('s', in_channels=1, n_classes=128, in_spatial_shape=256, npcb=npcb).to(device)
optim_G = torch.optim.Adam(net.parameters(), lr=1e-3)

logdir = 'logs/osb_xxxx'
if not os.path.exists(logdir):
    os.mkdir(logdir)

total_steps = 20000
writer = SummaryWriter(logdir)
def main():
    net.train()
    with trange(0, total_steps, desc='Training', ncols=0) as pbar:
        aversi = 0.0
        averPa = 0.0
        averSNR = 0.0
        input1 = 0.1 * torch.ones(bs, 1, 256, 256, device=device)  # placeholder
        for step in pbar:

            output = net(input1)
            eigenvalues, sigma_s, phases = output[:, :, :64], output[:, :, 64], output[:, :, 65:]

            eigenvalues = F.relu(eigenvalues)
            phases = torch.cat((phases, torch.zeros(bs, npcb, 1, device=device)), dim=2)
            phases = F.tanh(phases) * pi

            U0 = mixfield3(eigenvalues, phases)  # beam synthesis

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

            I = (U.abs() * circ_mask) ** 2  # apply aperture bucket
            flux = torch.sum(I, dim=[1, 2, 3]) / npcb
            e1 = torch.mean(flux ** 2)
            e2 = torch.mean(flux) ** 2
            si = e1 / e2 - 1.0  # scintillation index

            Pa = torch.sum(flux) / bs  # power in the bucket

            SNR_1 = torch.sqrt(dualeB / Pa ** 2 + si)  # inverse of SNR (linear)

            SNRdB = 10 * (torch.log(1 / SNR_1) / np.log(10))  # SNR (dB)

            loss = SNR_1

            aversi = aversi + si
            averPa = averPa + Pa
            averSNR = averSNR + SNRdB

            if step % 10 == 0 and step > 0:
                aversi = aversi / 10
                averPa = averPa / 10
                averSNR = averSNR / 10
                writer.add_scalar('SNR', averSNR, step)
                writer.add_scalar('si', aversi, step)
                writer.add_scalar('Pa', averPa, step)

                aversi = 0.0
                averPa = 0.0
                averSNR = 0.0

            if step % 10000 == 0:  # save model
                if step != 0:
                    save_model(logdir, net, step)

            optim_G.zero_grad()

            loss.backward()

            pbar.set_postfix({'SNR (dB)': '{0:1.4f}'.format(SNRdB), 'SI': '{0:1.4f}'.format(si),
                              'Aver Power': '{0:1.4f}'.format(Pa)})

            optim_G.step()
            gc.collect()


    writer.close()
    save_model(logdir, net, step+1)

if __name__ == '__main__':
    main()


print(time.time() - t0)
now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("finished training at: ", now.strftime("%Y-%m-%d-%H-%M"))

