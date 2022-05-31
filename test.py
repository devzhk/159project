# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

# %%
nx = 1024
F_Time = np.linspace(0, 1, nx)
u = np.cos(2*np.pi*F_Time)
u = torch.tensor(u)
u_h = torch.fft.fft(u, dim=0)
k_max = nx // 2
k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                 torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(nx)
ux_h = 1j * k_x * u_h
ux_oc = 2j * k_x * np.pi * u_h
ux = torch.fft.irfft(ux_h[:k_max + 1], dim=0, n=nx)
ux_oc = torch.fft.irfft(ux_oc[:k_max + 1], dim=0, n=nx)
line1, = plt.plot(- np.sin(2 * np.pi * F_Time), linestyle='-.',
                  alpha=0.9, label='exact gradient')
line2, = plt.plot(ux_oc, linestyle='--', alpha=0.9, label='My code')
line3, = plt.plot(ux, linestyle='--', alpha=0.9, label='New code')
plt.legend()
plt.show()
# %%
