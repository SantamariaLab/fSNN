import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.gridspec as gridspec
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
plt .close('all')
fig = plt.figure(figsize = [8.8, 3.])
gm = gridspec.GridSpec(170, 270)
axes = [plt.subplot(gm[:, i*100:i*100 + 70]) for i in range(3)]
##################################################################
C = 0.00055
dt = 0.0005
colors = ['k', 'g', 'r']
freqs = [0.1, 0.5, 2]
alphas = [1, 0.5, 0.2]

##################################################################
l = 5
b = 1.4
def plot_arrow(v, q, pos, ax, color):
    dx, dy = v[pos+1] - v[pos], q[pos+1] - q[pos]
    sx, sy = np.sign(dx), np.sign(dy)
    x, y = v[pos], q[pos] - b * sx

    f = dx**2 + dy**2
    xt, yt = x + sx * l * (dx**2/f)**0.5, y + sy * l * (dy**2/f)**0.5

    ax.annotate('',xy=(x, y), xytext=(xt, yt), arrowprops=dict(arrowstyle="->",color=color,lw=1.2))
##################################################################
displ = [0.25, 0.05, 0]
for a, alpha in  enumerate(alphas):
    for f, freq in enumerate(freqs):
        ttot = 1./freq
        time = np.arange(0, ttot, dt)
        w = 2 * np.pi * freq

        A = 20
        Vapp = A * np.cos(w*time)
        qth = C * A * w**(alpha - 1) * np.sin(w*time + alpha * np.pi /2.)

        axes[f].plot(Vapp, 1000*qth, color = colors[a], label = r'$\alpha$' + ' = %.1f'%alpha)
        for pos in [int((1+displ[a])*len(time)//4), int((3+displ[a])*len(time)//4)]:
            plot_arrow(Vapp, 1000*qth, pos, axes[f], colors[a])
##################################################################
for a, freq in  enumerate(freqs):
    axes[a].text(0.5, 19, '%.1f Hz'%freq, fontsize = 14, ha = 'center')

    axes[a].set_ylim(-18, 18)
    axes[a].set_xlim(-22, 22)

    axes[a].set_ylabel('Charge ('+r'$\mu$' +'C)', fontsize = 13)
    axes[a].set_xlabel('Voltage (V)', fontsize = 13)
axes[2].legend()


##################################################################
x1, x2, x3 = 0.015, 0.35, 0.68
y, fz = 0.94, 16
plt.figtext(x1, y, 'A', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x3, y, 'C', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')

##################################################################
fig.subplots_adjust(left = 0.08, bottom = 0.16, right = 0.98, top = 0.86)
plt.savefig('./fig_1.png', dpi = 300, transparent = False)
plt.savefig('../figure_1.pdf')

plt.savefig('../Figures/figure_1.pdf')



##################################################################
##### Same idea, but with numerical integration instead of close solution
##### This works with non sinusoidal inputs, in this case a sawthooth signal
##################################################################


# class FractionalMemcapacitor:
#     """
#     dQ^(1-alpha)/ dt^(1-alpha) = C V
#     """
#     def __init__(self,
#                  Capacitance=0.0001,
#                  fractional_order=0.9, memory_length=120,
#                  dt = 0.0001):
#         # State variable
#         self.charge = 0
#         # Capacitor parameters
#         self.Capacitance = Capacitance
#         # Fractional order parameters (1- alpha) 
#         self.fractional_order = (1-fractional_order) # fractional_order#
#         self.memory_length = memory_length
#         self.dt = dt
#         self.kernel = self._compute_kernel(dt, self.fractional_order)
#         # Grunwald-Letnikov coefficients for fractional derivative
#         self.gl_coefficients = self._compute_gl_coefficients(self.fractional_order, memory_length)
#         # History arrays to store past charge values for the fractional derivative
#         self.charge_history = np.zeros(memory_length, dtype=np.float64)
#         # Components for the history term in the integration step
#         self.history_component = 0

#     def _compute_gl_coefficients(self, alpha, length):
#         """
#         Compute Grunwald-Letnikov coefficients for the fractional derivative.
#         """
#         coeffs = np.zeros(length, dtype=np.float64)
#         if length > 0:
#             coeffs[0] = -alpha
#             for j in range(1, length):
#                 coeffs[j] = (1.0 - (alpha + 1.0) / (j + 1)) * coeffs[j - 1]
#         return coeffs

#     def _compute_kernel(self, dt, alpha):
#         return dt ** alpha

#     def reset(self):
#         """
#         Reset all network state variables (charge, spike states, and history) to their initial values.
#         """
#         self.charge = 0
#         self.charge_history.fill(0.0)

#     def update(self, input, dt=0.10):
#         """
#         Update the network state for one time step using fractional order dynamics.
#         """
#         # This is a dot product of GL coefficients with the charge history
#         self.history_component = np.dot(self.gl_coefficients, self.charge_history)
#         # The equation is V_new = (-V_old + bias + input) / tau * kernel - history_component
#         self.charge = self.kernel*self.Capacitance * input - self.history_component
#         #  update charge history
#         self.charge_history = np.roll(self.charge_history, 1, axis=0)
#         self.charge_history[0] = self.charge

# ##################################################################
# plt .close('all')
# fig = plt.figure(figsize = [8.8, 3.])
# gm = gridspec.GridSpec(170, 270)
# axes = [plt.subplot(gm[:, i*100:i*100 + 70]) for i in range(3)]
# ##################################################################
# C = 0.00055
# dt = 0.005
# ncycles = 3

# ##################################################################
# l = 4
# b = 1.4
# def plot_arrow(v, q, pos, ax, color):
#     dx, dy = v[pos+1] - v[pos], q[pos+1] - q[pos]
#     sx, sy = np.sign(dx), np.sign(dy)
#     x, y = v[pos], q[pos] - b * sx

#     f = dx**2 + dy**2
#     xt, yt = x + sx * l * (dx**2/f)**0.5, y + sy * l * (dy**2/f)**0.5

#     ax.annotate('',xy=(x, y), xytext=(xt, yt), arrowprops=dict(arrowstyle="->",color=color,lw=1.2))
# ##################################################################
# displ = [0.25, 0.05, 0]

# for a, alpha in  enumerate(alphas):
#     for f, freq in enumerate(freqs):
#         ttot = ncycles/freq
#         time = np.arange(0, ttot, dt)
#         w = 2 * np.pi * freq
#         cycle_points = int(len(time)/ncycles)
#         memory_length = len(time)

#         A = 20
#         Vapp = A * np.cos(w*time)
#         Vapp = A * signal.sawtooth(w*time, width = 0.5)
#     ##################################################################
#         N = FractionalMemcapacitor(C, alpha, memory_length, dt)
#         N.reset()

#         Q = np.zeros(len(time))
#         print(N.fractional_order)
#         for k, t in enumerate(time):
#             N.update(Vapp[k], dt)
#             Q[k] = N.charge
#     ##################################################################

#         # qth = C * A * w**(alpha - 1) * np.sin(w*time + alpha * np.pi /2.)

#         axes[f].plot(Vapp[-2*cycle_points:], 1000*Q[-2*cycle_points:], color = colors[a], label = r'$\alpha$' + ' = %.1f'%alpha)
#         # axes[f].plot(Vapp, 1000*qth, color = colors[a], label = r'$\alpha$' + ' = %.1f'%alpha)
#         for pos in [int((9+displ[a])*len(time)//12), int((11+displ[a])*len(time)//12)]:
#             plot_arrow(Vapp, 1000*Q, pos, axes[f], colors[a])
# ##################################################################
# for a, freq in  enumerate(freqs):
#     axes[a].text(0.5, 19, '%.1f Hz'%freq, fontsize = 14, ha = 'center')

#     axes[a].set_ylim(-18, 18)
#     axes[a].set_xlim(-22, 22)

#     axes[a].set_ylabel('Charge ('+r'$\mu$' +'C)', fontsize = 13)
#     axes[a].set_xlabel('Voltage (V)', fontsize = 13)
# axes[2].legend()

# ##################################################################
# x1, x2, x3 = 0.015, 0.35, 0.68
# y, fz = 0.94, 22
# plt.figtext(x1, y, 'A', ha = 'left', va = 'center', fontsize = fz)
# plt.figtext(x2, y, 'B', ha = 'left', va = 'center', fontsize = fz)
# plt.figtext(x3, y, 'C', ha = 'left', va = 'center', fontsize = fz)

# ##################################################################
# fig.subplots_adjust(left = 0.08, bottom = 0.16, right = 0.98, top = 0.86)
# plt.savefig('./fig_1_sawthooth.png', dpi = 300, transparent = False)

