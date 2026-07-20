import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from utils.fSNN import *
from utils.cartpole_depiction import *
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
##################################################################

plt .close('all')
fig = plt.figure(figsize = [8.2, 7.1])

gm = gridspec.GridSpec(280, 170)

ax_cp = plt.subplot(gm[:70,5:75])

ax_fsnn = plt.subplot(gm[70:175,:80])
ax_fsnn_fit = plt.subplot(gm[70:150, 95:])

ax_snn = plt.subplot(gm[175:,:80])
ax_snn_fit = plt.subplot(gm[180:260, 95:])

############################################################################
############################################################################
plot_cartpole(ax_cp)
plot_fSNN(ax_fsnn)
plot_SNN(ax_snn)
############################################################################
############################################################################

nevolutions = 40

alphas = [1.0, 0.95, 0.8, 0.65, 0.5, 0.35, 0.2]

with open('data_figures/evolutions.pkl', 'rb') as f:
    evolutions = pickle.load(f)



task_goal = 6000
def plot_fitness(ax, alpha, f, color):
    nepochs = len(f[0])
    mf, stdf = np.mean(f, axis = 0), np.std(f, axis = 0)/np.sqrt(nevolutions)
    ax.plot(np.arange(nepochs), mf, color = color, label = r'$\alpha$' + ' = %.2f'%alpha)
    ax.fill_between(np.arange(nepochs), mf - stdf, mf, color = color, alpha = 0.2)
    return f

###########################################################################
### Fractional
############################################################################

# ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

for a, alpha in enumerate(alphas[1:]):
    plot_fitness(ax_fsnn_fit, alpha, evolutions[alpha], colors[a+1])

ax_fsnn_fit.axhline(y = task_goal, linestyle = '--', color = 'k')


legend = ax_fsnn_fit.legend(loc='upper left', bbox_to_anchor=(.05, 1.25), ncol = 3)
legend.get_frame().set_alpha(0.0)

fig.subplots_adjust(left = 0.14, bottom = 0.16, right = 0.95, top = 0.95)

ax_fsnn_fit.set_xlim(-.1, 151)
ax_fsnn_fit.set_ylim(-10, 15010)
fz = 14
ax_fsnn_fit.set_xticks([50, 100, 150])
ax_fsnn_fit.set_yticks([0, 3000, 6000, 9000, 12000, 15000], ['0', '60', '120', '180', '240', '300'])
ax_fsnn_fit.set_xlabel('Epoch', fontsize = fz)
ax_fsnn_fit.set_ylabel('Fitness (s)', fontsize = fz)


###########################################################################
### Non Fractional
############################################################################

for a, alpha in enumerate(alphas[:1]):
    plot_fitness(ax_snn_fit, alpha, evolutions[alpha], 'k')

ax_snn_fit.axhline(y = task_goal, linestyle = '--', color = 'k')

legend = ax_snn_fit.legend(loc = 1)
legend.get_frame().set_alpha(0.0)

fig.subplots_adjust(left = 0.14, bottom = 0.16, right = 0.95, top = 0.95)

ax_snn_fit.set_xlim(-.1, 601)
ax_snn_fit.set_ylim(-10, 15010)
fz = 14
# ax_snn_fit.set_xticks([50, 100, 150, 200, 250])
ax_snn_fit.set_yticks([0, 3000, 6000, 9000, 12000, 15000], ['0', '60', '120', '180', '240', '300'])
ax_snn_fit.set_xlabel('Epoch', fontsize = fz)
ax_snn_fit.set_ylabel('Fitness (s)', fontsize = fz)


#################################################################
x1, x2 = 0.01, 0.45
y1, y2, y3 = 0.97, 0.74, 0.34
fz = 20
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x1, y2, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y2, 'C', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x1, y3, 'D', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y3, 'E', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
##################################################################


fig.subplots_adjust(left = 0, bottom = 0., right = 0.98, top = 1)
plt.savefig('figures/figure_3.png', dpi = 600, transparent = True)
plt.savefig('figures/figure_3.pdf')
plt.savefig('../Figures/figure_3.pdf')
