import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import pickle
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
import os

def plot_arrow(pi, pf, ax, color):
    ax.annotate('',xy=pi, xytext=pf, arrowprops=dict(arrowstyle="->",color=color,lw=1.2))

nampl = 0.1
size_parallel = 70 ## number of simulations in parallel used in hysteresys/run_I_slope.py 
############################################################################
with open('hysteresis/data.pkl', 'rb') as f:
    data = pickle.load(f)

alphas = data['alphas']
Imin = data['Imin']
Imax = data['Imax']
ttot = data['ttot']
slopes = data['slopes']

slopes = [1, 10, 60]

### the periodic input is divided in 56 bins
nbins = 56

Iampl = Imax - Imin
bins_scales = (2 * np.concatenate([np.arange(nbins/2+1), np.arange(nbins/2-1, -1, -1)]))/(nbins)
Ibins = Imin + Iampl * bins_scales

############################################################################

colors = ['k', 'g', 'r']
markers = ['-o', '-s', '-D']

##################################################################
plt .close('all')
fig = plt.figure(figsize = [8.8, 2.6])
gm = gridspec.GridSpec(170, 270)
axes = [plt.subplot(gm[:, i%3*100:i%3*100 + 74]) for i in range(3)]
##################################################################
colors = [['#525252','#252525','#000000'], 
        ['#74c476','#238b45','#00441b'],
        ['#fb6a4a','#cb181d','#67000d']]

#################################################################
for a, alpha in  enumerate(alphas):
    for s, slope in  enumerate(slopes):
        time_one_rank, sp_phase = data[alpha][slope]
        ttot = size_parallel * time_one_rank
        spk_per_bin = np.histogram(sp_phase, bins = nbins +1, range = [-0.5/nbins, 1 + 0.5/nbins])[0]
        spk_per_bin[0] += spk_per_bin[-1]
        spk_per_bin[-1] = spk_per_bin[0]
        rates = spk_per_bin*nbins/ttot 
        ###########################
        axes[a].plot(Ibins, rates, markers[s], lw = 0.5, ms = 1.5, color = colors[a][s], label = '%d nA/s'%slope, alpha = 0.95)

for a, alpha in  enumerate(alphas):
    dl = a * 0.3
    dr = a * 1.26
    plot_arrow((10 + dl, 300), (11 + dl, 400), axes[a], colors[a][1])
    plot_arrow((11.5 + dr, 300), (10.5 + dr, 200), axes[a], colors[a][1])
    axes[a].text(0.5, 1.06, r'$\alpha$' + ' = %.1f'%alpha, transform=axes[a].transAxes, fontsize = 14, ha = 'center')
    axes[a].set_ylabel('Firing Rate (spk/s)', fontsize = 13)
    axes[a].set_xlabel('Iapp (nA)', fontsize = 13) #r'$\sigma$' + 
    axes[a].set_xlim(7.5, 16.3)
    axes[a].set_ylim(-2, 502)
    axes[a].set_xticks([8, 10, 12, 14, 16])

    axes[a].legend(fontsize=8)

##################################################################
x1, x2, x3 = 0.015, 0.35, 0.68
y, fz = 0.94, 16
plt.figtext(x1, y, 'A', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x3, y, 'C', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')

##################################################################
# plt.show()
##################################################################
fig.subplots_adjust(left = 0.08, bottom = 0.2, right = 0.98, top = 0.86)
plt.savefig('./fig_2.png', dpi = 300, transparent = False)
plt.savefig('./figure_2.pdf')
plt.savefig('../Figures/figure_2.pdf')
