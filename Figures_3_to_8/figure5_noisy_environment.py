import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
##### function to fit a sigmoid to performance to noise curve
##################################################################
from scipy.optimize import curve_fit
def sigmoid_func(x, s, x_half):
    """
    Model function: 1 / (1 + exp(s * (x - x_half)))
    """
    return 1 / (1 + np.exp(-s * (x - x_half)))
##################################################################
##################################################################
nevolutions = 40
nepochs_fractional = 150
nepochs_non_fractional = 600
fitness_goal = 15002
############################################################################
############################################################################
#### Post-evolution evaluation noisy environment: "wind" 
############################################################################
data = {}
task_goal = 6000
colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
###########################################################################
####################################
#######  Load perturbated environment data fractional
####################################
alphas = [0.95, 0.8, 0.65, 0.5, 0.35, 0.2]
with open('non_fractional/selected/evaluation_perturbation.pkl', 'rb') as f:
    data[1.0] = pickle.load(f)

for alpha in alphas:
    with open('fractional/selected/evaluate_perturbation_%.1f.pkl'%alpha, 'rb') as f:
        data[alpha] = pickle.load(f)
####################################
#######  Load perturbated environment data non fractional
####################################
with open('non_fractional/selected/evaluation_1000.pkl', 'rb') as f:
    evaluation_non_fractional = pickle.load(f)

data[1.0][0] = {}
for evo in range(nevolutions):
    data[1.0][0][evo] = evaluation_non_fractional[evo]['fitness']
####################################
#######  Load non perturbated environment data
####################################
with open('fractional/selected/evaluation_1000.pkl', 'rb') as f:
    data_fractional = pickle.load(f)

for alpha in alphas:
    data[alpha][0] = {}
    for evo in range(nevolutions):
        data[alpha][0][evo] = data_fractional[alpha][evo]['fitness']
####################################
#######  Aggregate data
####################################
alphas = [1.0, 0.95, 0.8, 0.65, 0.5, 0.35, 0.2]

Amplitudes = np.arange(0.1, 15, 0.2)
Amplitudes = np.concatenate([[0], Amplitudes])

data_mean = {}
for a, alpha in enumerate(alphas):
    data_mean[alpha] = np.zeros((nevolutions, len(Amplitudes)))
    for evo in range(nevolutions):
        for a, Ampl in enumerate(Amplitudes):
            data_mean[alpha][evo, a] = np.mean(data[alpha][Ampl][evo])

####################################
group, max_fit = [], []
slope, Amp_half  = [], []

for a, alpha in enumerate(alphas):
    this_alpha = data_mean[alpha]
    selected = np.where((this_alpha[:,0] > 6000)) ## 300 seconds
    for sel in selected[0]: 
        fit = this_alpha[sel]
        group.extend(['%.2f'%alpha])
        max_fit.extend([fit[0]])
        popt, pcov = curve_fit(sigmoid_func, Amplitudes, fit/fit[0], p0=[2.0, 2.0])
        slope.extend([popt[0]])
        Amp_half.extend([popt[1]])

df = pd.DataFrame({'Max fit': max_fit, 'Group': group, 'slope': slope, 'Ahalf':Amp_half})
############################################################################
####################################
#######  Figure
####################################
plt .close('all')
fig = plt.figure(figsize = [12.3, 3.4])
gm = gridspec.GridSpec(170, 245)
ax1 = plt.subplot(gm[:,:95])
ax2 = plt.subplot(gm[:,110:170])
ax3 = plt.subplot(gm[:,185:245])
        

for a, alpha in enumerate(alphas):
    this_alpha = data_mean[alpha]
    selected = np.where((this_alpha[:,0] > 6000) * (this_alpha[:,0] < 15300))
    mfit, stdfit = np.mean(this_alpha[selected], axis = 0), np.std(this_alpha[selected], axis = 0)
    ax1.plot(Amplitudes, mfit/mfit[0], color = colors[a], label = r'$\alpha$' + ' = %.2f'%alpha)

############################################################################
############################################################################
# ax.set_xscale('log')
# ax.set_yscale('log')
ax1.set_xlim(-0.01, 6.2)
ax1.set_ylim(-0.01, 1.01)
ax1.legend()

fz = 14
ax1.set_ylabel("Normalized Performance", fontsize = fz)
ax1.set_xlabel(r"$\sigma$" + " Noise (N)", fontsize = fz)

############################################################################
############################################################################
sns.stripplot(x='Group', y='slope', ax = ax2, data=df, palette=colors)
sns.violinplot(x='Group', y='slope', ax = ax2, data=df, palette=colors, alpha = 0.4, inner = None)

ax2.set_yticks([-1, -2, -3, -4])
ax2.set_xlabel("Fractional order", fontsize = fz)
ax2.set_ylabel("Slope", fontsize = fz)
ax2.set_xticks(np.arange(7))
ax2.set_xticklabels(['1', '0.95', '0.8', '0.65', '0.5', '0.35', '0.2'])


############################################################################
############################################################################
sns.stripplot(x='Group', y='Ahalf', ax = ax3, data=df, palette=colors)
sns.violinplot(x='Group', y='Ahalf', ax = ax3, data=df, palette=colors, alpha = 0.4, inner = None)

ax3.set_yticks([1, 2, 3])
ax3.set_xlabel("Fractional order", fontsize = fz)
ax3.set_ylabel(r"$\sigma$" + " Noise (N)", fontsize = fz)
ax3.set_xticks(np.arange(7))
ax3.set_xticklabels(['1', '0.95', '0.8', '0.65', '0.5', '0.35', '0.2'])

############################################################################
############################################################################

x1, x2, x3 = 0.002, 0.43, 0.715
y1 = 0.95
fz = 22
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x3, y1, 'C', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
############################################################################
# ############################################################################

fig.subplots_adjust(left = 0.05, bottom = 0.14, right = 0.98, top = 0.92)
plt.savefig('figures/figure_5B.png', dpi = 600)
plt.savefig('figures/figure_5.pdf')
plt.savefig('../Figures/figure_5.pdf')
############################################################################
############################################################################
