import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from scipy import stats

##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
fitness_goal = 15002
Amplitudes = np.concatenate([[0], np.arange(0.1, 15, 0.2)])
nepisodes = 1000
############################################################################
Aindx = 0 ### analysis noiseless environment
Ampl = Amplitudes[Aindx] 

colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
############################################################################
with open('evaluate_hybrids/to_hybrid_evaluation.pkl', 'rb') as f:
    data = pickle.load(f)
############################################################################
def open_hybrid(alpha, evo):
    with open('evaluate_hybrids/data/rep_%.2f_%d.pkl'%(alpha, evo), 'rb') as f:
        this_eval = pickle.load(f)
    return this_eval['fitness_fractional'], this_eval['fitness_mixed']
############################################################################
alphas = [0.8, 0.65, 0.5, 0.35]

perf_data = {}
group, value, hybrid, group_delta, delta = [], [], [], [], []
for aindx, alpha in enumerate(alphas):
    nsel = data[alpha]['nselected']
    fit0 = np.zeros(nsel)
    fitH = np.zeros(nsel)
    for evo in np.arange(nsel):
        ##########
        fit_frac, fit_mix = open_hybrid(alpha, evo)
        group.extend([alpha])
        f1 = np.mean(fit_frac)
        value.extend([f1])
        hybrid.extend(['Normal'])
        #####
        f2 = np.mean(fit_mix)
        group.extend([alpha])
        value.extend([f2])
        hybrid.extend(['Mixed'])
        group_delta.extend([alpha])
        delta.extend([f2 - f1])
        fit0[evo], fitH[evo] = f1, f2
    perf_data[alpha] = [fit0, fitH]
        #############
############################################################################
df = pd.DataFrame({'Network': hybrid, 'Group': group, 'Fit': value})
############################################################################
plt.close('all')
fig = plt.figure(figsize=[11.4, 5.3])
gm = gridspec.GridSpec(230, 170)

axes = [_ for _ in range(5)]
axes[0] = plt.subplot(gm[:70, :30])
for i in range(1, 5):
    axes[i] = plt.subplot(gm[:70, i*33 + 8:i*33+30+ 8])
ax1 = plt.subplot(gm[106:, :90])
ax2 = plt.subplot(gm[106:, 110:])

for p in range(len(df)//2):
    i, j = int(2*p), int(2*p+1)
    alpha = group[i]
    f1, f2 = value[i], value[j]
    k = alphas.index(alpha)
    x1, x2 = k-0.21, k+0.21
    ax1.plot([x1, x2], [f1, f2], '--', color = colors[k], lw = 0.1)
    ax1.plot([x1], [f1], 'd', color = colors[k], markerfacecolor='w', ms = 5)
    ax1.plot([x2], [f2], 'o', color = colors[k], markerfacecolor='w', ms = 5)
    ax1.plot([x1, x2], [320*50, 320*50], 'k', lw = 1.3)
    ax1.text(x1, 310*50, 'F', fontsize = 9, ha = 'center', va = 'center')
    ax1.text(x2, 310*50, 'M', fontsize = 9, ha = 'center', va = 'center')


#### Statistical significance hybrid networks
for aindx, alpha in enumerate(alphas):
    Fract, Hybrid = perf_data[alpha]
    t_stat, p_value = stats.ttest_rel(Hybrid, Fract, alternative='greater')
    if p_value <= 0.0001:
        ax1.text(aindx, 325*50,  "****", fontsize = 9, ha = 'center', va = 'center')
    elif p_value <= 0.001:
        ax1.text(aindx, 325*50,  "***", fontsize = 9, ha = 'center', va = 'center')
    elif p_value <= 0.01:
        ax1.text(aindx, 325*50,  "**", fontsize = 9, ha = 'center', va = 'center')
    elif p_value <= 0.05:
        ax1.text(aindx, 325*50,  "*", fontsize = 9, ha = 'center', va = 'center')
    print('\n\n%.2f'%alpha)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.5f}")

fz = 14

ax1.set_ylim(6000, 16250)
ax1.set_xticks([0, 1, 2, 3], ['0.8', '0.65', '0.5', '0.35'])
ax1.set_yticks([7500, 10000, 12500, 15000], ['150', '200', '250', '300'])
ax1.set_xlabel('Fractional Order', fontsize = fz)
ax1.set_ylabel('Performance (s)', fontsize = fz)

############################################################################
df_delta = pd.DataFrame({'Group': group_delta, 'Fit': delta})
############################################################################
#################
categories = df_delta['Group'].unique()
# Let Seaborn handle the spacing natively!
sns.boxplot(
    data=df_delta, 
    x='Group', 
    y='Fit', 
    ax=ax2,
    order=categories,
    showfliers=False,      # Use this instead of fliersize=0 for cleaner line management
    palette=colors, # Base white palette
    linewidth=0.5, 
    width=0.85,
    gap=0.25             # <--- UNCOMMENT this if you are using Seaborn 0.13.0 or newer!
)

ax2.set_ylim(-250, 5020)
ax2.set_yticks([0, 1500, 3000, 4500], ['0', '30', '60', '90'])
ax2.set_xlabel('Fractional Order', fontsize = fz)
ax2.set_ylabel(r'$\Delta$' + ' Performance (s)', fontsize = fz)

############################################################################
######  Failing cumulative probability function
############################################################################
cum_prob_one = np.arange(1, nepisodes+1)/nepisodes
lenx = 5000
x_common = np.linspace(0, 15002, lenx)
def cpdf(fit):
    return np.interp(x_common, np.sort(fit), cum_prob_one)
############################################################################
alphas = [1.0, 0.8, 0.65, 0.5, 0.35]
alpha_label = ['1', '0.8', '0.65', '0.5', '0.35']
colors = ['#000000', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
fz = 13

for aindx, alpha in enumerate([1., 0.8, 0.65, 0.5, 0.35]):
    ax = axes[aindx]
    nsel = data[alpha]['nselected']
    if alpha == 1:
        cumfail = np.zeros((nsel, lenx))
        for evo in np.arange(nsel):
            fitness = data[alpha][evo]['fitness']
            cumfail[evo] = cpdf(fitness)
        #####
        m, std = np.mean(cumfail, axis = 0), np.std(cumfail, axis = 0)/np.sqrt(nsel)
        ax.plot(x_common/50, m, color = colors[aindx], label = 'Fractional')
        ax.fill_between(x_common/50, m-std, m+std, color = colors[aindx], alpha = 0.1)
    else:
        cumfail_frac, cumfail_mix = np.zeros((nsel, lenx)), np.zeros((nsel, lenx))
        #####
        for evo in np.arange(nsel):
            fit_frac, fit_mix = open_hybrid(alpha, evo)
            cumfail_frac[evo] = cpdf(fit_frac)
            cumfail_mix[evo] = cpdf(fit_mix)
    #####
        ###############################
        m, std = np.mean(cumfail_frac, axis = 0), np.std(cumfail_frac, axis = 0)/np.sqrt(nsel)
        ax.plot(x_common/50, m, color = colors[aindx], label = 'Fractional')
        ax.fill_between(x_common/50, m-std, m+std, color = colors[aindx], alpha = 0.1)
        ###############################
        m, std = np.mean(cumfail_mix, axis = 0), np.std(cumfail_mix, axis = 0)/np.sqrt(nsel)
        ax.plot(x_common/50, m, '--', color = colors[aindx], label = 'Mixed')
        ax.fill_between(x_common/50, m-std, m+std, color = colors[aindx], alpha = 0.1)
        ###############################
        ax.text(0.5, 1.1, r'$\alpha$' + ' = %s'%alpha_label[aindx], fontsize = fz, ha = 'center', transform=ax.transAxes)

    ax.set_xticks([60, 120, 180, 240, 300])
    ax.set_xlabel('Time (s)', fontsize = fz)
    if aindx in [0, 1]:
        ax.set_ylabel('Episode ending\nCPDF', fontsize = fz)
    else:
        # ax.legend(loc='upper left', fontsize = 9, frameon=False)
        ax.legend(loc=(0.05, 0.75), fontsize = 9, frameon=False)
        ax.set_yticklabels([])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 302)

for i in range(1,5):
    axes[i].set_yticklabels([])

############################################################################
#################################################################
############################################################################
for ax in axes + [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
############################################################################
x1, x2, x3 = 0.01, 0.25, 0.58
y1, y2 = 0.97, 0.56
fz = 18
plt.figtext(x1, y1, 'A1', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y1, 'A2', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x1, y2, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x3, y2, 'C', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
##################################################################
############################################################################
fig.subplots_adjust(left = 0.08, bottom = 0.1, right = 0.98, top = 0.94)
plt.savefig('figures/figure_hybrids.png', dpi = 600, transparent = False)
plt.savefig('figures/figure_hybrids.pdf', transparent = False)
plt.savefig('../Figures/figure_6.pdf', transparent = False)

