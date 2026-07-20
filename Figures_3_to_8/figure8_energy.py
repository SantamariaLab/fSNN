import numpy as np
import pickle
import pandas as pd
##################################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns
##################################################################
#### Load selected networks
with open('./evaluate_hybrids/to_hybrid_evaluation.pkl', 'rb') as f:
    selected = pickle.load(f)

### succesful solutions were evaluated for 400 seconds
ttot = 400 

alphas = [1.0, 0.8, 0.65, 0.5, 0.35]

Group, Rates, Flops = [], [], []
for a, alpha in enumerate(alphas):
    ##############################
    ##############################
    nselected = selected[alpha]['nselected']
    ##############################################
    for nindv in range(nselected):
        with open('dynamic_avalanches/data/spikes_%.2f_%d.pkl'%(alpha, nindv), 'rb') as f:
            simulation = pickle.load(f)
        ###########################
        spikes = simulation['spikes']
        flops = simulation['total_flops']
        ###########################
        Group.append('%.2f'%alpha)
        Rates.append(np.sum(spikes)/ttot)
        Flops.append(flops/ttot/10**6)
        ###########################
############################################################################

df = pd.DataFrame({'Group': Group, 'Rates': Rates, 'Flops': Flops})

############################################################################
############################################################################

plt.close('all')
fig = plt.figure(figsize=[5.25, 6.8])
gm = gridspec.GridSpec(160, 180)
ax1 = plt.subplot(gm[:60, :])
ax2 = plt.subplot(gm[80:, :])

############################################################################
fz = 13
alpha_label = ['1', '0.8', '0.65', '0.5', '0.35']
colors = ['#000000', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

############################################################################
categories = df['Group'].unique()

############################################################################
sns.boxplot(data=df, x='Group', y='Flops', ax=ax1, order=categories, palette=colors, linewidth=0.5, width=0.85, gap=0.25, showfliers=False)
ax1.plot(np.arange(len(alphas)), list(df.groupby('Group')['Flops'].mean())[::-1], 'ok')
sns.boxplot(data=df, x='Group', y='Rates', ax=ax2, order=categories, palette=colors, linewidth=0.5, width=0.85, gap=0.25, showfliers=False)
############################################################################

for ax in [ax1, ax2]:
    ax.set_xlabel("Fractional order", fontsize = fz)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(alpha_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.set_ylabel('MFLOPS/second', fontsize = fz)
ax1.set_ylim(0, 1)
ax2.set_ylabel('Spikes/second', fontsize = fz)
############################################################################
############################################################################
############################################################################
y1, y2 = 0.96, 0.52
x1 = 0.01
fz = 20
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x1, y2, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
##################################################################
fig.subplots_adjust(left = 0.14, bottom = 0.08, right = 0.96, top = 0.93)
plt.savefig('figures/figure_8_energy.png', dpi = 300)
plt.savefig('figures/figure_8_energy.pdf')
plt.savefig('../Figures/figure_8.pdf')