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
##################################################################
nevolutions = 40
nepochs_fractional = 150
nepochs_non_fractional = 600
fitness_goal = 15002
############################################################################
############################################################################
#### Post-evolution evaluation noisy environment: "wind" 
############################################################################
alphas = [0.95, 0.8, 0.65, 0.5, 0.35, 0.2]
task_goal = 6000
colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

###########################################################################
############################################################################
#### Post-evolution 1000 episodes fitness
############################################################################

with open('non_fractional/selected/evaluation_1000.pkl', 'rb') as f:
    data_non_fractional = pickle.load(f)

with open('fractional/selected/evaluation_1000.pkl', 'rb') as f:
    data = pickle.load(f)

data[1.0] = data_non_fractional

alphas = [1.0, 0.95, 0.8, 0.65, 0.5, 0.35, 0.2]

# Create a Pandas DataFrame for Seaborn
group_labels, B, fitness = [], [], []

for a, alpha in enumerate(alphas):
    ###########
    for indx in np.arange(nevolutions):
        group_labels.extend(['%.2f'%alpha])
        #############
        ind = data[alpha][indx]
        #############
        fitness.extend([np.mean(ind['fitness'])])
        B.extend([ind['indv'][0]])


# Create a dictionary to store mean fitness per individual
mean_performance = {} 
for a, alpha in enumerate(alphas):
    mean_performance[alpha] = np.array([np.mean(data[alpha][evo]['fitness']) for evo in range(nevolutions)])
with open('data_figures/evaluation_1000.pkl', 'wb') as f:
    pickle.dump(mean_performance, f)
############################################################################
############################################################################
plt .close('all')
fig = plt.figure(figsize = [7.3, 3.8])
gm = gridspec.GridSpec(170, 170)
ax = plt.subplot(gm[:,:])
ax.axhline(y = task_goal, linestyle = '--', color = 'k')

df = pd.DataFrame({'Value': fitness, 'Group': group_labels})
sns.stripplot(x='Group', y='Value', ax = ax, data=df, palette=colors)
sns.violinplot(x='Group', y='Value', ax = ax, data=df, palette=colors, alpha = 0.4, inner = None)

xpos = [1/14, 3/14, 5/14, 7/14, 9/14, 11/14, 13/14]
for a, alpha in enumerate(alphas):
    selected_elements = [[] for i, val in enumerate(fitness) if (val > task_goal)*(group_labels[i] == '%.2f'%alpha)]
    text = '%d/%d'%(len(selected_elements), nevolutions)
    ax.text(xpos[a], 1.03, text, transform=ax.transAxes, ha='center', fontsize=11)


ax.set_ylim(-10, 15010)
fz = 14
ax.set_yticks([0, 3000, 6000, 9000, 12000, 15000], ['0', '60', '120', '180', '240', '300'])
ax.set_xlabel("Fractional order", fontsize = fz)
ax.set_ylabel("1000 trial\nPerformance (s)", fontsize = fz)

fig.subplots_adjust(left = 0.12, bottom = 0.14, right = 0.96, top = 0.93)
plt.savefig('figures/figure_4.png', dpi = 600)
plt.savefig('figures/figure_4.pdf')
plt.savefig('../Figures/figure_4.pdf')
