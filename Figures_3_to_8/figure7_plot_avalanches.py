import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import pandas as pd

from scipy import stats
import statsmodels.api as sm
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
##################################################################
def loglog_linear_fit(x, y):
    mask = y > 10**-4
    x_filtered = x[mask]
    y_filtered = y[mask]
    log_x = np.log10(x_filtered)
    log_y = np.log10(y_filtered)
    # Linear regression in log-log space
    return stats.linregress(log_x, log_y)
##################################################################
def loglog_linear_fit_model(x, y):
    mask = y > 10**-4
    x_filtered = x[mask]
    y_filtered = y[mask]
    log_x = np.log(x_filtered)
    log_y = np.log(y_filtered)
    model = sm.OLS(log_y- log_y[0], log_x- log_x[0])
    results = model.fit()


    slope = results.params[0]
    intercept = -log_x[0]*slope
    std_err = results.bse[0]     # Standard error of the slope
    p_value = results.pvalues[0]   # P-value of the slope
    # Calculate the Sum of Squared Residuals R2
    y_pred = results.predict(log_x)
    ssr = np.sum((log_y - y_pred)**2)
    sst = np.sum((log_y - np.mean(log_y))**2)
    r2 = 1 - (ssr / sst)
    return slope, intercept, r2, p_value, std_err
##################################################################
#### Load selected networks
with open('evaluate_hybrids/to_hybrid_evaluation.pkl', 'rb') as f:
    selected = pickle.load(f)
###########################################################################
colors = ['k', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
############################################################################
##################################
n_bins = 50
def get_ccdf(events, n_bins):
    events = np.asarray(events)
    min_val = events[events > 0].min()
    max_val = events.max()
    # Create logarithmically spaced bin edges
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)

    # Count events in each bin and compute bin centers
    counts, _ = np.histogram(events, bins=bin_edges)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean as bin center
    # Normalize by bin width to get probability density, then compute CCDF
    bin_widths = np.diff(bin_edges)
    probabilities = (counts / bin_widths) / (counts / bin_widths).sum()

    # Keep only non-empty bins
    mask = counts > 0
    bin_centers = bin_centers[mask]
    counts = counts[mask]
    probabilities = probabilities[mask]

    ccdf = np.cumsum(probabilities[::-1])[::-1]
    mask = ccdf > 10**-4
    return bin_centers[mask], ccdf[mask]
###########################################################################
###########################################################################
###########################################################################
alphas = [1.0, 0.8, 0.65, 0.5, 0.35]

with open('dynamic_avalanches/avalaches_mean.pkl', 'rb') as f:
    avalanches_data = pickle.load(f)

ccdf_data = {}

for a, alpha in enumerate(alphas):#enumerate([0.5]):#
    ##############################
    ccdf_data[alpha] = {}
    ##############################
    nselected = selected[alpha]['nselected']
    ##############################################
    for nindv in range(nselected):
        ccdf_data[alpha][nindv] = {}
        ##############################
        aval = avalanches_data[alpha]['aval'][nindv]
        ##############################
        values, ccdf = get_ccdf(aval, n_bins)
        slope1, intercept, r_value, p_value, std_err = loglog_linear_fit(values, ccdf)
        slope, intercept, r_value, p_value, std_err = loglog_linear_fit_model(values, ccdf)
        ccdf_data[alpha][nindv]['ccdf'] = [values, ccdf]
        ccdf_data[alpha][nindv]['linear_fit'] = slope, intercept, r_value, p_value, std_err
    aval = np.concatenate([avalanches_data[alpha]['aval'][nindv] for nindv in range(nselected)])
    values, ccdf = get_ccdf(aval, n_bins)
    ccdf_data[alpha]['ccdf'] = [values, ccdf]
    ccdf_data[alpha]['linear_fit'] = slope, intercept, r_value, p_value, std_err

with open('dynamic_avalanches/avalanches_analysis.pkl', 'wb') as f:
    pickle.dump(ccdf_data, f)

############################################################################
plt.close('all')
fig = plt.figure(figsize=[11.6, 5.0])
gm = gridspec.GridSpec(200, 170)
axes = [plt.subplot(gm[:70, i*35:i*35+30]) for i in range(5)]
ax1 = plt.subplot(gm[110:, :46])
ax2 = plt.subplot(gm[110:, 62:108])
ax3 = plt.subplot(gm[110:, 124:])
############################################################################
fz = 13
alpha_label = ['1', '0.8', '0.65', '0.5', '0.35']
colors = ['#000000', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

############################################################################
############################################################################
############# Avalanches plots
############################################################################
with open('dynamic_avalanches/avalanches_analysis.pkl', 'rb') as f:
    ccdf_data = pickle.load(f)
############################################################################
############################################################################

Group, r2, p_value, std_err = [], [], [], []

for a, alpha in enumerate(alphas):#enumerate([0.5]):#
    nselected = selected[alpha]['nselected']
    Group += ['%.2f'%alpha] * nselected

    r2values = np.array([ccdf_data[alpha][n]['linear_fit'][2]**2 for n in range(nselected)])
    f = np.array([np.mean(selected[alpha][e]['fitness']) for e in range(nselected)])
    ###########################
    r2.extend(r2values)
    
    for n in range(nselected):
        data_x, data_y = ccdf_data[alpha][n]['ccdf']
        axes[a].loglog(data_x, data_y, '-', color = colors[a], ms = 1, lw = 0.68, alpha = 0.18)
    data_x, data_y = ccdf_data[alpha]['ccdf']
    axes[a].loglog(data_x, data_y, '.-', color = colors[a], ms = 1, lw = 2.3)
    #####################
    p_value.extend(np.array([ccdf_data[alpha][n]['linear_fit'][3] for n in range(nselected)]))
    std_err.extend(np.array([ccdf_data[alpha][n]['linear_fit'][4] for n in range(nselected)]))

for aindx, ax in enumerate(axes):
    ax.set_xlabel('Avalanche size', fontsize = fz)
    ax.text(0.5, 1.1, r'$\alpha$' + ' = %s'%alpha_label[aindx], fontsize = fz, ha = 'center', transform=ax.transAxes)
    if aindx == 0:
        ax.set_ylabel('CCDF', fontsize = fz)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(0.00008, 1.25)
    ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])


############################################################################
df = pd.DataFrame({'Group': Group, 'R2': r2, 'p_value': p_value, 'std_err': std_err})
############################################################################
#################
categories = df['Group'].unique()

sns.violinplot(data=df, x='Group', y='R2', ax=ax1, order=categories, palette=colors, linewidth=0.5, width=0.85, gap=0.25, alpha = 0.4, inner = None)
sns.stripplot(data=df, x='Group', y='R2', ax=ax1, palette=colors, s = 2)

ax1.set_ylabel('R' + r'$^2$' + ' linear fit', fontsize = fz)

############################################################################
############################################################################
############# Rates and CVs plots
############################################################################
with open('dynamic_avalanches/rates.pkl', 'rb') as f:
    rates_data = pickle.load(f)
############################################################################
Group_inputs, rate_inputs, cvs_inputs = [], [], []
Group_output, rate_output, cvs_output = [], [], []
###
for a, alpha in enumerate(alphas):
    ###
    rates, cvs = rates_data[alpha]['rates_inputs'], rates_data[alpha]['cvs_inputs']
    Group_inputs += ['%.2f'%alpha] * len(rates)
    rate_inputs.extend(rates)
    cvs_inputs.extend(cvs)
    ###
    rates, cvs = rates_data[alpha]['rates_output'], rates_data[alpha]['cvs_output']
    Group_output += ['%.2f'%alpha] * len(rates)
    rate_output.extend(rates)
    cvs_output.extend(cvs)
############################################################################

df_inputs = pd.DataFrame({'Group': Group_inputs, 'rates': rate_inputs, 'cvs': cvs_inputs})
df_output = pd.DataFrame({'Group': Group_output, 'rates': rate_output, 'cvs': cvs_output})
############################################################################

sns.violinplot(data=df_inputs, x='Group', y='rates', cut=0, ax=ax2, order=categories, palette=colors, linewidth=0.5, width=0.85, gap=0.25, alpha = 0.4, inner = None)
sns.stripplot(data=df_inputs, x='Group', y='rates', ax=ax2, palette=colors, s = 2)

sns.violinplot(data=df_output, x='Group', y='rates', cut=0.5, ax=ax3, order=categories, palette=colors, linewidth=0.5, width=0.85, gap=0.25, alpha = 0.4, inner = None)
sns.stripplot(data=df_output, x='Group', y='rates', ax=ax3, palette=colors, s = 2)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Fractional order", fontsize = fz)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['1', '0.8', '0.65', '0.5', '0.35'])

ax2.set_ylabel('Input layer\naverage spks/step', fontsize = fz)
ax3.set_ylabel('Output neuron\naverage spks/step', fontsize = fz)
############################################################################
############################################################################
for ax in axes + [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
############################################################################
x1, x2, x3 = 0.01, 0.34, 0.65
y1, y2 = 0.97, 0.51
fz = 20
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x1, y2, 'B', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x2, y2, 'C', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
plt.figtext(x3, y2, 'D', ha = 'left', va = 'center', fontsize = fz, family='sans-serif')
##################################################################
fig.subplots_adjust(left = 0.06, bottom = 0.14, right = 0.96, top = 0.93)
plt.savefig('figures/figure_7_dynamic.png', dpi = 300)
plt.savefig('figures/figure_7_dynamic.pdf')
plt.savefig('../Figures/figure_7.pdf')
