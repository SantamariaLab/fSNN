import numpy as np
import pickle
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
##################################################################
nevolutions = 40
fitness_goal = 15002
task_goal = 6000
############################################################################
selected = {}
####################################
#######  Load non perturbated environment data
####################################
with open('../non_fractional/selected/evaluation_1000.pkl', 'rb') as f:
    data = pickle.load(f)
####################################
selected[1.0] = {}
nselected = 0
for evo in range(nevolutions):
    if np.mean(data[evo]['fitness']) > task_goal:
        selected[1.0][nselected] = {}
        selected[1.0][nselected]['indv'] = data[evo]['indv']
        selected[1.0][nselected]['fitness'] = data[evo]['fitness']
        nselected +=1
selected[1.0]['nselected'] = nselected
#############################################################################
with open('../fractional/selected/evaluation_1000.pkl', 'rb') as f:
    data = pickle.load(f)
####################################
alphas = [0.95, 0.8, 0.65, 0.5, 0.35, 0.2]                  
for alpha in alphas:
    selected[alpha] = {}
    nselected = 0
    for evo in range(nevolutions):
        if np.mean(data[alpha][evo]['fitness']) > task_goal:
            selected[alpha][nselected] = {}
            selected[alpha][nselected]['indv'] = data[alpha][evo]['indv']
            selected[alpha][nselected]['fitness'] = data[alpha][evo]['fitness']
            nselected +=1
    selected[alpha]['nselected'] = nselected
####################################

with open('to_hybrid_evaluation.pkl', 'wb') as f:
    pickle.dump(selected, f)
