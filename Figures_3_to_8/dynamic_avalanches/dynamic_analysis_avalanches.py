import numpy as np
import pickle
from scipy import signal
from scipy import stats
###########################################################################
###########################################################################
#### Load selected networks
with open('../evaluate_hybrids/to_hybrid_evaluation.pkl', 'rb') as f:
    selected = pickle.load(f)



def get_avalanches(spikes, avg_silent_length):
    spks = np.sum(spikes, axis = 0)
    ###########################
    if avg_silent_length == -1:
        #### get average silent periods
        is_silent = (spks == 0)
        padded_silent = np.pad(is_silent, (1, 1), mode='constant', constant_values=False)
        diff = np.diff(padded_silent.astype(int))
        starts = np.where(diff == 1)[0]

        ends = np.where(diff == -1)[0]
        silent_lengths = ends - starts

        if len(silent_lengths) > 0:
            avg_silent_length = np.mean(silent_lengths)
        else:
            avg_silent_length = 0.0
    ### identify what the network is doing
    in_silence = (spks[0] == 0)
    #####
    avalanches, this_avalanche = [], 0
    avalanches_count, this_avalanche_count = [], 0
    silences, this_silence = [], 0
    for s in spks:
        if s == 0: ### No spike
            this_silence += 1
            if not in_silence: ### active period ends
                if this_silence > int(avg_silent_length):
                    in_silence = True 
                    avalanches.append(this_avalanche)
                    avalanches_count.append(this_avalanche_count)
                    this_avalanche = 0
                    this_avalanche_count = 0
                else:
                    this_avalanche += s
                    this_avalanche_count += 1
        else: ## at least one spike detected
            this_avalanche += s
            this_avalanche_count += 1
            if in_silence: ## silence period ends
                in_silence = False
                silences.append(this_silence)
                this_silence = 0
        ###############
    return np.array(avalanches), np.array(avalanches_count), np.array(silences)

alphas = [1.0, 0.8, 0.65, 0.5, 0.35]
all_data = {}

for a, alpha in enumerate(alphas):
    all_data[alpha] = {}
    ##############################
    ##############################
    nselected = selected[alpha]['nselected']
    aval_alpha, avalcounts_alpha, silences_alpha = [], [], []
    ##############################################
    for nindv in range(nselected):
        aval, avalcounts, silences = [], [], []
        with open('data/spikes_%.2f_%d.pkl'%(alpha, nindv), 'rb') as f:
            simulation = pickle.load(f)
        spikes = simulation['spikes']
        ##############################
        aval, avalcounts, silences = get_avalanches(spikes, -1)
        ##############################
        aval_alpha.append(aval)
        avalcounts_alpha.append(avalcounts)
        silences_alpha.append(silences)
        ##############################
    ##############################
    all_data[alpha]['aval'] = aval_alpha
    all_data[alpha]['avalcounts'] = avalcounts_alpha
    all_data[alpha]['silences'] = silences_alpha
    ############################################################################

with open('avalaches_mean.pkl', 'wb') as f:
    pickle.dump(all_data, f)