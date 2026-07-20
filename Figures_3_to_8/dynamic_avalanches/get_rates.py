import numpy as np
import pickle
from scipy import signal
from scipy import stats
###########################################################################
###########################################################################
#### Load selected networks
with open('../evaluate_hybrids/to_hybrid_evaluation.pkl', 'rb') as f:
    selected = pickle.load(f)

ttot = np.arange(0, 200.02, 0.001) ### the cartpole was evaluated for 400 seconds -> 20000 steps -> 200000 SNN steps


def get_rates(spikes):
    nneurons = len(spikes)
    nsteps = (len(spikes[0])/10) #number of cart-pole steps
    rates, cvs = np.zeros(nneurons), np.zeros(nneurons)
    ###########################
    for s, spks in enumerate(spikes):
        spk_times = time[spks == 1]
        isis = np.diff(spk_times)
        # rates[s] = 1./np.mean(isis)
        rates[s] = spks.sum()/nsteps
        cvs[s] = np.std(isis)/np.mean(isis)
    ############################
    ### desitions
    chunks = spikes[-1].reshape(-1, 10) ### decitions block
    left_probability = (np.sum(chunks, axis=1) > 0).sum() /len(chunks)
    print(left_probability)
    return rates, cvs

alphas = [1.0, 0.8, 0.65, 0.5, 0.35]
all_rates = {}

for a, alpha in enumerate(alphas):
    all_rates[alpha] = {}
    rates_inputs = []
    rates_output = []
    cvs_inputs = []
    cvs_output = []
    ##############################
    ##############################
    nselected = selected[alpha]['nselected']
    ##############################################
    for nindv in range(nselected):
        with open('data/spikes_%.2f_%d.pkl'%(alpha, nindv), 'rb') as f:
            simulation = pickle.load(f)
        spikes = simulation['spikes']
        ###########################
        rates, cvs = get_rates(spikes)
        rates_output.append(rates[-1])
        cvs_output.append(cvs[-1])
        rates_inputs.extend(rates[:-1])
        cvs_inputs.extend(cvs[:-1])
        ##############################
    ##############################
    all_rates[alpha]['rates_inputs'] = rates_inputs
    all_rates[alpha]['cvs_inputs'] = cvs_inputs
    all_rates[alpha]['rates_output'] = rates_output
    all_rates[alpha]['cvs_output'] = cvs_output
    ############################################################################

with open('rates.pkl', 'wb') as f:
    pickle.dump(all_rates, f)