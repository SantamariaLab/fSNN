import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import pickle
from flifnetwork import FractionalLIFNeuron
from scipy import signal
############################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
############################################################################
nampl = 0.1
############################################################################
Vthreshold = 10
Vreset = 0
memory_length = 1500
############################################################################
############################################################################
dt = 1
## reduce this times for testing
ttot = 300 * 1000
tadapt = 30 * 1000

Imin, Imax = 6, 20
Iampl = Imax - Imin

alphas = [1, 0.5, 0.2]
slopes = [1, 10, 60]
tau = 1.2

############################################################################
############################################################################
if rank == 0:
    data = {}
    data['alphas'] = alphas
    data['Imin'] = Imin
    data['Imax'] = Imax
    data['ttot'] = size * (ttot - tadapt)/1000
    data['slopes'] = slopes

############################################################################
############################################################################
def get_hysteresis(alpha, tau, slope):
    '''
    Simulate neuron with a periodic stimuli and return the sp_phase
    which is a list containing the phase of the stimuli at which the spikes ocurr 
    sp_phase will later be used to calculate the firing rate as a function of phase
    '''
    freq = slope/(2*Iampl)
    period = 1./freq
    ##### calculating eff time to get full Iapp cycles in adaptation and simulation
    period_ms = 1000./freq
    ttot_eff = (ttot//period_ms)*period_ms
    tadapt_eff = (tadapt//period_ms)*period_ms
    valid_time = (ttot_eff - tadapt_eff)/1000
    #######################
    time = np.arange(0, ttot_eff, dt) /1000
    #######################
    Iapp = Imin + Iampl*0.5*(1 + signal.sawtooth(2*np.pi*time*freq, width = 0.5))
    noise = np.random.normal(0, nampl, len(time))
    Iapp = Iapp * (1+noise)

    #######################
    spikes = []
    N = FractionalLIFNeuron(tau, 0, Vthreshold, Vreset, alpha, memory_length, dt)
    N.reset()
    #######################
    for k, t in enumerate(time):
        if N.spike_state:
            spikes.append(t)
        N.update(Iapp[k], dt)
    #######################
    sp = np.array(spikes)
    sp = sp[np.where(sp>tadapt_eff/1000)]
    sp_phase = ((sp%period)/period)
    #######################
    #######################
    return valid_time, sp_phase
############################################################################
############################################################################
for a,  alpha in  enumerate(alphas):
    if rank == 0:
        data[alpha] = {}
    ######################################################################
    for s, slope in enumerate(slopes):
        freq = slope/(2*Iampl)
        ###########################################################################
        valid_time, sp_phase = get_hysteresis(alpha, tau, slope)
        # ############################################################################
        all_phases = comm.gather(sp_phase, root = 0)
        if rank == 0:
            print(alpha, slope, freq, 1/freq)
            data[alpha][slope] = [valid_time, np.concatenate(all_phases)]
############################################################################
############################################################################
if rank ==0:
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)

