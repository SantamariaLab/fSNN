import numpy as np
import matplotlib.pylab as plt
import gymnasium as gym
import pickle

sys.path.insert(0, '../utils')
from flifnetwork import FractionalLIFNetwork 
from lifnetwork import LIFNetwork 

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
############################################################################
############################################################################
##########   SNN hyperparameters
dt = 1.0
steps_per_episode = 10
memory_length = 120 ## fractional memory extend 12 episodes back

n_inputs = 2 ### hardest problem
n_outputs = 1 ### R neuron
Vthreshold = 10
Vreset = 0
### fractional
n_hidden_frac = 6
memb_tau_frac = 3.5
### non fractional
n_hidden = 12
memb_tau = 10
############################################################################
############################################################################
def observation_to_inputs(observation):
    ans = observation / np.array([2.4, 2.2, 0.05, 2.0])
    return ans[[0, 2]]
############################################################################
############################################################################
############################################################################
def evaluate_fitness(individual1, individual2, fitness_goal, dt):
    ######### load non fractiona network
    B1 = individual1[0]
    input_weights1 = np.vstack([individual1[1:1+n_hidden], individual1[1+1*n_hidden:1+2*n_hidden]])
    output_weights1 = np.vstack([individual1[1+2*n_hidden:1+3*n_hidden]])
    Net1 = LIFNetwork(n_hidden, n_outputs, memb_tau, B1, Vthreshold, Vreset, input_weights1,
                                output_weights1, dt)
    ######### load fractional network
    B2 = individual2[0]
    input_weights2 = np.vstack([individual2[1:1+n_hidden_frac], individual2[1+1*n_hidden_frac:1+2*n_hidden_frac]])
    output_weights2 = np.vstack([individual2[1+2*n_hidden_frac:1+3*n_hidden_frac]])
    Net2 = FractionalLIFNetwork(n_hidden_frac, n_outputs, memb_tau_frac, B2, Vthreshold, Vreset, input_weights2,
                                output_weights2, alpha, memory_length, dt, use_cython=True)
    ### open cartpole environment
    env = gym.make('CartPole-v1')
    ########################################
    shifted = False
    ########################################
    observation, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        ######################### Network 1 control
        if not shifted:
            action1 = Net1.simulate(observation_to_inputs(observation), steps_per_episode)
            action2 = Net2.simulate(observation_to_inputs(observation), steps_per_episode)
            observation, reward, terminated, truncated, info = env.step(action1)
            done = terminated or (total_reward > fitness_goal) 
            total_reward += reward
            ######### minimal period of network 1 and pole near equilibrium
            if (total_reward > shift_point):
                shifted = True
        ######################### Network 2 take control
        else:
            action2 = Net2.simulate(observation_to_inputs(observation), steps_per_episode)
            observation, reward, terminated, truncated, info = env.step(action2)
            done = terminated or (total_reward > fitness_goal) 
            total_reward += reward
    env.close()
    return total_reward
############################################################################
##########   GA hyperparameters
max_fitness = 15000
nepisodes = 1000
shift_point = 6  ## based on derivative of the cpd

#################################
with open('to_hybrid_evaluation.pkl', 'rb') as f:
    selected = pickle.load(f)
#################################
### Load best non fractional individual
indv1 = selected[1.0][29]['indv']


for alpha in [0.8, 0.65, 0.5, 0.35]:
    for evo in np.arange(selected[alpha]['nselected']):
        ######### load fractional individual
        indv2 = selected[alpha][evo]['indv']
        eval = []
        if rank == 0:
            this_hybrid = {}
            this_hybrid['alpha'] = alpha
            this_hybrid['indv_indx'] = evo
            this_hybrid['fitness_fractional'] = selected[alpha][evo]['fitness']
        ############################################################################
        for rep in range(nepisodes):
            if rank == rep%size:
                fit = evaluate_fitness(indv1, indv2, max_fitness, dt)
                eval.append(fit)
        eval = np.array(eval)
        all_fit = comm.gather(eval, root=0)
        if rank == 0:
            all_fit = np.concatenate(all_fit)
            this_hybrid['fitness_mixed'] = all_fit
            print('alpha %.1f, evo %d'%(alpha, evo))
            print('no inital memory performance = %d, adaptation memory performance = %d'%(np.mean(this_hybrid['fitness_fractional']), np.mean(all_fit)))
            ##################################
            with open('data/rep_%.2f_%d.pkl'%(alpha, evo), 'wb') as f:
                pickle.dump(this_hybrid, f)

############################################################################
