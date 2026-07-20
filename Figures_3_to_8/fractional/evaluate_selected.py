import numpy as np
import matplotlib.pylab as plt
import gymnasium as gym
import pickle

sys.path.insert(0, '../utils')
from flifnetwork import FractionalLIFNetwork 

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
############################################################################
##########   SNN hyperparameters
dt = 1.0
steps_per_episode = 10
memory_length = 120 ## fractional memory extend 12 episodes back

n_inputs = 2 ### hardest problem
n_hidden = 6
n_outputs = 1 ### L, R neurons
memb_tau = 3.5
Vthreshold = 10
Vreset = 0

############################################################################
############################################################################
def observation_to_inputs(observation):
    ans = observation / np.array([2.4, 2.2, 0.05, 2.0])
    return ans[[0, 2]]
############################################################################
def evaluate_fitness(individual, fitness_goal, dt):
    B = individual[0]
    input_weights = np.vstack([individual[1:1+n_hidden], individual[1+1*n_hidden:1+2*n_hidden]])
    output_weights = [individual[1+2*n_hidden:1+3*n_hidden]]
    ### open cartpole environment
    env = gym.make('CartPole-v1')
    ### create network
    Net = FractionalLIFNetwork(n_hidden, n_outputs, memb_tau, B, Vthreshold, Vreset, input_weights,
                                output_weights, alpha, memory_length, dt, use_cython=True)
    ########
    observation, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # action = env.action_space.sample()
        action = Net.simulate(observation_to_inputs(observation), steps_per_episode)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or (total_reward > fitness_goal) 
        total_reward += reward
    env.close()
    return total_reward
############################################################################
############################################################################
############################################################################
##########   GA hyperparameters
nevolutions = 40
max_fitness = 15000
#################################
alphas = [0.95, 0.8, 0.65, 0.5, 0.35, 0.2]
## SEvaluate best individual of each evolution on 1000 episodes test
if rank == 0:
    evaluation = {}

for a, alpha in enumerate(alphas):
    ##################################
    with open('selected/selected_%.1f.pkl'%(alpha), 'rb') as f:
        selected = pickle.load(f)
    ##################################
    if rank == 0:
        print('\nalpha = %.1f'%alpha)
        evaluation[alpha] = {}
    ############################################################################
    for evol in np.arange(nevolutions):
        ###########################################################################
        indv = selected[evol]['indv']
        eval = []
        for rep in range(1000):
            if rank == rep%size:
                eval.append(evaluate_fitness(indv, max_fitness, dt))
        eval = np.array(eval)
        all_eval = comm.gather(eval, root=0)
        if rank == 0:
            all_eval = np.concatenate(all_eval)
            print('evol = %d, fitness = %d'%(evol, np.mean(all_eval)))
            evaluation[alpha][evol] = {}
            evaluation[alpha][evol]['indv'] = indv
            evaluation[alpha][evol]['fitness'] = all_eval
        ##################################
if rank == 0:
    with open('selected/evaluation_1000.pkl', 'wb') as f:
        pickle.dump(evaluation, f)
    ###################################

############################################################################
############################################################################
############################################################################