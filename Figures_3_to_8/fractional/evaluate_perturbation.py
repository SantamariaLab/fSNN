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
### Noisy cartpole environment
############################################################################
############################################################################
class CartPoleWithNoise(gym.Wrapper):
    def __init__(self, env):
        """
        Wrap the cartpole environment to add an esternal noise at each step
        Args:
            env: The cartpole environment
        """
        super(CartPoleWithNoise, self).__init__(env)

    def step(self, ext_noise, action):
        # Get the normal force based on action (either +10N or -10N)
        base_force = 10.0 if action == 1 else -10.0
        # Add OU noise
        total_force = base_force + ext_noise
        # Convert back to the expected action format (0 or 1)
        self.unwrapped.force_mag = abs(total_force)
        modified_action = 1 if total_force > 0 else 0
        # Call the original step function
        observation, reward, terminated, truncated, info = self.env.step(modified_action)
        # Restore the original force magnitude
        self.unwrapped.force_mag = 10.0
        # Add noise information to info dict
        info['noise'] = ext_noise
        info['total_force'] = total_force
        return observation, reward, terminated, truncated, info
############################################################################
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
def evaluate_fitness(individual, fitness_goal, dt, Ampl):
    B = individual[0]
    input_weights = np.vstack([individual[1:1+n_hidden], individual[1+1*n_hidden:1+2*n_hidden]])
    output_weights = [individual[1+2*n_hidden:1+3*n_hidden]]
    ### open cartpole environment
    env = gym.make('CartPole-v1')
    env = CartPoleWithNoise(env)
    ### create network
    Net = FractionalLIFNetwork(n_hidden, n_outputs, memb_tau, B, Vthreshold, Vreset, input_weights,
                                output_weights, alpha, memory_length, dt, use_cython=True)
    ########
    observation, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        ext_noise = np.random.normal(0, Ampl)
        action = Net.simulate(observation_to_inputs(observation), steps_per_episode)
        observation, reward, terminated, truncated, info = env.step(ext_noise, action)
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

Amplitudes = np.arange(0.1, 15, 0.2)
## Evaluate best individual of each evolution on 1000 episodes test at different enviroenmental noise
for a, alpha in enumerate(alphas):
    ##################################
    with open('./selected/selected_%.1f.pkl'%(alpha), 'rb') as f:
        selected = pickle.load(f)
    ##################################
    if rank == 0:
        print('\nalpha = %.1f'%alpha)
        evaluation= {}
    for Ampl in Amplitudes: 
        if rank == 0:
            all_indv_evaluations = []
            evaluation[Ampl]= {}
        ############################################################################
        for evol in np.arange(nevolutions):
            ###########################################################################
            indv = selected[evol]['indv']
            eval = []
            for rep in range(1000):
                if rank == rep%size:
                    eval.append(evaluate_fitness(indv, max_fitness, dt, Ampl))
            eval = np.array(eval)
            all_eval = comm.gather(eval, root=0)
            if rank == 0:
                all_eval = np.concatenate(all_eval)
                all_indv_evaluations.append(all_eval)
                evaluation[Ampl][evol] = all_eval
            ##################################
        if rank == 0:
            print('noise = %.1f, performance = %.1f'%(Ampl, np.mean(np.concatenate(all_indv_evaluations))))
        ##################################
    if rank == 0:
        with open('selected/evaluate_perturbation_%.1f.pkl'%(alpha), 'wb') as f:
            pickle.dump(evaluation, f)
        ###################################

############################################################################
############################################################################
############################################################################