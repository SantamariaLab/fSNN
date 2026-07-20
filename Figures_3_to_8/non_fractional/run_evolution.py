import numpy as np
import gymnasium as gym
from mpi4py import MPI
import pickle

sys.path.insert(0, '../utils')
from lifnetwork import LIFNetwork 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################################################
#########   Evolutionary Algoritm
############################################################################
def create_individual(n_parameters):
    """Creates a random individual
    Bias, Input gain, membrane tau and weights."""
    B = np.random.uniform(10, 12, 1)
    W = np.random.uniform(-10, 10, int(n_inputs*n_hidden))
    WR = np.random.uniform(-10, 10, int(n_hidden))
    return np.concatenate([B, W, WR])


def selection(population, fitnesses, num_parents=2):
    """Selects parents based on fitness (roulette wheel selection)."""
    fitness_sum = sum(fitnesses)
    probabilities = [f / fitness_sum for f in fitnesses]
    parents_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return [population[i] for i in parents_indices]

def crossover(parents, crossover_rate=0.12):
    """Performs crossover (weights and bias)."""
    if np.random.rand() < crossover_rate:
        ### Swap elements with prob = 0.1
        parent1, parent2 = parents
        crossover_points = np.random.rand(len(parent1)) < 0.1

        temp = parent1[crossover_points].copy()
        parent1[crossover_points] = parent2[crossover_points]
        parent2[crossover_points] = temp
        return parent1, parent2
    else:
        return parents[0], parents[1]

def mutate_parameter(value, mutation_strength, l_lim, h_lim):
    value += np.random.uniform(-mutation_strength, mutation_strength)
    return np.clip(value, l_lim, h_lim) 

def mutation(individual, mutation_rate=0.15):
    """Performs mutation with clamping, treating each parameter individually."""
    do_mutation = np.random.rand(n_parameters) < mutation_rate
    mutated_individual = individual.copy()
    ##########
    if do_mutation[0]: 
        mutated_individual[0] = mutate_parameter(individual[0], 1, 8.0, 22) ## mutate neurons bias
    ##########
    for p in range(1, n_parameters):
        if do_mutation[p]: 
            mutated_individual[p] = mutate_parameter(individual[p], 2, -20, 20) ## mutate weigths
    return mutated_individual

############################################################################
###########################################################################
##########   SNN hyperparameters
dt = 1.0
steps_per_episode = 10
memory_length = 120 ## fractional memory extend 12 episodes back

n_inputs = 2 ### hardest problem
n_hidden = 12
n_outputs = 1 ###  R neuron
memb_tau = 1.2
Vthreshold = 10
Vreset = 0
#############
alpha = 1.0

n_parameters = int(1 + n_inputs*n_hidden + n_outputs*n_hidden) # B + weights:(inputs->hidden) + weights:(hidden->output)

##########   GA hyperparameters
num_generations = 600
population_size = size # defined on the job batch file, I'll use 150 cores (each core evaluate one individual in the population)

##########   GYM evaluation hyperparameters
nrep_per_evaluation = 10

def observation_to_inputs(observation):
    ans = observation / np.array([2.4, 2.2, 0.05, 2.0])
    return ans[[0,2]]
    ### hardest task, using only position and angle

def evaluate_fitness(individual, num_episodes, fitness_goal, dt):
    B = individual[0]
    input_weights = np.vstack([individual[1:1+n_hidden], individual[1+1*n_hidden:1+2*n_hidden]])
    output_weights = np.vstack([individual[1+2*n_hidden:1+3*n_hidden]])
    ### open cartpole environment
    env = gym.make('CartPole-v1')
    ### create network
    Net = LIFNetwork(n_hidden, n_outputs, memb_tau, B, Vthreshold, Vreset, input_weights,
                                output_weights, dt)
    ########
    total_reward = 0
    for r in range(num_episodes):
        observation, info = env.reset()
        done = False
        episode = 0
        while not done:
            # action = env.action_space.sample()
            action = Net.simulate(observation_to_inputs(observation), steps_per_episode)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or (episode > fitness_goal) 
            total_reward += reward
            episode += 1
    env.close()
    return total_reward / num_episodes
    

############################################################################
#########   Main GA Loop with MPI
############################################################################
#############
taus = [10]
number_of_evolutions = 40
fitness_goal = 15000

for tau in taus:
    memb_tau = tau
    for evolution in range(number_of_evolutions):
        if rank == 0:
            evolution_data = {}
            population = [create_individual(n_parameters) for _ in range(population_size)]
            ### All indivuduals in the EA would have the same random input weights
        else:
            population = None

        ## Flag to stop evolution if it reaches max fitness over 10 consecutive epochs
        should_break = False
        number_of_generation_with_maximum_fitneess = 0

        for generation in range(num_generations):
            ### Evaluate Population
            #############
            #broadcast population to all processes.
            population = comm.bcast(population, root=0) 
            #split population to each process.
            local_population = np.array_split(population, size)[rank] 
            #calculate fitness for each local population.
            local_fitnesses = [evaluate_fitness(individual, nrep_per_evaluation, fitness_goal, dt) for individual in local_population] 
            # print(rank, local_fitnesses)
            #gather all fitnesses to root process.
            all_fitnesses = comm.gather(local_fitnesses, root=0) 
            #############
            if rank == 0:
                ### Analyze fitnesses and create new population
                ########## gather local results
                fitnesses = []
                for local_fitness in all_fitnesses:
                    fitnesses.extend(local_fitness)

                ########## Save generation
                evolution_data[generation] = {}
                evolution_data[generation]['fitnesses'] = fitnesses
                evolution_data[generation]['population'] = population
                
                ########## Check if max fitness goal is meet
                best_fitness = max(fitnesses)
                if best_fitness >= fitness_goal:
                    number_of_generation_with_maximum_fitneess += 1
                else:
                    number_of_generation_with_maximum_fitneess = 0
                ###########
                if number_of_generation_with_maximum_fitneess >= 10:
                    should_break = True
                #########################
                if (generation%80) == 4:
                    print(f"Generation {generation}, Best Fitness: {best_fitness}")
                #########################
                ### Adjust elitists and random size during evolution
                elite_size = int(population_size*0.1) # Number of best individuals to keep
                # Number of new random individuals each generation decreases linearly from 60% to 0%
                if elite_size%2 == 0:
                    random_size = 2 * int(population_size * 0.3 * (num_generations - generation)/num_generations) 
                if elite_size%2 == 1:
                    random_size = 1 + 2 * int(population_size * 0.3 * (num_generations - generation)/num_generations) 

                sorted_indices = np.argsort(fitnesses)[::-1]
                elite_indices = sorted_indices[:elite_size]
                elite = [population[i] for i in elite_indices]
                new_random = [create_individual(n_parameters) for _ in range(random_size)]

                new_population = elite
                new_population.extend(new_random)

                for _ in range((population_size - elite_size - random_size) // 2):
                    parents = selection(population, fitnesses)
                    child1, child2 = crossover(parents)
                    child1 = mutation(child1)
                    child2 = mutation(child2)
                    new_population.extend([child1, child2])

                population = new_population
            ##### stop if max fitness is reached 10 times in a row
            should_break = comm.bcast(should_break, root = 0)
            if should_break:
                break


        # Get the best individual (root process only)
        if rank == 0:
            print(f'tau {tau}, evolution {evolution};  Best fitness: {best_fitness}')
            with open('data/evolution_%.1f_%d.pkl'%(tau, evolution), 'wb') as f:
                pickle.dump(evolution_data, f)
    ############################################################################
