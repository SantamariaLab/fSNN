import numpy as np
import gymnasium as gym
import pickle
import signal
import subprocess
import os
import time
import re
from flifnetwork import FractionalLIFNetwork
from lifnetwork import LIFNetwork

###########################################################################
##########  utilities to meausure Flops
###########################################################################

def start_perf():
    """Attach perf to current process and return the perf subprocess."""
    pid = os.getpid()
    perf_process = subprocess.Popen(
        ['perf', 'stat',
         '-e', 'fp_ret_sse_avx_ops.add_sub_flops',
         '-e', 'fp_ret_sse_avx_ops.mult_flops',
         '-e', 'fp_ret_sse_avx_ops.mac_flops',
         '-e', 'fp_ret_sse_avx_ops.div_flops',
         '-e', 'fp_ret_x87_fp_ops.all',
         '-p', str(pid)],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    time.sleep(3.0)
    return perf_process

def stop_perf(perf_process):
    """Stop perf"""
    perf_process.send_signal(signal.SIGINT)
    try:
        _, stderr = perf_process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        perf_process.kill()
        _, stderr = perf_process.communicate()
    stats = parse_perf_output(stderr.decode())
    return compute_total_flops(stats)

def parse_perf_output(perf_output):
    """Parse perf stat stderr into a dictionary."""
    stats = {}
    for line in perf_output.splitlines():
        line = line.strip()
        match = re.match(r'^([\d,]+)\s+([\w.]+)', line)
        if match:
            value_str, event_name = match.groups()
            stats[event_name] = int(value_str.replace(',', ''))
    return stats

def compute_total_flops(perf_stats):
    """
    Compute total FLOPs for AMD CPU.
    mac_flops counts multiply-accumulate as 2 FLOPs (mul + add).
    """
    flops  = perf_stats.get('fp_ret_sse_avx_ops.add_sub_flops', 0) * 1
    flops += perf_stats.get('fp_ret_sse_avx_ops.mult_flops', 0)     * 1
    flops += perf_stats.get('fp_ret_sse_avx_ops.mac_flops', 0)      * 2
    flops += perf_stats.get('fp_ret_sse_avx_ops.div_flops', 0)      * 1
    flops += perf_stats.get('fp_ret_x87_fp_ops.all', 0)             * 1
    return flops

###########################################################################
##########   SNN hyperparameters
###########################################################################

dt               = 1.0
steps_per_episode = 10
memory_length    = 120
n_inputs         = 2
n_outputs        = 1
Vthreshold       = 10
Vreset           = 0
n_hidden_frac    = 6
memb_tau_frac    = 3.5
n_hidden         = 12
memb_tau         = 10

###########################################################################
##########   Cart-pole simulation
###########################################################################

def observation_to_inputs(observation):
    ans = observation / np.array([2.4, 2.2, 0.05, 2.0])
    return ans[[0, 2]]

def evaluate_activity(individual, alpha, fitness_goal, dt):
    """Run simulation and return episode data."""
    if alpha == 1:
        B = individual[0]
        input_weights  = np.vstack([individual[1:1+n_hidden],
                                    individual[1+n_hidden:1+2*n_hidden]])
        output_weights = np.vstack([individual[1+2*n_hidden:1+3*n_hidden]])
        Net = LIFNetwork(n_hidden, n_outputs, memb_tau, B, Vthreshold, Vreset,
                         input_weights, output_weights, dt)
    else:
        B = individual[0]
        input_weights  = np.vstack([individual[1:1+n_hidden_frac],
                                    individual[1+n_hidden_frac:1+2*n_hidden_frac]])
        output_weights = np.vstack([individual[1+2*n_hidden_frac:1+3*n_hidden_frac]])
        Net = FractionalLIFNetwork(n_hidden_frac, n_outputs, memb_tau_frac, B,
                                   Vthreshold, Vreset, input_weights, output_weights,
                                   alpha, memory_length, dt, use_cython=True)

    env = gym.make('CartPole-v1')
    observation, _ = env.reset()
    done    = False
    episode = 0
    states, spikes, voltages, memory = [], [], [], []

    while not done:
        action, step_spikes, step_voltages, step_memory = Net.simulate(
            observation_to_inputs(observation), steps_per_episode)
        observation, _, terminated, _, _ = env.step(action)
        states.append(observation)
        spikes.append(np.array(step_spikes))
        voltages.append(np.array(step_voltages))
        memory.append(np.array(step_memory))
        done = terminated or (episode > fitness_goal)
        episode += 1

    env.close()

    spikes   = np.array(spikes).transpose(2, 0, 1)
    voltages = np.array(voltages).transpose(2, 0, 1)
    memory   = np.array(memory).transpose(2, 0, 1)

    return (episode,
            np.array(states).T,
            spikes.reshape(spikes.shape[0], -1),
            voltages.reshape(voltages.shape[0], -1),
            memory.reshape(memory.shape[0], -1))

###########################################################################
##########   Main
########## Run every succesful Network and store dynamic
########## only if episode last 400 seconds
###########################################################################

alphas = [1.0, 0.8, 0.65, 0.5, 0.35]
goal   = 20000

with open('../evaluate_hybrids/to_hybrid_evaluation.pkl', 'rb') as f:
    selected = pickle.load(f)

flops_summary = {}

for alpha in alphas:
    nselected = selected[alpha]['nselected']
    flops_summary[alpha] = np.zeros(nselected)

    for nindv in range(nselected):
        indv      = selected[alpha][nindv]['indv']
        not_done  = True

        while not_done:
            perf_proc = start_perf()
            episode, states, spikes, voltages, memory = evaluate_activity(
                indv, alpha, goal, dt)
            total_flops = stop_perf(perf_proc)

            print('alpha %.2f | indv %d | episodes %d | FLOPs %d' % (
                alpha, nindv, episode, total_flops))

            if episode > goal:

                flops_summary[alpha][nindv] = total_flops

                with open('data/spikes_%.2f_%d.pkl' % (alpha, nindv), 'wb') as f:
                    pickle.dump({
                        'states':      states,
                        'spikes':      spikes,
                        'voltages':    voltages,
                        'memory':      memory,
                        'total_flops': total_flops
                    }, f)
                not_done = False

# save flops summary
with open('data/flops_summary.pkl', 'wb') as f:
    pickle.dump(flops_summary, f)
