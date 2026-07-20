import numpy as np
from lifnetwork_step import update_state

class LIFNetwork:
    """
    A two-layer Leaky Integrate-and-Fire (LIF) neural network
    """
    def __init__(self, hidden_layer_size=8, output_layer_size=2,
                 membrane_time_constant=1.2, neurons_bias=2.0,
                 threshold_voltage=10.0, reset_voltage=0.0,
                 input_weights=None, output_weights=None,
                 dt = 1, 
                 use_cython=True):
        # Network dimensions
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Neuron state variables
        self.hidden_layer_voltages = np.zeros(hidden_layer_size, dtype=np.float64)
        self.hidden_layer_spike_states = np.zeros(hidden_layer_size, dtype=np.int32)
        self.output_layer_voltages = np.zeros(output_layer_size, dtype=np.float64)
        self.output_layer_spike_states = np.zeros(output_layer_size, dtype=np.int32)

        # Connectivity matrices
        self.hidden_weights = np.asarray(input_weights, dtype=np.float64)
        self.output_weights = np.asarray(output_weights, dtype=np.float64)

        # Network parameters
        self.membrane_time_constant = membrane_time_constant
        self.threshold_voltage = threshold_voltage
        self.reset_voltage = reset_voltage
        self.neurons_bias = neurons_bias

        self.dt = dt

        # Accumulator for synaptic inputs to the output layer
        self.output_layer_synaptic_inputs = np.zeros(output_layer_size, dtype=np.float64)
        
        # Flag that allow to shift from pure python to cython integration imlementation
        self.use_cython = use_cython

    def reset(self):
        """
        Reset all network state variables (voltages, spike states, and history) to their initial values.
        """
        self.hidden_layer_voltages.fill(0.0)
        self.output_layer_voltages.fill(0.0)
        self.hidden_layer_spike_states.fill(0)
        self.output_layer_spike_states.fill(0)
        self.output_layer_synaptic_inputs.fill(0.0)


    def update(self, cartpole_inputs_weighted, dt=0.10):
        """
        Update the network state for one time step using fractional order dynamics.
        The update rule for voltage $V$ is:
        $V(t) = \frac{1}{\tau}(-V(t-\Delta t) + I_{bias} + I_{syn}) (\Delta t)^\alpha - \sum_{j=1}^{L} c_j^{(\alpha)} V(t-j\Delta t)$
        """
        if self.use_cython:
            update_state(
                cartpole_inputs_weighted,
                self.hidden_layer_voltages,
                self.hidden_layer_spike_states,
                self.neurons_bias,
                self.threshold_voltage,
                self.reset_voltage,
                self.dt,
                self.membrane_time_constant,
                self.output_weights,
                self.output_layer_voltages,
                self.output_layer_spike_states,
                self.output_layer_synaptic_inputs,
            )
        else:
            # Reset spike states for the current time step
            self.hidden_layer_spike_states.fill(0)
            self.output_layer_spike_states.fill(0)

            # --- Update Hidden Layer ---
            # Calculate the new hidden layer voltages based on the fractional LIF dynamics
            # The equation is V_new = (-V_old + bias + input) / tau * kernel - history_component
            self.hidden_layer_voltages = self.hidden_layer_voltages + (
                -self.hidden_layer_voltages + self.neurons_bias + cartpole_inputs_weighted
            ) * self.dt / self.membrane_time_constant

            # Detect spikes in the hidden layer and reset voltages of spiking hidden neurons
            spiked_hidden = self.hidden_layer_voltages >= self.threshold_voltage
            self.hidden_layer_spike_states[spiked_hidden] = 1
            self.hidden_layer_voltages[spiked_hidden] = self.reset_voltage

            # Update synaptic inputs to the output layer from spiking hidden neurons
            if np.any(spiked_hidden):
                self.output_layer_synaptic_inputs += np.sum(
                    self.output_weights[:, spiked_hidden], axis=1
                )

            # --- Update Output Layer ---
            self.output_layer_voltages = self.output_layer_voltages + (
                -self.output_layer_voltages + self.neurons_bias + self.output_layer_synaptic_inputs
            ) * self.dt / self.membrane_time_constant

            # Detect spikes in the output layer and reset voltages
            spiked_output = self.output_layer_voltages >= self.threshold_voltage
            self.output_layer_spike_states[spiked_output] = 1
            self.output_layer_voltages[spiked_output] = self.reset_voltage

            # Clear the synaptic input accumulator for the next time step
            self.output_layer_synaptic_inputs.fill(0.0)
        ####################################################################################

    def simulate(self, cartpole_inputs, nsteps):
        """
        Simulate the network for a given number of steps (nsteps)
        and return a control decision based on output layer spikes.
        """  
        # Compute weighted inputs to the hidden layer from the cartpole inputs
        cartpole_inputs_weighted = np.matmul(cartpole_inputs, self.hidden_weights)
        # reset spike counting in the output layer
        right_spikes = 0
        # Simulate the network for the specified duration
        for _ in range(int(nsteps)):
            self.update(cartpole_inputs_weighted, self.dt)
            right_spikes += self.output_layer_spike_states[0]
        # Return control decision based on which output neuron spiked more
        return int(right_spikes > 0)
