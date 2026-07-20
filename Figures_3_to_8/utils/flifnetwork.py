import numpy as np
from flifnetwork_step import update_state  # Make sure this is compiled and available

class FractionalLIFNetwork:
    """
    A two-layer fractional order Leaky Integrate-and-Fire (LIF) neural network

    This class simulates the dynamics of a neural network where neuron voltages
    are governed by a fractional-order differential equation, approximated
    using the Grunwald-Letnikov definition.
    """
    def __init__(self, hidden_layer_size=8, output_layer_size=2,
                 membrane_time_constant=1.2, neurons_bias=2.0,
                 threshold_voltage=10.0, reset_voltage=0.0,
                 input_weights=None, output_weights=None,
                 fractional_order=0.9, memory_length=120,
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

        # Fractional order parameters
        self.fractional_order = fractional_order
        self.memory_length = memory_length
        self.dt = dt
        self.kernel = self._compute_kernel(dt, fractional_order)
        # Grunwald-Letnikov coefficients for fractional derivative
        self.gl_coefficients = self._compute_gl_coefficients(fractional_order, memory_length)

        # History arrays to store past voltage values for the fractional derivative
        self.hidden_voltage_history = np.zeros((memory_length, hidden_layer_size), dtype=np.float64)
        self.output_voltage_history = np.zeros((memory_length, output_layer_size), dtype=np.float64)
        # Components for the history term in the integration step
        self.hidden_history_component = np.zeros(hidden_layer_size, dtype=np.float64)
        self.output_history_component = np.zeros(output_layer_size, dtype=np.float64)

        # Accumulator for synaptic inputs to the output layer
        self.output_layer_synaptic_inputs = np.zeros(output_layer_size, dtype=np.float64)

        # Flag that allow to shift from pure python to cython integration imlementation
        self.use_cython = use_cython

    def _compute_gl_coefficients(self, alpha, length):
        """
        Compute Grunwald-Letnikov coefficients for the fractional derivative.
        The coefficients $c_j^{(\alpha)}$ are defined recursively as:
        $c_0^{(\alpha)} = -\alpha$
        $c_j^{(\alpha)} = (1 - \frac{\alpha+1}{j+1}) c_{j-1}^{(\alpha)}$ for $j \ge 1$
        """
        coeffs = np.zeros(length, dtype=np.float64)
        if length > 0:
            coeffs[0] = -alpha
            for j in range(1, length):
                coeffs[j] = (1.0 - (alpha + 1.0) / (j + 1)) * coeffs[j - 1]
        return coeffs


    def _compute_kernel(self, dt, alpha):
        return dt ** alpha


    def reset(self):
        """
        Reset all network state variables (voltages, spike states, and history) to their initial values.
        """
        self.hidden_layer_voltages.fill(0.0)
        self.output_layer_voltages.fill(0.0)
        self.hidden_layer_spike_states.fill(0)
        self.output_layer_spike_states.fill(0)
        self.hidden_voltage_history.fill(0.0)
        self.output_voltage_history.fill(0.0)
        self.output_layer_synaptic_inputs.fill(0.0)
        self.hidden_history_component.fill(0.0)
        self.output_history_component.fill(0.0)

    def reset_memory_component(self):
        """
        Reset the temporary memory components used in the integration step.
        This should be called at the beginning of each time step update.
        """
        self.hidden_history_component.fill(0.0)
        self.output_history_component.fill(0.0)

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
                self.hidden_voltage_history,
                self.gl_coefficients,
                self.neurons_bias,
                self.threshold_voltage,
                self.reset_voltage,
                self.kernel,
                self.membrane_time_constant,
                self.output_weights,
                self.output_layer_voltages,
                self.output_layer_spike_states,
                self.output_voltage_history,
                self.output_layer_synaptic_inputs,
                self.hidden_history_component,
                self.output_history_component
            )
        else:
            # Reset spike states for the current time step
            self.hidden_layer_spike_states.fill(0)
            self.output_layer_spike_states.fill(0)

            # Reset temporary memory components
            self.reset_memory_component()

            # --- Calculate memory components for hidden and output layers ---
            # This is a dot product of GL coefficients with the voltage history
            self.hidden_history_component = np.dot(self.gl_coefficients, self.hidden_voltage_history)
            self.output_history_component = np.dot(self.gl_coefficients, self.output_voltage_history)

            # --- Update Hidden Layer ---
            # Calculate the new hidden layer voltages based on the fractional LIF dynamics
            # The equation is V_new = (-V_old + bias + input) / tau * kernel - history_component
            self.hidden_layer_voltages = (
                -self.hidden_layer_voltages + self.neurons_bias + cartpole_inputs_weighted
            ) * self.kernel / self.membrane_time_constant - self.hidden_history_component

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
            self.output_layer_voltages = (
                -self.output_layer_voltages + self.neurons_bias + self.output_layer_synaptic_inputs
            ) * self.kernel / self.membrane_time_constant - self.output_history_component

            # Detect spikes in the output layer and reset voltages
            spiked_output = self.output_layer_voltages >= self.threshold_voltage
            self.output_layer_spike_states[spiked_output] = 1
            self.output_layer_voltages[spiked_output] = self.reset_voltage

            # Clear the synaptic input accumulator for the next time step
            self.output_layer_synaptic_inputs.fill(0.0)
        ####################################################################################
        ## At this point voltages and spike states are updated (numpy or cython implementation)
        # --- Update Voltage Histories ---
        # Shift all history entries down by one, effectively discarding the oldest entry
        # and making room for the newest voltage at index 0.
        self.hidden_voltage_history = np.roll(self.hidden_voltage_history, 1, axis=0)
        self.output_voltage_history = np.roll(self.output_voltage_history, 1, axis=0)
        self.hidden_voltage_history[0, :] = self.hidden_layer_voltages
        self.output_voltage_history[0, :] = self.output_layer_voltages

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
