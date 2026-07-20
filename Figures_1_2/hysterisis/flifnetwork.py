import numpy as np

class FractionalLIFNeuron:
    """
    A fractional order Leaky Integrate-and-Fire (LIF) neural neuron
    """
    def __init__(self,
                 membrane_time_constant=20.0, neurons_bias=2.0,
                 threshold_voltage=20.0, reset_voltage=0.0,
                 fractional_order=0.9, memory_length=120,
                 dt = 0.1):
        # Network dimensions
        # Neuron state variables
        self.voltage = 0
        self.spike_state = 0
        self.spike_interp = 0 ## store the fraction of dt given vthreshold overshoot

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
        self.voltage_history = np.zeros(memory_length, dtype=np.float64)
        # Components for the history term in the integration step
        self.history_component = 0

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
        Reset all network state variables (voltage, spike states, and history) to their initial values.
        """
        self.voltage = 0
        self.spike_state = 0
        self.voltage_history.fill(0.0)
        self.history_component = 0

    def reset_memory_component(self):
        """
        Reset the temporary memory components used in the integration step.
        This should be called at the beginning of each time step update.
        """
        self.history_component = 0

    def update(self, input, dt=0.10):
        """
        Update the network state for one time step using fractional order dynamics.
        The update rule for voltage $V$ is:
        $V(t) = \frac{1}{\tau}(-V(t-\Delta t) + I_{bias} + I_{syn}) (\Delta t)^\alpha - \sum_{j=1}^{L} c_j^{(\alpha)} V(t-j\Delta t)$
        """
        ### if spikes, one step of refractory period
        if self.spike_state:
            self.voltage = self.reset_voltage
            self.spike_state = 0
            self.history_component = np.dot(self.gl_coefficients, self.voltage_history)
        else:
            # Reset temporary memory components
            self.reset_memory_component()

            # --- Calculate memory components for hidden and output layers ---
            # This is a dot product of GL coefficients with the voltage history
            self.history_component = np.dot(self.gl_coefficients, self.voltage_history)

            # The equation is V_new = (-V_old + bias + input) / tau * kernel - history_component
            self.voltage = (
                -self.voltage + self.neurons_bias + input
            ) * self.kernel / self.membrane_time_constant - self.history_component

            # Detect spikes in the hidden layer and reset voltage of spiking hidden neurons
            self.spike_state = self.voltage >= self.threshold_voltage
            if self.spike_state:
                self.spike_interp = dt * (self.voltage - self.threshold_voltage)/ (self.voltage - self.voltage_history[0])
                self.voltage = self.reset_voltage
        ####################################################################################
        self.voltage_history = np.roll(self.voltage_history, 1, axis=0)
        self.voltage_history[0] = self.voltage


