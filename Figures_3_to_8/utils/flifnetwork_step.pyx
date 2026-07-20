
# fSNN step of integration
# compile with python setup.py build_ext --inplace

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def update_state(np.ndarray[np.float64_t, ndim=1] cartpole_inputs,
                 np.ndarray[np.float64_t, ndim=1] hidden_layer_voltages,
                 np.ndarray[np.int32_t, ndim=1] hidden_layer_spike_states,
                 np.ndarray[np.float64_t, ndim=2] hidden_voltage_history,
                 np.ndarray[np.float64_t, ndim=1] gl_coefficients,
                 double neurons_bias, double threshold_voltage,
                 double reset_voltage, double kernel, double tau,
                 np.ndarray[np.float64_t, ndim=2] output_weights,
                 np.ndarray[np.float64_t, ndim=1] output_layer_voltages,
                 np.ndarray[np.int32_t, ndim=1] output_layer_spike_states,
                 np.ndarray[np.float64_t, ndim=2] output_voltage_history,
                 np.ndarray[np.float64_t, ndim=1] output_layer_synaptic_inputs, 
                 np.ndarray[np.float64_t, ndim=1] hidden_history_component,
                 np.ndarray[np.float64_t, ndim=1] output_history_component):
    cdef int i, j
    cdef int h_size = hidden_layer_voltages.shape[0]
    cdef int o_size = output_layer_voltages.shape[0]
    cdef int mem_len = gl_coefficients.shape[0]
    cdef double h_mem, o_mem

    # Compute hidden memory component and update voltages
    for i in range(h_size):
        h_mem = 0.0
        for j in range(mem_len):
            h_mem += gl_coefficients[j] * hidden_voltage_history[j, i]

        hidden_history_component[i] = h_mem

        hidden_layer_voltages[i] = (
            (-hidden_layer_voltages[i] + neurons_bias + cartpole_inputs[i]) * kernel / tau
            - h_mem
        )
        if hidden_layer_voltages[i] >= threshold_voltage:
            hidden_layer_spike_states[i] = 1
            hidden_layer_voltages[i] = reset_voltage
            for j in range(o_size):
                output_layer_synaptic_inputs[j] += output_weights[j, i]
        else:
            hidden_layer_spike_states[i] = 0

    # Compute output memory component and update voltages
    for i in range(o_size):
        o_mem = 0.0
        for j in range(mem_len):
            o_mem += gl_coefficients[j] * output_voltage_history[j, i]
        
        output_history_component[i] = o_mem

        output_layer_voltages[i] = (
            (-output_layer_voltages[i] + neurons_bias + output_layer_synaptic_inputs[i]) * kernel / tau
            - o_mem
        )
        if output_layer_voltages[i] >= threshold_voltage:
            output_layer_spike_states[i] = 1
            output_layer_voltages[i] = reset_voltage
        else:
            output_layer_spike_states[i] = 0

    # Clear synaptic inputs
    for i in range(o_size):
        output_layer_synaptic_inputs[i] = 0.0
