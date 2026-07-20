import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
######################################################################################
######################################################################################
######################################################################################

def plot_fSNN(ax):
    num_inputs=2
    num_hidden=6
    num_outputs=1
    # ax.set_aspect('equal')
    ax.axis('off') # Hide the axes

    # --- Node Drawing Parameters ---
    node_radius = 0.17
    SL = 0.38
    layer_spacing = 2.0 # Horizontal spacing between layers
    node_spacing_y = 0.48 # Vertical spacing between nodes in a layer

    # --- Calculate Node Positions ---
    # Input Layer
    input_layer_x = 0
    input_layer_y_start = (num_inputs - 1) * (node_spacing_y + 0.3) / 2
    input_node_positions = []
    for i in range(num_inputs):
        y = input_layer_y_start - i * (node_spacing_y + 0.3)
        input_node_positions.append((input_layer_x, y))

    # Hidden Layer
    hidden_layer_x = input_layer_x + layer_spacing
    hidden_layer_y_start = (num_hidden - 1) * node_spacing_y / 2
    hidden_node_positions = []
    for i in range(num_hidden):
        y = hidden_layer_y_start - i * node_spacing_y
        hidden_node_positions.append((hidden_layer_x, y))

    # Output Layer
    output_layer_x = hidden_layer_x + layer_spacing
    output_layer_y_start = (num_outputs - 1) * (node_spacing_y + 0.3) / 2
    output_node_positions = []
    for i in range(num_outputs):
        y = output_layer_y_start - i * (node_spacing_y + 0.3)
        output_node_positions.append((output_layer_x, y))

    all_node_positions = {
        'input': input_node_positions,
        'hidden': hidden_node_positions,
        'output': output_node_positions
    }

    color_output = ["#a3be89", 'b']
    Output_labels = ['L/R', 'R']
    Input_labels = [r'${x}$', r'${\theta}$']
    # --- Draw Nodes and Labels ---
    for layer_name, positions in all_node_positions.items():
        for i, (x, y) in enumerate(positions):
            color = 'w' if layer_name == 'input' else '0.82'
            if layer_name == 'input':
                circle = plt.Rectangle((x-SL/2, y-SL/2), SL, SL, color=color, ec='black', lw=1.5, zorder=0)
                ax.text(x, y, Input_labels[i], ha='center', va='center', fontsize=19, weight='bold')
            elif layer_name == 'hidden':
                ax.text(x, y, r'$fLIF_{%d}$'%(i+1), ha='center', va='center', fontsize=8)
                circle = plt.Circle((x, y), node_radius + 0.02, color=color, ec='black', lw=1., zorder=0)
            else: # Output layer
                ax.text(x, y, Output_labels[i], ha='center', va='center', fontsize=13, weight='bold')
                circle = plt.Circle((x, y), node_radius+0.05, color=color_output[i], ec='black', lw=1.5, zorder=0)
            ax.add_patch(circle)

    # --- Draw Edges using annotate ---
    # Input to Hidden connections
    for i_node_pos in input_node_positions:
        for h_node_pos in hidden_node_positions:
            xfrom, yfrom, xto, yto = i_node_pos[0] + 0.05, i_node_pos[1], h_node_pos[0], h_node_pos[1]
            length = np.sqrt((yfrom - yto)**2 + (xfrom - xto)**2)
            xytext = (xfrom + node_radius, yfrom)
            xy = (xto - 1.1*node_radius * (xto - xfrom)/ length, yto - 1.1 *node_radius * (yto - yfrom)/length)
            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(arrowstyle='-', color='#b15928', lw=0.8, shrinkA=node_radius*1.1, shrinkB=node_radius*1.1))

    aspan = 120 
    ainit = (180 - aspan/2) /180 * np.pi
    astep = aspan/(num_hidden-1) /180 * np.pi

    # Hidden to Output connections
    for hindx, h_node_pos in enumerate(hidden_node_positions):
        for oindx, o_node_pos in enumerate([output_node_positions[-1]]): # Now iterating through all output nodes
            xytext = (h_node_pos[0] + node_radius, h_node_pos[1])
            angle = ainit + hindx * astep
            x = o_node_pos[0] + (1.2 *node_radius) * np.cos(angle)
            y = o_node_pos[1] + (1.2 *node_radius) * np.sin(angle)
            xy = (x, y)

            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(arrowstyle='-|>', color='#6a3d9a', lw=1.0))

    # Adjust plot limits
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.6, 1.6)

######################################################################################
######################################################################################
######################################################################################
def plot_SNN(ax):
    num_inputs=2
    num_hidden=12
    num_outputs=1

    # ax.set_aspect('equal')
    ax.axis('off') # Hide the axes

    # --- Node Drawing Parameters ---
    node_radius = 0.17
    SL = 0.38
    layer_spacing = 2.0 # Horizontal spacing between layers
    node_spacing_y = 0.27 # Vertical spacing between nodes in a layer

    # --- Calculate Node Positions ---
    # Input Layer
    input_layer_x = 0
    input_layer_y_start = (num_inputs - 1) * (node_spacing_y + 0.3) / 2
    input_node_positions = []
    for i in range(num_inputs):
        y = input_layer_y_start - i * (node_spacing_y + 0.5)
        input_node_positions.append((input_layer_x, y))

    # Hidden Layer
    hidden_layer_x = input_layer_x + layer_spacing
    hidden_layer_y_start = (num_hidden - 1) * node_spacing_y / 2
    hidden_node_positions = []
    for i in range(num_hidden):
        y = hidden_layer_y_start - i * node_spacing_y
        hidden_node_positions.append((hidden_layer_x, y))

    # Output Layer
    output_layer_x = hidden_layer_x + layer_spacing
    output_layer_y_start = (num_outputs - 1) * (node_spacing_y + 0.3) / 2
    output_node_positions = []
    for i in range(num_outputs):
        y = output_layer_y_start - i * (node_spacing_y + 0.5)
        output_node_positions.append((output_layer_x, y))

    all_node_positions = {
        'input': input_node_positions,
        'hidden': hidden_node_positions,
        'output': output_node_positions
    }

    color_output = ["#a3be89", 'b']

    Output_labels = ['L/R', 'R']
    Input_labels = [r'${x}$', r'${\theta}$']
    # --- Draw Nodes and Labels ---
    for layer_name, positions in all_node_positions.items():
        for i, (x, y) in enumerate(positions):
            color = 'w' if layer_name == 'input' else '0.82'
            if layer_name == 'input':
                circle = plt.Rectangle((x-SL/2, y-SL/2), SL, SL, color=color, ec='black', lw=1.5, zorder=0)
                ax.text(x, y, Input_labels[i], ha='center', va='center', fontsize=19, weight='bold')
            elif layer_name == 'hidden':
                ax.text(x, y, r'$LIF_{%d}$'%(i+1), ha='center', va='center', fontsize=6)
                circle = plt.Circle((x, y), node_radius-0.045, color=color, ec='black', lw=0.5, zorder=0)
            else: # Output layer
                ax.text(x, y, Output_labels[i], ha='center', va='center', fontsize=13, weight='bold')
                circle = plt.Circle((x, y), node_radius+0.05, color=color_output[i], ec='black', lw=1.5, zorder=0)
            ax.add_patch(circle)

    # --- Draw Edges using annotate ---
    # Input to Hidden connections
    for i_node_pos in input_node_positions:
        for h_node_pos in hidden_node_positions:
            xfrom, yfrom, xto, yto = i_node_pos[0] + 0.05, i_node_pos[1], h_node_pos[0], h_node_pos[1]
            length = np.sqrt((yfrom - yto)**2 + (xfrom - xto)**2)
            xytext = (xfrom + node_radius, yfrom)
            xy = (xto - 1.1*node_radius * (xto - xfrom)/ length, yto - 1.1 *node_radius * (yto - yfrom)/length)
            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(arrowstyle='-', color='#b15928', lw=0.8, shrinkA=node_radius*1.1, shrinkB=node_radius*1.1))

    aspan = 120 
    ainit = (180 - aspan/2) /180 * np.pi
    astep = aspan/(num_hidden-1) /180 * np.pi

    # Hidden to Output connections
    for hindx, h_node_pos in enumerate(hidden_node_positions):
        for oindx, o_node_pos in enumerate([output_node_positions[-1]]): # Now iterating through all output nodes
            xytext = (h_node_pos[0] + node_radius - 0.04, h_node_pos[1])
            angle = ainit + hindx * astep
            x = o_node_pos[0] + (1.2 *node_radius) * np.cos(angle)
            y = o_node_pos[1] + (1.2 *node_radius) * np.sin(angle)
            xy = (x, y)

            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(arrowstyle='-|>', color='#6a3d9a', lw=1.0))

    # Adjust plot limits
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.7, 1.7)
