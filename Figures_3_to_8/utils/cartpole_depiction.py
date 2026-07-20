import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

##############################################################################
##############################################################################

def plot_cartpole(ax):
	theta = -0.24
	x_cart = 0.04

	cart_width, cart_height = 0.28, 0.18
	pole_width, pole_length = 0.05, 0.8
	track_length = 4.8


	# Set the limits of the plot to match the environment's scale
	ax.set_xlim(-track_length / 2, track_length / 2)
	ax.set_ylim(-0.5, 1.25)
	ax.set_aspect('equal')
	# ax.axis('off')

	# Calculate cart and pole positions
	cart_x_min = x_cart - cart_width / 2
	cart_y_min = - cart_height/2
	pole_x_end = x_cart + pole_length * np.sin(theta)
	pole_y_end = pole_length * np.cos(theta)

	# Draw the track
	ax.hlines(0, -track_length / 2, track_length / 2, color='black', linewidth=1.2)

	# Draw the cart
	cart = Rectangle((cart_x_min, cart_y_min), cart_width, cart_height, facecolor='0.08')
	ax.add_patch(cart)

	# Draw the pole
	ax.plot([x_cart, pole_x_end], [0, pole_y_end], color='red', linewidth=6, solid_capstyle='round')

	##############################################################################
	##############################################################################
	# x arrow
	y = -0.15
	xytext = (0.5, y)
	xy = (1, y)
	ax.annotate('', xy=xy ,xytext=xytext, arrowprops=dict(arrowstyle='->', color='k', lw=2.2))

	ax.text(0.4, y, r'${x}$', fontsize = 21, ha = 'center', va = 'center')

	# angle
	r = pole_length
	xytext = (-0.25, 0.82)
	xy = (0.045, 0.9)

	style = "Simple, tail_width=0.5, head_width=4, head_length=8"
	kw = dict(arrowstyle=style, color="k")
	a = patches.FancyArrowPatch(xy, xytext, connectionstyle="arc3,rad=.3", **kw)
	ax.add_patch(a)
	ax.plot([0.04, 0.04], [0, 0.85], '--k', lw = 0.35, zorder = 0)

	ax.text(-0.04, 0.75, r'${\theta}$', fontsize = 21, ha = 'center', va = 'center')

	ax.set_xlim(-1.2, 1.2)
	ax.set_ylim(-0.3, 1.)
	ax.axis('off')

