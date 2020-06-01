#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from numpy import genfromtxt
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
import sklearn.metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#labels = range(100)
def format_fn(tick_val, tick_pos): 
	return labels[int(tick_val)]



fs = 19

#pth = 'arth_1'; layers = [1,2,3,4,5,6,7,8]; title = 'Adversarial Dataset : layer output in 1 D'
#pth = 'arth_2'; layers = [1,2,3,4,5,6,7,8,9]; title = 'Adversarial 1 Dataset : layer output in 1 D'
#pth = 'arth_3'; layers = [1,2,3,4,5,6,7,8,9]; title = 'Adversarial Dataset\nOutput of Each layer in 1D'; title2='Adversarial Dataset'; yl=13; ms = 400
pth = 'gauss'; layers = [1,2,3,4,5,6,7,8,9,10,11,12]; title = 'Random Dataset\nOutput of 12 layers in 1D'; title2='Random Dataset in 2D'; yl=16; ms = 150
#pth = 'gauss4'; layers = [1,2,3,4,5,6,7,8,9,10,11,12]; title = 'Adversarial Dataset\nOutput of 12 layers in 1D'; title2='Adversarial'; yl=15; ms = 160


plt.subplot(122)
#plt.gcf().set_size_inches(5.0,8)		# for vertical 
plt.gcf().set_size_inches(10,5)		# for horizontal
plt.gca().spines["bottom"].set_linewidth(2)
plt.gca().spines["top"].set_linewidth(2)
plt.gca().spines["left"].set_linewidth(2)
plt.gca().spines["right"].set_linewidth(2)
plt.gca().tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5, labelsize='xx-large')
#plt.gcf().set_size_inches(80,2)		# for horizontal
for i in layers:
	z = genfromtxt(pth + '/' + str(i) + '/zℓ.csv', delimiter=',')
	l = int(len(z)/2)

	y = np.zeros(len(z[0:l])) + i
	if i == 1:
		#plt.plot(z[0:l], y, 'x', color='b', label='Class 1')
		#plt.plot(z[l:l*2], y, 'x', color='g', label='Class 2')
		plt.scatter(z[0:l], y, marker='o', s=ms, label='Class 1', facecolors='none', edgecolors='b')
		plt.scatter(z[l:l*2], y, c='g', marker='+', s=ms, label='Class 2')

	else:
		plt.scatter(z[0:l], y, marker='o', s=ms, facecolors='none', edgecolors='b')
		plt.scatter(z[l:l*2], y, c='g', marker='+', s=ms)
		#plt.plot(z[0:l], y, 'x', color='b')
		#plt.plot(z[l:l*2], y, 'x', color='g')

#plt.gca().yaxis.grid(True, linestyle='dotted')
plt.gca().yaxis.grid(True, linewidth=3, alpha=0.4)
#plt.xlabel('Sample Location in 1 Dimension', fontsize=fs-2, fontweight='bold')
#plt.ylabel('Layer ID', fontsize=fs-2, fontweight='bold')
plt.xlabel('Sample Location in 1 Dimension', fontsize=fs-1)
plt.ylabel('Layer ID', fontsize=fs-1)

plt.title(title, fontsize=fs)
#plt.title(title, fontsize=fs, fontweight='bold')
#plt.axis([-0.6, len(score)-1, -0.1, 1.05])

plt.yticks(np.arange(0, 18, 1.0))

#plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))
#plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))





legend_properties = {'weight':'bold', 'size':'xx-large'}
plt.legend(fontsize=fs, prop=legend_properties, framealpha=0.3)
plt.ylim(0, yl)

plt.subplot(121)
x = genfromtxt('../data/' + pth + '.csv', delimiter=',')
plt.gca().spines["bottom"].set_linewidth(2)
plt.gca().spines["top"].set_linewidth(2)
plt.gca().spines["left"].set_linewidth(2)
plt.gca().spines["right"].set_linewidth(2)
plt.gca().tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5, labelsize='xx-large')

plt.scatter(x[0:l, 0], x[0:l, 1], marker='o', s=ms, label='Class 1', facecolors='none', edgecolors='b')
plt.scatter(x[l:2*l, 0], x[l:l*2, 1], c='g', marker='+', s=ms, label='Class 2')
#plt.plot(x[0:l, 0], x[0:l, 1], 'x', color='b', label='Class 1')
#plt.plot(x[l:2*l, 0], x[l:l*2, 1], 'x', color='g', label='Class 2')

#plt.xlabel('Dimension 1', fontsize=fs-2, fontweight='bold')
#plt.ylabel('Dimension 2', fontsize=fs-2, fontweight='bold')
#plt.title(title2, fontsize=fs, fontweight='bold')
plt.xlabel('Dimension 1', fontsize=fs)
plt.ylabel('Dimension 2', fontsize=fs)
plt.title(title2, fontsize=fs+2)



legend_properties = {'weight':'bold', 'size':'xx-large'}
plt.legend(fontsize=fs, prop=legend_properties)









plt.tight_layout()
plt.savefig(pth + '/output_summary.png')
plt.show()




