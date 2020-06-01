#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
import sklearn.metrics
import matplotlib.pyplot as plt

# Gauss 1
#ℍᶩ = np.array([8.6829 ,17.3074 ,19.5493 ,23.9346 ,30.5797 ,420.076 ,844.156 ,869.907 ,870.000])
#σ = np.array([0.5955 ,0.0214 ,0.0321 ,0.1500 ,0.1500 ,0.0000 ,0.1500 ,0.1500 ,0.1500])

## Gauss 2
#ℍᶩ = np.array([ 69.3451 ,72.1495 ,77.5979 ,89.3722 ,89.9944 ,90.0000]) 
#σ = np.array([ 1.4142 ,0.1500 ,0.0000 ,0.1500 ,0.1500 ,0.1500])

##Arth 1
#ℍᶩ = np.array([ -4.0086 ,-4.0086 ,3.8161 ,7.4152 ,11.8690 ,15.8995 ,18.6640 ,19.8758 ,19.8920])
#σ = np.array([ 0.2233 ,0.2233  ,0.1071  ,0.3000  ,0.3000  ,0.3000  ,0.3000  ,0.3000  ,0.3000])

#Arth 2
#ℍᶩ = np.array([ -4.0173 ,3.5470 ,4.7757 ,7.4705 ,11.4484 ,16.8952 ,19.6929 ,19.8920 ,19.8937])
#σ = np.array([0.2977 ,0.064 ,0.300 ,0.300 ,0.300 ,0.300 ,0.300 ,0.300 ,0.300])


##Arth 3
#ℍᶩ = np.array([3.2781 ,6.6876 ,12.628 ,18.312 ,19.724 ,19.904 ,19.904])
#σ = np.array([0.074, 0.30, 0.30, 0.30, 0.30, 0.30, 0.300])

data = 'Random'
ℍᶩ = np.array([8.044 ,18.76 ,29.72 ,41.06 ,43.73 ,45.2 ,57.85 ,80.20 ,439.9 ,834.8 ,869.4 ,869.9])
σ = np.array([ 0.595 ,0.032 ,0.150 ,0.150 ,0.150 ,0.150 ,0.075 ,0.150 ,0.001 ,0.150 ,0.150 ,0.150])


#data = 'Adversarial'
#ℍᶩ = np.array([ 39.99 ,49.93 ,59.90 ,69.29 ,78.97 ,616.5 ,1476.5 ,1551.9 ,1559.9 ,1560.0 ,1560.0 ,1560.0])
#σ = np.array([ 0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15 ,0.15])






Xa = np.arange(1,ℍᶩ.shape[0]+1)
plt.subplot(121)
plt.plot(Xa, ℍᶩ)
plt.xlabel('Layer ID')
plt.ylabel('HSIC after the layer')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title(data + ': HSIC at each Layer')
plt.tight_layout()
#plt.axhline(y=0, color='k')
#plt.axvline(x=0, color='k')

plt.subplot(122)
plt.plot(Xa, σ)
plt.xlabel('Layer ID')
plt.ylabel('σ value used')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title(data + ': σ at each Layer')
#plt.axhline(y=0, color='k')
#plt.axvline(x=0, color='k')


#plt.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.295, wspace=0.215)
plt.tight_layout()
plt.show()

