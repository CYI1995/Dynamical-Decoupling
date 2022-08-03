
import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import string

def sparse(array,L,k):

    M = int(L/k)
    new_array = np.zeros(M)
    for j in range(M):
        idx = (j+1)*k
        new_array[j] = array[idx-1]
    return new_array

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

X0 = np.load('time_steps2_err2.npy')
X1 = np.load('time_steps3_err2.npy')
X2 = np.load('time_steps2_err1.npy')
X3 = np.load('time_steps3_err1.npy')
Y0 = np.load('CDD2_err2.npy')
Y1 = np.load('CDD3_err2.npy')
Y2 = np.load('CDD2_err1.npy')
Y3 = np.load('CDD3_err1.npy')



fig, axs = plt.subplot_mosaic([['a)'], ['b)']],
                              constrained_layout=True)

# fig, axs = plt.subplot_mosaic([['a)','b)']],
#                               constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == 'a)'):
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\eta$')
        ax.plot(X2,Y2,"--",c = "r",label = 'CDD2')
        ax.plot(X3,Y3,c = "b",label = 'CDD3')
        ax.set_ylim(0,0.000005)
        ax.legend(loc = 2)
    if(label == 'b)'):
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\eta$')
        ax.plot(X0,Y0,"--",c = "r",label = 'CDD2')
        ax.plot(X1,Y1,c = "b",label = 'CDD3')
        ax.set_ylim(0,0.00000005)
        ax.legend(loc = 2)

plt.show()