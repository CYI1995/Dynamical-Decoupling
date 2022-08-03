
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

N = 100
S1 = np.linspace(0,1,N)
S2 = np.linspace(1,2,N)
S3 = np.linspace(2,3,N)
S4 = np.linspace(3,4,N)
Y1 = np.zeros(N)
Y2 = np.zeros(N)
Y3 = np.zeros(N)
Y4 = np.zeros(N)
F1 = np.zeros(N)
F2 = np.zeros(N)
F3 = np.zeros(N)
F4 = np.zeros(N)
FF1 = np.zeros(N)
FF2 = np.zeros(N)
FF3 = np.zeros(N)
FF4 = np.zeros(N)
for i in range(N):
    x = (i+1)/N
    Y1[i] = 1
    Y2[i] = -1 
    Y3[i] = 1
    Y4[i] = -1
    F1[i] = x
    F2[i] = 1-x
    F3[i] = x
    F4[i] = 1-x
    FF1[i] = (x)**2/2
    FF2[i] = 1 - (x - 1)**2/2
    FF3[i] = 1 - (x)**2/2 
    FF4[i] = (1 - x)**2/2

vert_x1 = np.zeros(20)
vert_x2 = np.zeros(20)
vert_x3 = np.zeros(20)
vert_y = np.zeros(20)
for l in range(20):
    vert_x1[l] = 1
    vert_x2[l] = 2 
    vert_x3[l] = 3
    vert_y[l] = l/9.5 - 1


# fig, axs = plt.subplot_mosaic([['a)','b)']],
#                               constrained_layout=True)

# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
#             fontsize='medium', va='bottom', fontfamily='serif')
#     if(label == 'a)'):
#         ax.set_xlabel('$t$')
#         ax.set_ylabel('$PDD$')
#         ax.plot(S1,Y1, c = "k")
#         ax.plot(S2,Y2, c = "k")
#         ax.plot(S3,Y1, c = "k")
#         ax.plot(S4,Y2, c = "k")
#         ax.plot(vert_x1,vert_y,linestyle = '-.',c = "r")
#         ax.plot(vert_x2,vert_y,linestyle = '-.',c = "r")
#         ax.plot(vert_x3,vert_y,linestyle = '-.',c = "r")
#         # ax.legend(loc = 1)
#     if(label == 'b)'):
#         ax.set_xlabel('$t$')
#         ax.set_ylabel('$CDD$')
#         ax.plot(S1,Y1, c = "k")
#         ax.plot(S2,Y2, c = "k")
#         ax.plot(S3,Y3, c = "k")
#         ax.plot(S4,Y4, c = "k")
#         ax.plot(vert_x1,vert_y,linestyle = '-.',c = "r")
#         ax.plot(vert_x2,vert_y,linestyle = '-.',c = "r")
#         ax.plot(vert_x3,vert_y,linestyle = '-.',c = "r")
#         # ax.legend(loc = 1)

fig, axs = plt.subplot_mosaic([['a)'], ['b)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == 'a)'):
        ax.set_xlabel('$t/t_0$')
        ax.set_ylabel('$\hat{H}_{SB}(t)/\lambda$')
        ax.plot(S1,Y1, c = "k")
        ax.plot(S2,Y2, c = "k")
        ax.plot(S3,Y3, c = "k")
        ax.plot(S4,Y4, c = "k")
        ax.plot(vert_x1,vert_y,linestyle = '-.',c = "r")
        ax.plot(vert_x2,vert_y,linestyle = '-.',c = "r")
        ax.plot(vert_x3,vert_y,linestyle = '-.',c = "r")
        # ax.legend(loc = 1)
    if(label == 'b)'):
        ax.set_xlabel('$t/t_0$')
        ax.set_ylabel('$S(t)/\lambda t_0$')
        ax.plot(S1,F1, c = "k")
        ax.plot(S2,F2, c = "k")
        ax.plot(S3,F3, c = "k")
        ax.plot(S4,F4, c = "k")
        ax.plot(vert_x1,vert_y,linestyle = '-.',c = "r")
        ax.plot(vert_x2,vert_y,linestyle = '-.',c = "r")
        ax.plot(vert_x3,vert_y,linestyle = '-.',c = "r")
        # ax.legend(loc = 1)
    # if(label == 'c)'):
    #     ax.set_xlabel('$t$')
    #     ax.set_ylabel('$S^{(2)}(t)/\lambda^2$')
    #     ax.plot(S1,FF1, c = "k")
    #     ax.plot(S2,FF2, c = "k")
    #     ax.plot(S3,FF3, c = "k")
    #     ax.plot(S4,FF4, c = "k")
    #     ax.plot(vert_x1,vert_y,linestyle = '-.',c = "r")
    #     ax.plot(vert_x2,vert_y,linestyle = '-.',c = "r")
    #     ax.plot(vert_x3,vert_y,linestyle = '-.',c = "r")
    #     # ax.legend(loc = 1)

plt.show()
