import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt


# Function filter gives when the pulses are applied.
def filter(j,N):
    return math.sin(j*math.pi/(2*N +2))**2

dim = 2
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])

# t0 is the length of a single period. 
# eps is the size of decoherence.
# Mt_0 is the total simulation time.
# N + 1 is the order of Uhrig's dynamical decoupling.
H0 = X
t0 = 0.1
eps = 0.001
H = H0 + eps*Z
M = 1000
X_axis = np.zeros(M)
Y_axis = np.zeros(M)
N = 3

# U_dd is Uhrig's dynamical decoupling of a single period.
# U_id is ideal evolution of a single period.
U_dd = np.identity(dim,dtype = complex)
for j in range((N+1)):
    dt = (filter(j+1,N) - filter(j,N))*t0
    U_temp = X.dot(scipy.linalg.expm(-1j*H*dt))
    U_dd = U_temp.dot(U_dd)
U_id = scipy.linalg.expm(-1j*H0*t0)


# U1 is accumulation of U_dd, U2 is accumulation of U_id.
U1 = np.identity(dim,dtype = complex)
U2 = np.identity(dim,dtype = complex)
for m in range(M):
    U1 = U_dd.dot(U1)
    U2 = U_id.dot(U2)

    X_axis[m] = (m+1)*t0
    Y_axis[m] = srs.matrix_norm(U1 - U2,dim)

# The error scaling is revealed by ploting \|U_1 - U_2\| versus time
plt.scatter(X_axis,Y_axis,label = 'dd')
plt.legend()
plt.show()



