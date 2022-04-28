import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs
import bacon_shor as bs

def matrix_norm(M,dim):
    eig,vec = np.linalg.eig(M)

    eig_new = np.zeros(dim)
    for i in range(dim):
        eig_new[i] = abs(eig[i])

    mx = np.argmax(eig_new)
    return eig_new[mx]

# Function filter gives when the pulses are applied.
def filter(j,N):
    return math.sin(j*math.pi/(2*N +2))**2

dim = 4
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
XI = np.kron(X,I)
IX = np.kron(I,X)
ZZ = np.kron(Z,Z)

# t0 is the length of a single period. 
# eps is the size of decoherence.
# Mt_0 is the total simulation time.
# N + 1 is the order of Uhrig's dynamical decoupling.
H0 = IX
eps = 0.1
H = H0 + eps*ZZ
M = 100
N = 10
X_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)
Y3_axis = np.zeros(N)

# U_dd is Uhrig's dynamical decoupling of a single period.
# U_id is ideal evolution of a single period.
for i in range(N):
    t0 = 0.5**(i+1)

    U_id = scipy.linalg.expm(-1j*H0*t0)
    N1 = 3
    U_dd1 = np.identity(dim,dtype = complex)
    for j in range((N1+1)):
        dt = (filter(j+1,N1) - filter(j,N1))*t0
        U_temp = XI.dot(scipy.linalg.expm(-1j*H*dt))
        # H_temp = H0 + ((-1)**j)*eps*ZZ
        # U_temp = scipy.linalg.expm(-1j*H_temp*dt)
        U_dd1 = U_temp.dot(U_dd1)
    N2 = 5
    U_dd2 = np.identity(dim,dtype = complex)
    for j in range((N2+1)):
        dt = (filter(j+1,N2) - filter(j,N2))*t0
        U_temp = XI.dot(scipy.linalg.expm(-1j*H*dt))
        U_dd2 = U_temp.dot(U_dd2)
    N3 = 7
    U_dd3 = np.identity(dim,dtype = complex)
    for j in range((N3+1)):
        dt = (filter(j+1,N3) - filter(j,N3))*t0
        U_temp = XI.dot(scipy.linalg.expm(-1j*H*dt))
        U_dd3 = U_temp.dot(U_dd3)

    # U1 is accumulation of U_dd, U2 is accumulation of U_id.
    U0 = np.identity(dim,dtype = complex)
    U1 = np.identity(dim,dtype = complex)
    U2 = np.identity(dim,dtype = complex)
    U3 = np.identity(dim,dtype = complex)
    for m in range(M):
        U0 = U_id.dot(U0)
        U1 = U_dd1.dot(U1)
        U2 = U_dd2.dot(U2)
        U3 = U_dd3.dot(U3)

    X_axis[i] = t0
    Y1_axis[i] = matrix_norm(U1 - U0,dim)
    Y2_axis[i] = matrix_norm(U2 - U0,dim)
    Y3_axis[i] = matrix_norm(U3 - U0,dim)

# The error scaling is revealed by ploting \|U_1 - U_2\| versus time
plt.scatter(X_axis,Y1_axis,label = 'N=3')
plt.scatter(X_axis,Y2_axis,label = 'N=5')
plt.scatter(X_axis,Y3_axis,label = 'N=7')
plt.loglog()
plt.legend()
plt.show()
