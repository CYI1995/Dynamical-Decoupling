import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs

def concatenated(U,XX,ZZ):

    ZU = ZZ.dot(U)
    XU = XX.dot(U)
    CDD = (XU.dot(ZU)).dot(XU.dot(ZU))
    return CDD

def KL(H,XX,YY,ZZ):

    H0 = H 
    H1 = ZZ.dot(H.dot(ZZ))
    H2 = YY.dot(H.dot(YY))
    H3 = XX.dot(H.dot(XX))

    return H0 + H1 + H2 + H3


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

A1 = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])
A2 = np.array([1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1])
A3 = np.array([1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1])
A4 = np.array([1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1])
L = 8

for k in range(N):
    t0 = 0.5**(k+1)
    X_axis[k] = t0

    dt = t0/L
    U_id = scipy.linalg.expm(-1j*H0*t0)
    CDD1 = np.identity(dim,dtype = complex)
    for i in range(L):
        H_temp = H0 + eps*A1[i]*ZZ
        U_temp = scipy.linalg.expm(-1j*H_temp*dt)
        CDD1 = U_temp.dot(CDD1)
    CDD2 = np.identity(dim,dtype = complex)
    for i in range(L):
        H_temp = H0 + eps*A2[i]*ZZ
        U_temp = scipy.linalg.expm(-1j*H_temp*dt)
        CDD2 = U_temp.dot(CDD2)
    CDD3 = np.identity(dim,dtype = complex)
    for i in range(L):
        H_temp = H0 + eps*A3[i]*ZZ
        U_temp = scipy.linalg.expm(-1j*H_temp*dt)
        CDD3 = U_temp.dot(CDD3)

    U0 = np.identity(dim,dtype = complex)
    U1 = np.identity(dim,dtype = complex)
    U2 = np.identity(dim,dtype = complex)
    U3 = np.identity(dim,dtype = complex)
    for m in range(M):
        U0 = U_id.dot(U0)
        U1 = CDD1.dot(U1)
        U2 = CDD2.dot(U2)
        U3 = CDD3.dot(U3)

    Y1_axis[k] = srs.matrix_norm(U1 - U0,dim)
    Y2_axis[k] = srs.matrix_norm(U2 - U0,dim)
    Y3_axis[k] = srs.matrix_norm(U3 - U0,dim)


plt.scatter(X_axis,Y1_axis,label = 'CDD1')
plt.scatter(X_axis,Y2_axis,label = 'CDD2')
plt.scatter(X_axis,Y3_axis,label = 'CDD3')
plt.loglog()
plt.legend()
plt.show()
