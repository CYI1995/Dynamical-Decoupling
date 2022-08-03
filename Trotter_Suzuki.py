import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import random

def dec_to_bin(num,size):
    array_temp = np.zeros(size)

    for i in range(size):
        num_temp = num%2
        array_temp[size - 1 -i] = num_temp
        num = int(num/2)

    return array_temp

def projector(vec,dim):
    P = np.zeros((dim,dim),dtype = complex)
    for i in range(dim):
        for j in range(dim):
            P[i][j] = vec[i]*(np.conj(vec[j]))

    return P

def H_X(site):
    dim = 2**site
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,site)
        for j in range(site):
            if(bin_i[j] == 0):
                tar = int(i + 2**(site - 1 - j))
                M[i][tar] = 1
                M[tar][i] = 1
                
    return M

def H_Z(site):
    dim = 2**site
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,site)
        sum_of_one = 0
        for j in range(site):
            sum_of_one = sum_of_one + bin_i[j]
        M[i][i] = site - 2*sum_of_one

    return M

def H_ZZ(site):
    dim = 2**site
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,site)
        sum_of_diff = 0
        sum_of_same = 0
        for j in range(site-1):
            if(bin_i[j] != bin_i[j+1]):
                sum_of_diff = sum_of_diff + 1
            else:
                sum_of_same = sum_of_same + 1
        M[i][i] = sum_of_same - sum_of_diff

    return M

def matrix_norm(M,dim):
    M2 = M.dot(np.conj(M).T)
    eig,vec = np.linalg.eig(M2)

    eig_new = np.zeros(dim)
    for i in range(dim):
        eig_new[i] = abs(eig[i])

    mx = np.argmax(eig_new)
    return math.sqrt(eig_new[mx])

# Function filter gives when the pulses are applied.
def filter(j,N):
    return math.sin(j*math.pi/(2*N))**2

def Trotter_2nd(H1,H2,dt):

    return scipy.linalg.expm(-1j*H1*dt/2).dot(scipy.linalg.expm(-1j*H2*dt).dot(scipy.linalg.expm(-1j*H1*dt/2)))

dim = 4
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
XI = np.kron(X,I)
IX = np.kron(I,X)
ZZ = np.kron(Z,Z)
H0 = XI
eps = 0.1
t0 = 0.05
M = 1000
X_axis = np.zeros(M)
Y1_axis = np.zeros(M)
Y2_axis = np.zeros(M)
Y3_axis = np.zeros(M)
Y4_axis = np.zeros(M)

# U_dd is Uhrig's dynamical decoupling of a single period.
# U_id is ideal evolution of a single dt.
U_id = scipy.linalg.expm(-1j*H0*t0)

N1 = 16
U_dd1 = np.identity(dim,dtype = complex)
for j in range((N1)):
    dt = (filter(j+1,N1) - filter(j,N1))*t0
    H_temp = (H0 + ((-1)**j)*eps*ZZ)
    U_temp = scipy.linalg.expm(-1j*H_temp*dt)
    U_dd1 = U_temp.dot(U_dd1)

H_sum = (H0 + eps*ZZ)/2
H_diff = (H0 - eps*ZZ)/2
p = 1/(4 - 4**(1/3))
Trotter_4th = Trotter_2nd(H_sum,H_diff,p*t0).dot(Trotter_2nd(H_sum,H_diff,p*t0))
Trotter_4th = Trotter_2nd(H_sum,H_diff,(1-4*p)*t0).dot(Trotter_4th)
Trotter_4th = (Trotter_2nd(H_sum,H_diff,p*t0).dot(Trotter_2nd(H_sum,H_diff,p*t0))).dot(Trotter_4th)

U0 = np.identity(dim,dtype = complex)
U1 = np.identity(dim,dtype = complex)
U2 = np.identity(dim,dtype = complex)
U3 = np.identity(dim,dtype = complex)
U4 = np.identity(dim,dtype = complex)

for m in range(M):
    U0 = U_id.dot(U0)
    U1 = U_dd1.dot(U1)
    # U2 = U_dd2.dot(U2)
    # U3 = U_dd3.dot(U3)
    U4 = Trotter_4th.dot(U4)

    X_axis[m] = (m+1)*t0
    Y1_axis[m] = matrix_norm(U1 - U0,dim)
    # Y2_axis[m] = matrix_norm(U2 - U0,dim)
    # Y3_axis[m] = matrix_norm(U3 - U0,dim)
    Y4_axis[m] = matrix_norm(U4 - U0,dim)

plt.plot(X_axis,Y1_axis,label = 'Uhrig, N=16')
# plt.scatter(X_axis,Y2_axis,label = 'N=3')
# plt.scatter(X_axis,Y3_axis,label = 'N=6')
plt.plot(X_axis,Y4_axis,label = '4th Totter')
plt.legend()
plt.show()
