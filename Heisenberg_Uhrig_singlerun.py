import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs

def concatenated(U,P):

    return P.dot(U).dot(P.dot(U))

def trace_distance(M,dim):

    M_temp = M.dot(np.conj(M).T)
    sum_of_trace = 0
    for i in range(dim):
        sum_of_trace = sum_of_trace + math.sqrt(M_temp[i][i].real)

    return (sum_of_trace/2)

def partial_trace(M,dim1,dim2):

    M_temp = np.zeros([dim1,dim1],dtype = 'c16')
    for i1 in range(dim1):
        digit1 = i1*(dim2)
        for i2 in range(dim1):
            digit2 = i2*(dim2)
            entry = 0
            for index in range(dim2):
                real_index1 = digit1 + index 
                real_index2 = digit2 + index 
                entry = entry + M[real_index1][real_index2]
                    # print(i1,i2,real_index1,real_index2)
            M_temp[i1][i2] = entry

    return M_temp

def projector(vec,dim):
    P = np.zeros((dim,dim),dtype = 'c16')
    for i in range(dim):
        for j in range(dim):
            P[i][j] = vec[i]*(np.conj(vec[j]))

    return P

def filter(j,N):
    return math.sin(j*math.pi/(2*N +2))**2

def rand_vec(dim):

    vec = np.zeros(dim)
    rand_vec = np.random.normal(0,1,dim)

    norm = math.sqrt(abs(np.vdot(rand_vec,rand_vec)))

    for i in range(dim):
        vec[i] = rand_vec[i]/norm 

    return vec

dim = 8

X = np.array([[0,1],[1,0]],dtype = 'f8')
Z = np.array([[1,0],[0,-1]],dtype = 'f8')
I = np.array([[1,0],[0,1]],dtype = 'f8')
Y = 1j*X.dot(Z)

w0 = 1
b1 = 2
b2 = 2
w1 = 1
w2 = 1
d1 = 2
d2 = 3
d3 = 4

# HS = w0*np.kron(X,np.kron(I,I))
HS = 0
HSE = b1*np.kron(Z,np.kron(Z,I)) + b2*np.kron(Z,np.kron(I,Z))
HE = w1*np.kron(I,np.kron(Z,I)) + w2*np.kron(I,np.kron(I,Z))
HE = HE + d1*np.kron(I,np.kron(Z,Z)) + d2*np.kron(I,np.kron(X,X)) + d3*np.kron(I,np.kron(Y,Y))

H = HS + HSE + HE
H0 = HS + HE
pulse = np.kron(X,np.kron(I,I))

M = 10
T = 0.1
U_ideal = scipy.linalg.expm(-1j*H0*T)
vec_temp = rand_vec(2)
init_ket = np.array([vec_temp[0],0,0,0,vec_temp[1],0,0,0])
# init_ket = rand_vec(dim)
vec_ideal = U_ideal.dot(init_ket)
proj_ideal = projector(vec_ideal,dim)

X_axis = np.zeros(M)
Y1_axis = np.zeros(M)
Y2_axis = np.zeros(M)
for m in range(M):
    Num_of_U = m

    U_dd1 = np.identity(dim,dtype = 'c16')
    U_dd2 = np.identity(dim,dtype = 'c16')
    for j in range((Num_of_U+1)):
        dt = (filter(j+1,Num_of_U) - filter(j,Num_of_U))*T
        H_temp = H0 + ((-1)**j)*HSE
        U_temp = scipy.linalg.expm(-1j*H_temp*dt)
        H_temp_2 = H0 + ((-1)**(j+1))*HSE
        U_temp_2 = scipy.linalg.expm(-1j*H_temp_2*dt)
        U_dd1 = U_temp.dot(U_dd1)
        U_dd2 = U_temp_2.dot(U_dd2)
 
    vec_1 = U_dd1.dot(init_ket)
    proj_1 = projector(vec_1,dim)

    X_axis[m] = m+1
    Y1_axis[m] = srs.matrix_norm((U_dd1 - U_ideal),dim)
    # Y1_axis[m] = srs.matrix_norm((U_dd2 - U_dd1),dim)
    Y2_axis[m] = trace_distance(partial_trace(proj_1 - proj_ideal,2,4),2)


plt.plot(X_axis,Y1_axis,label = 'entire space',marker = '.')
plt.plot(X_axis,Y2_axis,label = 'subspace',marker = '.')
plt.loglog()
# plt.ylim(0,1)
plt.legend()
plt.show()