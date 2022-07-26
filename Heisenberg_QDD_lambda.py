import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs

def concatenated(U,P):

    return P.dot(U).dot(P.dot(U))

# def concatenated(U,P1,P2):

#     U_temp = P1.dot(U).dot(P2.dot(U))

#     return U_temp.dot(U_temp)

def trace_distance(M,dim):

    M_temp = M.dot(np.conj(M).T)
    sum_of_trace = 0
    for i in range(dim):
        sum_of_trace = sum_of_trace + math.sqrt(M_temp[i][i].real)

    return (sum_of_trace/2)

def partial_trace(M,dim1,dim2,sig):

    if(sig==0):

        M_temp = np.zeros([dim1,dim1],dtype = complex)
        for i1 in range(dim1):
            digit1 = i1*(dim2)
            for i2 in range(dim1):
                digit2 = i2*(dim2)
                entry = 0
                for index in range(dim2):
                    real_index1 = digit1 + index 
                    real_index2 = digit2 + index 
                    entry = entry + M[real_index1][real_index2]
                M_temp[i1][i2] = entry

    else:

        M_temp = np.zeros([dim2,dim2],dtype = complex)
        for i1 in range(dim2):
            digit1 = i1
            for i2 in range(dim2):
                digit2 = i2
                entry = 0
                for index in range(dim1):
                    real_index1 = digit1 + index*(dim2)
                    real_index2 = digit2 + index*(dim2) 
                    entry = entry + M[real_index1][real_index2]
                M_temp[i1][i2] = entry


    return M_temp

def projector(vec,dim):
    P = np.zeros((dim,dim),dtype = complex)
    for i in range(dim):
        for j in range(dim):
            P[i][j] = vec[i]*(np.conj(vec[j]))

    return P

def rand_vec(dim):

    vec = np.zeros(dim)
    rand_vec = np.random.normal(0,1,dim)

    norm = math.sqrt(abs(np.vdot(rand_vec,rand_vec)))

    for i in range(dim):
        vec[i] = rand_vec[i]/norm 

    return vec

def filter(j,N):
    return math.sin(j*math.pi/(2*N +2))**2

def Uhrig(pulse,N,t,H,dim):

    U_dd1 = np.identity(dim,dtype = 'c16')
    for j in range((N+1)):
        dt = (filter(j+1,N) - filter(j,N))*t
        U_temp = scipy.linalg.expm(-1j*H*dt)
        U_dd1 = pulse.dot(U_temp.dot(U_dd1))

    return U_dd1


dim = 8

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = 1j*X.dot(Z)

w1 = 5
w2 = 1
d1 = 2
d2 = 1
d3 = 3
N = 50
X_axis = np.zeros(N)
Y0_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)
Y3_axis = np.zeros(N)
Y4_axis = np.zeros(N)

H_env = w1*np.kron(Z,I) + w2*np.kron(I,Z) + w2*np.kron(I,X) + w1*np.kron(X,I)
H_env = H_env + d1*np.kron(Z,Z) + d2*np.kron(X,X) + d3*np.kron(Y,Y)

HE = np.kron(I,H_env)

M = 4
t = 0.1
Num_of_t = 20
pulse_i = np.kron(I,np.kron(I,I))
pulse_x = np.kron(X,np.kron(I,I))
pulse_z = np.kron(Z,np.kron(I,I))
pulse_y = np.kron(Y,np.kron(I,I))


U_ideal = scipy.linalg.expm(-1j*HE*t*Num_of_t)
v1 = rand_vec(2)
v2 = rand_vec(4)
init_ket = np.array([v1[0]*v2[0],v1[0]*v2[1],v1[0]*v2[2],v1[0]*v2[3],v1[1]*v2[0],v1[1]*v2[1],v1[1]*v2[2],v1[1]*v2[3]])
vec_ideal = U_ideal.dot(init_ket)
proj_ideal = projector(v1,2)

for i in range(N):

    b = 5/(1.1**(i+1))

    # HSE = 0
    HSE = 2*b*np.kron(Z,np.kron(Z,I)) + 2*b*np.kron(Z,np.kron(I,Z)) + 2*b*np.kron(Z,np.kron(Z,Z))
    HSE = HSE + b*np.kron(X,np.kron(X,I)) + b*np.kron(X,np.kron(I,X)) + b*np.kron(X,np.kron(X,X))

    H = HSE + HE

    Num_of_U = M + 1

    QDD = np.identity(dim,dtype = 'c16')
    for j in range((Num_of_U+1)):
        dt = (filter(j+1,Num_of_U) - filter(j,Num_of_U))*t
        U_temp = Uhrig(pulse_x,Num_of_U,dt,H,dim)
        QDD = pulse_y.dot(U_temp.dot(QDD))

    QDD_total = np.identity(dim,dtype = 'c16')
    for k in range(Num_of_t):
        QDD_total = QDD_total.dot(QDD)

    BI = partial_trace(QDD,2,4,1)
    BX = partial_trace(pulse_x.dot(QDD_total),2,4,1)
    BY = partial_trace(pulse_y.dot(QDD_total),2,4,1)
    BZ = partial_trace(pulse_z.dot(QDD_total),2,4,1)

    vec_qdd = QDD_total.dot(init_ket)
    proj_qdd = projector(vec_qdd,dim)

    X_axis[i] = 1.1**(i+1)
    Y0_axis[i] = math.sqrt(1 - abs(np.vdot(vec_qdd,vec_ideal))**2)
    Y1_axis[i] = trace_distance(partial_trace(proj_qdd,2,4,0) - proj_ideal,2)
    # Y2_axis[i] = srs.matrix_norm(BX,4)
    # Y3_axis[i] = srs.matrix_norm(BY,4)
    # Y4_axis[i] = srs.matrix_norm(BZ,4)

    bx_v = BX.dot(v2)
    by_v = BY.dot(v2)
    bz_v = BZ.dot(v2)
    Y2_axis[i] = math.sqrt(abs(np.vdot(bx_v,bx_v)))
    Y3_axis[i] = math.sqrt(abs(np.vdot(by_v,by_v)))
    Y4_axis[i] = math.sqrt(abs(np.vdot(bz_v,bz_v)))


# plt.title('CDD vs PDD, HS not 0')
plt.xlabel('1/decoherence')
plt.ylabel('suppressed error')
plt.plot(X_axis,Y0_axis,label = 'fullspace',marker = '.')
plt.plot(X_axis,Y1_axis,label = 'subspace',marker = '.')
plt.plot(X_axis,Y2_axis,label = 'BX',marker = '.')
plt.plot(X_axis,Y3_axis,label = 'BY',marker = '.')
plt.plot(X_axis,Y4_axis,label = 'BZ',marker = '.')
plt.loglog()
plt.legend()
plt.show()