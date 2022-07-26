import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs

# def concatenated(U,P):

#     return P.dot(U).dot(P.dot(U))

def concatenated(U,P1,P2):

    U_temp = P1.dot(U).dot(P2.dot(U))

    return (-1)*U_temp.dot(U_temp)

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


dim = 8

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = 1j*X.dot(Z)

w0 = 1
w1 = 5
w2 = 5
d1 = 2
d2 = 2
d3 = 2

N = 25
X_axis = np.zeros(N)
Y0_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)
Y3_axis = np.zeros(N)
Y4_axis = np.zeros(N)

HE = w1*np.kron(I,np.kron(Z,I)) + w2*np.kron(I,np.kron(I,Z)) + w1*np.kron(I,np.kron(I,X)) + w2*np.kron(I,np.kron(X,I))
HE = HE + d1*np.kron(I,np.kron(Z,Z)) + d2*np.kron(I,np.kron(X,X)) + d3*np.kron(I,np.kron(Y,Y))

M = 4
T = 0.1
pulse_i = np.kron(I,np.kron(I,I))
pulse_x = np.kron(X,np.kron(I,I))
pulse_z = np.kron(Z,np.kron(I,I))
pulse_y = np.kron(Y,np.kron(I,I))

U_ideal = scipy.linalg.expm(-1j*HE*T)
v1 = rand_vec(2)
v2 = rand_vec(4)
init_ket = np.array([v1[0]*v2[0],v1[0]*v2[1],v1[0]*v2[2],v1[0]*v2[3],v1[1]*v2[0],v1[1]*v2[1],v1[1]*v2[2],v1[1]*v2[3]])
vec_ideal = U_ideal.dot(init_ket)
proj_ideal = projector(vec_ideal,dim) 

for i in range(N):

    b = 10/(1.1**(i+1))
    HSE = b*np.kron(Z,np.kron(Z,I)) + b*np.kron(X,np.kron(I,X)) + b*np.kron(Y,np.kron(I,Y))
    HSE = HSE + b*np.kron(Z,np.kron(I,Z)) + b*np.kron(X,np.kron(I,X)) + b*np.kron(Y,np.kron(I,Y))

    H = HSE + HE

    Num_of_U = 4**M
    dt = T/Num_of_U

    U_ideal = scipy.linalg.expm(-1j*HE*dt)
    CDD = scipy.linalg.expm(-1j*H*dt)
    for l in range(M):
        CDD = concatenated(CDD,pulse_x,pulse_z)
        # U_ideal = concatenated(U_ideal,pulse_i,pulse_i)

    vec_cdd = CDD.dot(init_ket)
    proj_cdd = projector(vec_cdd,dim)

    # X_axis[i] = 1.1**(i+1)
    # # Y1_axis[i] = srs.matrix_norm((PDD - U_ideal),dim)
    # Y1_axis[i] = srs.matrix_norm(((-1)*CDD - U_ideal),dim)
    # # Y1_axis[i] = trace_distance(partial_trace(proj_pdd - proj_ideal,2,4),2)


    BI = partial_trace(CDD,2,4,1)/2
    BX = partial_trace(pulse_x.dot(CDD),2,4,1)/2
    BY = partial_trace(pulse_y.dot(CDD),2,4,1)/2
    BZ = partial_trace(pulse_z.dot(CDD),2,4,1)/2

    X_axis[i] = 1/b
    Y0_axis[i] = math.sqrt(abs(1 - abs(np.vdot(vec_cdd,vec_ideal))**2))
    Y1_axis[i] = trace_distance(partial_trace(proj_cdd - proj_ideal,2,4,0),2)
    bx_v = BX.dot(v2)
    by_v = BY.dot(v2)
    bz_v = BZ.dot(v2)
    Y2_axis[i] = math.sqrt(abs(np.vdot(bx_v,bx_v)))
    Y3_axis[i] = math.sqrt(abs(np.vdot(by_v,by_v)))
    Y4_axis[i] = math.sqrt(abs(np.vdot(bz_v,bz_v)))

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



# plt.plot(X_axis,Y1_axis,label = 'fullspace',marker = '.')
# plt.plot(X_axis,Y2_axis,label = 'subspace',marker = '.')
# plt.loglog()
# # plt.ylim(0,1)
# plt.legend()
# plt.show()