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

def partial_trace(M,dim1,dim2):

    # M_temp = np.zeros([dim1,dim1],dtype = complex)
    # for i1 in range(dim1):
    #     digit1 = i1*(dim2)
    #     for i2 in range(dim1):
    #         digit2 = i2*(dim2)
    #         entry = 0
    #         for index in range(dim2):
    #             real_index1 = digit1 + index 
    #             real_index2 = digit2 + index 
    #             entry = entry + M[real_index1][real_index2]
    #         M_temp[i1][i2] = entry

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

b = 0.01
M1 = np.array([[1,0],[0,0]])
M2 = np.array([[1-b,0],[0,b]])
d = trace_distance(M1-M2,2)