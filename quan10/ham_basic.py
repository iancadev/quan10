import numpy as np
import copy
from qiskit.quantum_info import Pauli, SparsePauliOp

class Object(object):
    pass
G = Object()


def norm(b):
    v = copy.copy(b)
    for i in range(len(v)):
        v[i] /= sum(v)
    return v, sum(v)

def avg(x):
    avg = sum(x)/len(x)
    return avg

def reven(v):
    b = []
    for i in range(1,len(v)):
        b.append(v[i]-v[i-1])
    return b

def add(b, r):
    v = copy.copy(b)
    for i in range(len(v)):
        v[i] += r
    return v

def multip(b, r):
    v = copy.copy(b)
    for i in range(len(v)):
        v[i] *= r
    return v

def inner(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def stitch(v):
    s = ""
    for i in v:
        s += i
    return s

def dict_to_vec(dict):
    keys_list = list(dict.keys())

    vector = []
    for i in range(2**sum(d)):
        vector.append(dict.get(i, 0))
        
    return vector

def read_data(df):
    # don't forget to set program.size
    
    B, R, delta = G.B, G.R, G.delta
    
    holder = []
    for i in range(1,5):
        holder.append(df.iloc[:, i].tolist())
    
    holder2 = []
    for i in holder:
        holder2.append(i[0:int(len(holder[0])/24)])
    holder = holder2
    
    # PRICE
    p = []
    for i in holder:
        p.append(i)
        
    # CURRENT PRICE
    P = []
    for i in holder:
        P.append(i[-1])
    
    # REVENUE
    r = []
    for i in holder:
        r.append(reven(i))

    # AVERAGE REVENUE
    u = []
    for i in r:
        u.append(avg(i))

    # CONVOLUTION MATRIX
    convolu = np.zeros((len(holder), len(holder)))

    for i in range(len(holder)):
        for j in range(len(holder)):
            convolu[i,j] = inner(add(r[i],-u[i]), add(r[j],-u[j]))

    # CONVERT TO BITS
    d = []
    for i in range(len(holder)):
        d.append(round(np.log2(int(B/P[i])+1)))

    # REVERSION MATRIX
    conver = np.zeros((len(d), sum(d)))
    
    k = 0
    for i in range(len(d)):
        for j in range(d[i]):
            conver[i,k] = 2**j
            k += 1

    global h, h_, hi, J, J_, J__, ji, pi, pi_, bi
    I = []
    for i in range(sum(d)):
        I.append("I")
    
    ############### hi Zi ################
    h = (np.array([u])*np.array([P])/B)
    h_ = []
    for i in range(len(d)):
        for j in range(d[i]):
            h_.append(h[0][i]*(2**j))
        
    hi = []
    for i in range(sum(d)):
        I[i] = "Z"
        hi.append(stitch(I))
        I[i] = "I"
    ######################################
    
    ############# ji Zi Zj ###############
    J = ((convolu*(np.array([P])*(1/B)).reshape(-1, 1)).T)*(np.array([P])/B).reshape(-1, 1)
    J_ = np.dot(conver.T,(np.dot(J, conver)))
    
    J__ = []
    ji = []
    for i in range(sum(d)):
        for j in range(sum(d)):
            I[i] = "Z"
            I[j] = "Z"
            if j != i:
                ji.append(stitch(I))
                J__.append(J_[i,j])
            I[i] = "I"
            I[j] = "I"
    ######################################
    
    ########### Budget Penalty ###########
    pi = (np.array([P])*1/B)
    pi_ = []
    for i in range(len(d)):
        for j in range(d[i]):
            pi_.append(pi[0][i]*(2**j))

    bi = copy.copy(hi)
    ######################################

    G.h_, G.hi, G.J__, G.ji, G.pi_, G.bi, G.size, G.conver, G.P, G.d = (
        h_, hi, J__, ji, pi_, bi, sum(d), conver, P, d
    )

    program.size = sum(d)


def initialize_program(B, R, delta):
    G.B = 200 # Budget
    G.R = 0.5 # Risk Ratio
    G.delta = 1 # Penalty Term

def ham_sparseop():
    return [
        [SparsePauliOp(G.hi, G.h_)],
        [SparsePauliOp(G.ji, G.J__)],
        [SparsePauliOp(G.bi, G.pi_)]
    ]

def cost(result):
    return result[0].data.evs[0] + G.R * result[0].data.evs[1] + G.delta * (result[0].data.evs[2] - (2-sum(G.pi_)*(1/G.B)))**2

def interpret(ans_r):
    result = ans_r
    
    dict = result.data.meas.get_int_counts()
    max_key = max(dict, key=dict.get)
    
    binary = bin(max_key)[2:]
    binary_ = "0"*(G.size - len(binary))+binary

    tempvec = np.zeros((1,G.size))
    for i in range(G.size):
        tempvec[0,i] += int(binary_[i])
    
    xvec = np.dot(G.conver, tempvec[0])
    print("optimal vector:", xvec)
    print("money invested:", np.dot(G.P, xvec))

program = Object()
program.initialize = initialize_program
program.size = None
program.read_data = read_data
program.ham_sparseop = ham_sparseop
program.cost_qiskit_result = cost
program.interpret_qiskit_result = interpret


