## Testing Synthetic Dataset on OMTL_LogDet

d = 10
K = 3
n = 100
EPOCH = 0.5*300
eta = 1e-7

import numpy as np
import copy

np.random.seed(10)

def sym(A):
    return (A + np.transpose(A)) / 2

def LogDet(A, W, eta):
    return np.linalg.inv(np.linalg.inv(A) + eta*sym(np.dot(np.transpose(W), W)))

w1 = np.array([np.random.uniform(-1,1) for x in range(10)])
w2 = -w1
w3 = np.array([np.random.uniform(-1,1) for x in range(8)])

dot1 = np.dot(w1[0:8],w3) + w1[8]

w3 = np.append(w3, [1, -dot1/w1[9]])

x_1 = np.zeros((d,n))
x_2 = np.zeros((d,n))
x_3 = np.zeros((d,n))
y_1 = np.zeros(n)
y_2 = np.zeros(n)
y_3 = np.zeros(n)

for i in range(n):
    current_Rand_for_x_1 = np.array([np.random.randint(0,100) for x in range(10)])
    x_1[:,i] = current_Rand_for_x_1
    y_1[i] = np.sign(np.dot(w1,x_1[:,i]))
    
    current_Rand_for_x_2 = np.array([np.random.randint(0,100) for x in range(10)])
    x_2[:,i] = current_Rand_for_x_2
    y_2[i] = np.sign(np.dot(w2,x_2[:,i]))
    
    current_Rand_for_x_3 = [np.random.randint(0,100) for x in range(10)]
    x_3[:,i] = current_Rand_for_x_3
    y_3[i] = np.sign(np.dot(w3,x_3[:,i]))
    
x_1_with_Task = np.zeros((d + 1,n))
x_1_with_Task = np.vstack((np.ones((1,n)) , x_1))

x_2_with_Task = np.zeros((d + 1,n))
x_2_with_Task = np.vstack((np.ones((1,n)) , x_2))

x_3_with_Task = np.zeros((d + 1,n))
x_3_with_Task = np.vstack((np.ones((1,n)) , x_3))

x_all_tasks = np.hstack((x_1 , x_2 , x_3))
y_all_tasks = np.hstack((y_1,y_2,y_3))


indices = np.hstack((np.zeros((100)) , np.ones((100)) , 2*np.ones((100))))

""" Initialization """
A = (1.0/K) * np.eye(K)
w = np.zeros(K*d)
W = np.reshape(w, (d, K), order='F')
s = 0

rand_Indices = (np.array(np.random.permutation(300)))

## Testing our data
for t in range(300):
    x = x_all_tasks[:,rand_Indices[t]]; 
    y = y_all_tasks[rand_Indices[t]]; 
    i = int(indices[rand_Indices[t]])
    phi = np.concatenate([np.zeros(i*d), x, np.zeros((K-i-1)*d)], axis=0)
    y_pred = np.sign(np.inner(w, phi))
    if y_pred == 0: y_pred = 1
    
    if y != y_pred:
        # update w_s and A_s
        Wlast = copy.deepcopy(W)
        w = w + y * np.dot(np.linalg.inv(np.kron(A, np.eye(d))),phi)
        W = np.reshape(w, (d, K), order='F')
        if t >= EPOCH:
            A = LogDet(A, Wlast, eta)
        s += 1
        

print ("Task relatedness Matrix for OMTL LogDet:\n" , np.linalg.inv(A))
print ('\n')
print ("Learned task weights correlation for OMTL LogDet:\n", np.corrcoef([w[0:10],w[10:20],w[20:30]]))
print ("\n")
print ("True weights correlation for OMTL LogDet:\n", np.corrcoef([w1,w2,w3]))
print('\n')

nIncorrect = 0
#Testing Accuracy:
for dataSetNumber in range(100):
    ypred_1 = np.sign(np.dot(w[0:10],x_1[:,dataSetNumber]))
    if ypred_1 != y_1[dataSetNumber]:
        nIncorrect +=  1
    
    ypred_2 = np.sign(np.dot(w[10:20],x_2[:,dataSetNumber]))
    if ypred_2 != y_2[dataSetNumber]:
        nIncorrect +=  1
        
    ypred_3 = np.sign(np.dot(w[20:30],x_3[:,dataSetNumber]))
    if ypred_3 != y_3[dataSetNumber]:
        nIncorrect +=  1 
        
print("Accuracy :", (1-nIncorrect/(3*n))*100)         
        