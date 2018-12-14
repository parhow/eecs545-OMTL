## Testing real Data using novel OMTL approaches OMTL_LogDet , OMTL_Von , OMTL_Cov

import numpy as np
from scipy.linalg import expm, logm, sqrtm
import scipy
import pickle
import copy

def sym(A):
    return (A + np.transpose(A)) / 2

def LogDet(A, W, eta):
    return np.linalg.inv(np.linalg.inv(A) + eta*sym(np.transpose(W) @ W))

def vonNeumann(A, W, eta):
    return expm(logm(A) - eta*sym(np.transpose(W) @ W))

# ### Landmine Dataset
with open('./LandmineData_feature.pkl', 'rb') as f:
    X = pickle.load(f)

with open('./LandmineData_label.pkl', 'rb') as f:
    Y = pickle.load(f)

with open('./LandmineData_taskID.pkl', 'rb') as f:
    I = pickle.load(f)
n = X.shape[0]
d = X.shape[1]
K = len(np.unique(I))
""" Permutation """
np.random.seed(0)
rp = np.random.permutation(n)
X = X[rp]
Y = Y[rp]
I = I[rp]
""" Training / test split """
ntrain = int(0.7 * n) # 1482
ntest = n - ntrain # 13338
X_train = X[:ntrain]
X_test = X[ntrain:]
Y_train = Y[:ntrain]
Y_test = Y[ntrain:]
I_train = I[:ntrain]
I_test = I[ntrain:]


# ### Spam Dataset
with open('./spam_feature.pkl', 'rb') as f:
    X = pickle.load(f)

with open('./spam_label.pkl', 'rb') as f:
    Y = pickle.load(f)

with open('./spam_taskID.pkl', 'rb') as f:
    I = pickle.load(f)
n = X.shape[0]
d = X.shape[1]
K = len(np.unique(I))
""" Dimensionality reduction """
np.random.seed(0)
d_reduce = 100
X = X[:, np.random.choice(range(d), d_reduce)]
d = d_reduce
""" Permutation """
np.random.seed(0)
rp = np.random.permutation(n)
X = X[rp]
Y = Y[rp]
I = I[rp]
""" Training / test split """
ntrain = int(0.7 * n) # 1482
ntest = n - ntrain # 13338
X_train = X[:ntrain]
X_test = X[ntrain:]
Y_train = Y[:ntrain]
Y_test = Y[ntrain:]
I_train = I[:ntrain]
I_test = I[ntrain:]


# ### Parameters
EPOCH = 0.1 * ntrain
print(EPOCH)
eta = 1e-8

# ### OMTL Training common

""" Initialization """
A = (1/K) * np.identity(K)
w = np.zeros(K*d)
W = np.reshape(w, (d, K), order='F')
s = 0

for t in range(ntrain):
    x = X_train[t]; y = Y_train[t]; i = I_train[t]
    phi = np.concatenate([np.zeros(i*d), x, np.zeros((K-i-1)*d)], axis=0)
    y_pred = np.sign(np.inner(w, phi))
    if y_pred == 0: y_pred = 1
    
    if y != y_pred:
        # update w_s and A_s
        Wlast = copy.deepcopy(W)
        w = w + y * np.linalg.inv(np.kron(A, np.identity(d))) @ phi
        W = np.reshape(w, (d, K), order='F')
        if t >= EPOCH:
            """ select one of them """
            #A = LogDet(A, Wlast, eta)
            #A = vonNeumann(A, Wlast, eta)
            #A = np.cov(Wlast, rowvar=False)
            #A = BatchOpt(Wlast)
        s += 1


# ### Validation
correct_total = 0
for k in range(K):
    mask = np.squeeze(I_test == k)
    nk = np.sum(mask)
    y_pred = np.sign(X_test[mask] @ w[k*d : (k+1)*d])
    y_pred[y_pred == 0] = 1
    correct = np.sum(y_pred == np.squeeze(Y_test[mask]))
    print("Task %d:" % k)
    print('\tCorrect prediction: %d' % correct)
    print('\tOut of %d samples' % nk)
    print('\tAccuracy: %f\n' % (correct/nk))

    correct_total += correct

print('Overall accuracy: %f' % (correct_total/ntest))


# ### Parameter tuning
epoch_list = np.arange(0.1, 1.1, 0.1)
eta_list = [1e-6, 1e-7, 1e-8]
update_list = ['log', 'vonNeumann', 'cov', 'batch']
for epoch in epoch_list:
    for eta in eta_list:
        for update in update_list:
            print('##############################')
            print('# epoch = %f' % epoch)
            print('# eta = %f' % eta)
            print('# update method = %s' % update)
            print('##############################')
            try:
                EPOCH = epoch * ntrain
                """ Initialization """
                A = (1/K) * np.identity(K)
                w = np.zeros(K*d)
                W = np.reshape(w, (d, K), order='F')
                s = 0

                for t in range(ntrain):
                    x = X_train[t]; y = Y_train[t]; i = I_train[t]
                    phi = np.concatenate([np.zeros(i*d), x, np.zeros((K-i-1)*d)], axis=0)
                    y_pred = np.sign(np.inner(w, phi))
                    if y_pred == 0: y_pred = 1

                    if y != y_pred:
                        Wlast = W
                        w = w + y * np.linalg.inv(np.kron(A, np.identity(d))) @ phi
                        W = np.reshape(w, (d, K), order='F')
                        if t >= EPOCH:
                            if update == 'log': A = LogDet(A, Wlast, eta)
                            if update == 'vonNeumann': A = vonNeumann(A, Wlast, eta)
                            if update == 'cov': A = np.cov(Wlast, rowvar=False)
                            if update == 'batch': A = BatchOpt(Wlast)

                        s += 1

                correct_total = 0
                for k in range(K):
                    mask = np.squeeze(I_test == k)
                    nk = np.sum(mask)
                    correct = np.sum(np.sign(X_test[mask] @ w[k*d : (k+1)*d]) == np.squeeze(Y_test[mask]))

                    correct_total += correct

                print('\tOverall accuracy: %f\n\n' % (correct_total/ntest))
            except Exception as inst:
                print('\tERROR occurred')
                print(inst)
                print('\n')



epoch_list = np.arange(0.1, 1.1, 0.1)
eta_list = [1e-6, 1e-7, 1e-8]
for epoch in epoch_list:
    for eta in eta_list:
        print('##############################')
        print('# epoch = %f' % epoch)
        print('# eta = %f' % eta)
        print('##############################')
        try:
            A = np.linalg.inv((1/(K+1)) * (np.identity(K) + np.ones((K, K)))) # fixed
            w = np.zeros(K*d)
            W = np.reshape(w, (d, K), order='F')
            s = 0

            for t in range(ntrain):
                x = X_train[t]; y = Y_train[t]; i = I_train[t]
                phi = np.concatenate([np.zeros(i*d), x, np.zeros((K-i-1)*d)], axis=0)
                y_pred = np.sign(np.inner(w, phi))
                if y_pred == 0: y_pred = 1

                if y != y_pred:
                    Wlast = copy.deepcopy(W)
                    w = w + y * np.linalg.inv(np.kron(A, np.identity(d))) @ phi
                    W = np.reshape(w, (d, K), order='F')
                    s += 1
            correct_total = 0
            for k in range(K):
                mask = np.squeeze(I_test == k)
                nk = np.sum(mask)
                y_pred = np.sign(X_test[mask] @ w[k*d : (k+1)*d])
                y_pred[y_pred == 0] = 1
                correct = np.sum(y_pred == np.squeeze(Y_test[mask]))

                correct_total += correct

            print('\tOverall accuracy: %f\n\n' % (correct_total/ntest))
        except Exception as inst:
            print('\tERROR occurred')
            print(inst)
            print('\n')
