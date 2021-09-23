from GrassmannianFusion import GrassmannianFusion
from Initialization import *
from helper_functions import evaluate
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.cluster import DBSCAN
import time
import seaborn as sns

def main():
    m = 100 #100-300
    n = 100 #100-300
    r = 3 #3-5
    K = 3
    missing_rate = 0.6

    #init low rank subspace based on orthonormal basis
    shape = (m,n,r)
    X, masks, X_lowRank_array,labels = create_n_subspace_clusters(n_clusters=K, shape = shape)

    #observed index
    Omega = np.random.choice(m*n, size = int(m*n * (1-missing_rate) ), replace= False )

    #create observed matrix
    X_omega = np.zeros((m,n))
    for p in Omega:
        X_omega[p // n, p % n] = X[p // n, p % n]


    lambda_in = 1 #usually e-5
    weight_f_in = 1
    print('Paramter: lambda = ',lambda_in,', K = ',K,', m = ', m, ', n = ',n,', r = ',r,', missing_rate =', missing_rate)

    GF = GrassmannianFusion(X = X_omega,
                            Omega = Omega,
                            r = r,
                            lamb = lambda_in,
                            weight_factor = weight_f_in,
                            g_threshold= 1e-6,
                            bound_zero = 1e-10,
                            singular_value_bound = 1e-5,
                            g_column_norm_bound = 1e-5,
                            U_manifold_bound = 1e-5)

    GF.train(max_iter = 50, step_size = 1)
    U_array = GF.get_U_array()
    d_matrix = GF.distance_matrix()


    #ax = sns.heatmap(d_matrix)
    sc = SpectralClustering(n_clusters=K,affinity = 'nearest_neighbors',random_state=0).fit(d_matrix)
    print('SC Accuracy:' , 1 - evaluate(sc.labels_, truth , K))



    S = np.array(S)
    S_mat = np.array(S[0]).reshape((1,100,100,3))
    for si in S[1:]:
        si_reshape = np.array(si).reshape((1,100,100,3))
        S_mat = np.concatenate((S_mat, si_reshape))

    optional_params = {'GoogleColab': True, 'objective_plot': True, 'max_epoch': 200, 'final_picture': True}
    optional_params['folder_name'] = 'saved_result'
    embedding = grasscare_plot(S = S_mat, labels = labels, video = True, optional_params = optional_params)


if __name__ == '__main__':
    main()
