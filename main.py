from GrassmannianFusion import GrassmannianFusion
from Initialization import *
from helper_functions import evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import seaborn as sns

def main():
    m = 100 #100-300
    n = 100 #100-300
    r = 3 #3-5
    K = 3
    missing_rate = 0.6
    init_params = (m,n,r,K,missing_rate)
    
    #all-in-one init function
    X_omega, labels, Omega, info = initialize_X_with_missing(init_params)
    
    #parameter for training
    lambda_in = 1 #usually e-5
    weight_f_in = 1
    print('Paramter: lambda = ',lambda_in,', K = ',K,', m = ', m, ', n = ',n,', r = ',r,', missing_rate =', missing_rate)
    
    #object init
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
    
    #reusable train function
    GF.train(max_iter = 10, step_size = 1)
    
    #U_array getter function
    S = GF.get_U_array()
    
    #d_matrix getter function
    d_matrix = GF.distance_matrix()


    #ax = sns.heatmap(d_matrix)
    sc = SpectralClustering(n_clusters=K,affinity = 'nearest_neighbors',random_state=0).fit(d_matrix)
    print('SC Accuracy:' , 1 - evaluate(sc.labels_, labels , K))

    
    GF.train(max_iter = 50, step_size = 1)
    d_matrix = GF.distance_matrix()
    sc = SpectralClustering(n_clusters=K,affinity = 'nearest_neighbors',random_state=0).fit(d_matrix)
    print('SC Accuracy:' , 1 - evaluate(sc.labels_, labels , K))


    '''
    optional_params = {'GoogleColab': True, 'objective_plot': True, 'max_epoch': 200, 'final_picture': True}
    optional_params['folder_name'] = 'saved_result'
    embedding = grasscare_plot(S = S_mat, labels = labels, video = True, optional_params = optional_params)
    '''

if __name__ == '__main__':
    main()
