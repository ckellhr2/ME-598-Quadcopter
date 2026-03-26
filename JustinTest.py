import numpy as np
#import scipy.stats as stat

#stat.wasserstein_distance_nd

#print(np.zeros((2,2)))

#for lol in range(10):
#    print(lol)
    

a = np.array([[2, 3], [4, 5]])
b = np.array([[2, 3, 4], [4, 5, 6]])

np.save("Test.npy", np.array([{'a': a, 'b': b}]), allow_pickle=True)

dictus = np.load("Test.npy", allow_pickle=True)
a_new = dictus[0]['a']
b_new = dictus[0]['b']

lolvar = 6