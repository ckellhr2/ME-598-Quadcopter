import numpy as np
"""
testola = np.array([1,5,90])
difs = np.diff(testola)
print(difs)
"""

"""
import scipy.stats as stat
stat.wasserstein_distance_nd
"""

"""
#print(np.zeros((2,2)))

#for lol in range(10):
#    print(lol)
"""

"""


packitup = []

a = np.array([[2, 3], [4, 5]])
b = np.array([[2, 3, 4], [4, 5, 6]])

packitup.append({
    'a': a,
    'b': b
})

c = np.array([[1, 1], [1, 1]])
d = np.array([[2, 2, 2], [2, 2, 2]])

packitup.append({
    'c': c,
    'd': d
})

np.save("Test.npy", {
    'a': a,
    'b': b
}, allow_pickle=True)

dictus = np.load("Test.npy", allow_pickle=True)
#a_new = dictus[0]['a']
#b_new = dictus[0]['b']
"""

dictus = np.load("MidtermPresentationData_Ideal.npy", allow_pickle=True)

lolvar = 6