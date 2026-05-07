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

dictus = np.load("NonIdeal_3runs.npy", allow_pickle=True)

"""
str1 = "Hello "
str2 = "World!"
num1 = 2001
num2 = 5

print(str1+str2+" "+str(num1)+f" is {num2} years before {num1+num2}")
"""
"""
def run_experiments(num_runs=5):
    all_NonIdealruns = []
    all_Idealruns = []
    for i in range(num_runs):
        print(f"=== Running simulation {i+1}/{num_runs} ===")
        all_NonIdealruns.append(i)
        all_Idealruns.append(i+1)
    return all_NonIdealruns, all_Idealruns

nIdl, Idl = run_experiments(5)
"""

lolvar = 6