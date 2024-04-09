import numpy as np


A = np.arange(15).reshape(3, 5)  # matrika 3x5
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]

a = A[0]  # vektor
# [0 1 2 3 4]

B = np.arange(10).reshape(2, 5)  # matrika 2x5
# [[0 1 2 3 4]
#  [5 6 7 8 9]]

b = B[0]
# [0 1 2 3 4]


# Povprečje stolpcev matrike A

[np.mean(A[:, i]) for i in range(A.shape[1])]
np.mean(A, axis=0)  # boljše


# Centriranje stolpcev matrike A

A - np.mean(A, axis=0)  # boljše


# Dolžina vektorja a

sum(v**2 for v in a)**0.5
a.dot(a)**0.5  # boljše
np.linalg.norm(a)  # še boljše


# Skalarni produkt a in b

sum([ai*bi for ai, bi in zip(a, b)])
a.dot(b)  # boljše


# Skalarni produkt za vsako vrstico matrike A z vektorjem b

[v.dot(b) for v in A]
A.dot(b)  # boljše


# Skalarni produkt za vsako vrstico matrike A z vsako vrstico matrike B

[[v.dot(w) for w in B] for v in A]
A.dot(B.T)  # boljše


# Kako pa množenje dveh vektorjev organizirati tako, da bomo dobili matriko?
# Nekaj takega bi recimo potrebovali, če bi sami računali kovariančno matriko.

# Želimo torej zmnožiti  [[0]   z   [[0 1 2 3 4]]
#                         [1]
#                         [2]
#                         [3]
#                         [4]]

# Brez numpya bi to bilo
[[v1*v2 for v2 in b] for v1 in a]  # prepočasno!

# Transponiranje ne dela, ker numpy 1D arrayi nimajo orientacije: dobimo skalarni produkt
a.T.dot(b)  # 30
a.dot(b.T)  # 30
np.dot(a.T, b)  # 30

# Prva rešitev: ročno ustvarimo 2D array s pravo orientacijo
np.dot(a.reshape(-1, 1), b.reshape(1, -1))

# Druga rešitev
np.outer(a, b)
