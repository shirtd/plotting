import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def stuple(x, *args, **kwargs):
    return tuple(sorted(x, *args, **kwargs))

def stzip(x, y, *args, **kwargs):
    return stuple(zip(x,y), *args, **kwargs)

def seqzip(l):
    return zip(l,l[1:]+[l[0]])

plt.ion()

fig, ax = plt.subplots(1,1)

ax.set_xlim(0,1)
ax.set_ylim(0,1)

X = np.random.rand(100, 2)

ax.scatter(X[:,0], X[:,1], s=1)

V = Voronoi(X)

R = [r for r in V.regions if len(r) and not -1 in r]
F = [V.vertices[r] for r in R]

# for f in F:
#     f = np.vstack([f, f[0]])
#     ax.plot(f[:,0], f[:,1], c='black')

E = {stuple(e) for r in R for e in seqzip(r)}

for e in V.vertices[list(E)]:
    ax.plot(e[:,0], e[:,1], c='black')
