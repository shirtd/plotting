from sklearn.decomposition import PCA
import scipy.spatial as spatial
import numpy.linalg as la
import numpy as np

def get_bounds(P):
    return [tuple(map(lambda f: f(v[a] for v in P), (min, max))) for a in range(3)]

def max_bounds(frames):
    if isinstance(frames, dict):
        frames = list(frames.values())
    P = np.vstack([o.P for frame in frames for o in frame.skeletons.values()])
    return get_bounds(P)

def line_project(x, v):
    if x.ndim == 1: return x.dot(v) / v.dot(v) * v
    return np.array([line_project(y, v) for y in x])

def to_line(Q, a):
    b = np.array([1., 0.])
    n = a / la.norm(a)
    x = b - np.dot(b, n) * n
    if la.norm(x) == 0:
        print(a, x)
        raise Exception
    x /= la.norm(x)
    return np.array([[q.dot(x)] for q in Q])

def plane_project(x, v):
    if x.ndim == 1: return x - x.dot(v) / la.norm(v) * v / la.norm(v)
    return np.array([plane_project(y, v) for y in x])

def to_plane(Q, a, axis=0):
    b = np.array([0., 0., 0.])
    b[axis] = 1
    n = a / la.norm(a)
    x = b - np.dot(b, n) * n
    x /= la.norm(x)
    y = np.cross(n, x)
    return np.array([[q.dot(x), q.dot(y)] for q in Q])

def bounding_box(P):
    if len(P) == 1:
        A = np.eye(3, dtype=float)
    elif len(P) == 2:
        x = P[1] - P[0] / la.norm(P[1] - P[0])
        y = np.array([1., 0., -x[0] / x[2]])
        z = np.cross(x, y)
        A = np.vstack((x, y / la.norm(y), z / la.norm(z)))
    else:
        pca = PCA(3)
        pca.fit_transform(P)
        A = pca.components_
    W = np.array([np.matmul(A, p) for p in P])
    H = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    B = np.array([[f(x.dot(p) for p in P) for f in (min, max)] for x in A])
    C = np.array([[B[b, g] for b, g in zip(range(3), h)] for h in H])
    A_inv = la.inv(A)
    idxs = ((0,3,1), (4,1,1), (6,5,1), (2,7,1), (0,6,2), (5,3,2))
    face_indices = [[i, i+k, j, j-k] for i, j, k in idxs]
    return np.array([np.matmul(A_inv, c) for c in C]), face_indices, A

def cylinder_volume(x, y):
    h, r = abs(x[0] - x[1]), abs(y[0] - y[1]) / 2
    return h * np.pi * r ** 2

def get_length(X, v):
    v /= la.norm(v)
    x = X.dot(v) * v[:, None]
    L = x.T.dot(v)
    return la.norm(L.max() - L.min())

def principal_length(X):
    if len(X) > 2:
        pca = PCA(1)
        pca.fit_transform(X)
        v = pca.components_[0]
        return get_length(X, v)
    return la.norm(X[0] - X[1]) if len(X) == 2 else 0

def line_plane_intersect(n, p, l, q):
    n, l = n / la.norm(n), l / la.norm(l)
    d = (p - q).dot(n) / l.dot(n)
    return d * l + q

def edge_intersects_face(n, p, u, v):
    x = line_plane_intersect(n, p, u - v, u)
    z = (v - u).dot(x - u)
    return -0.1 <= z and z <= la.norm(u - v) ** 2

def get_orthogonal_vectors(n):
    n = n / la.norm(n)
    u = np.array([n[1] - n[2], n[2] - n[0], n[0] - n[1]])
    v = np.cross(n, u)
    return u / la.norm(u), v / la.norm(v)

def homogenize(U, x):
    return np.vstack((np.hstack((U, x[:, None])), np.ones(4)))

def in_segment_test(U, V, x):
    return la.det(homogenize(U, x)) * la.det(homogenize(V, x)) < 0

def signed_volume(a, b, c, d):
    return np.dot(np.cross(b - a, c - a), d - a) / 6

def collinear(x, y, z):
    yz, zx, xy = y[1] - z[1], z[1] - x[1], x[1] - y[1]
    return abs((x[0] * yz + y[0] * zx + z[0] * xy) / 2) < 1e-1

def lineside(x, y, z):
    return la.det(np.vstack((np.vstack((x, y, z)).T, np.ones(3)))) > 0

def insphere(U, p):
    V = np.vstack((p, U))
    N = np.array([la.norm(v) ** 2 for v in V])
    X = np.vstack((np.hstack((V, N[:,None])).T, np.ones(5)))
    Y = np.vstack((U.T, np.ones(4)))
    return la.det(X) / la.det(Y) > 0

def incircle(U, p):
    V = np.vstack((p, U))
    n = np.cross(U[0] - U[1], U[1] - U[2])
    X = to_plane(V, n)
    N = np.array([la.norm(x) ** 2 for x in X])
    Y = np.vstack((np.hstack((X, N[:, None])).T, np.ones(4)))
    Z = np.vstack((X[1:].T, np.ones(3)))
    return la.det(Y) / la.det(Z) <= 0

def triangle_edge_intersect(P, t, e):
    t, e = P[list(t)], P[list(e)]
    u, v = t[1] - t[0], t[2] - t[0]
    n = np.cross(u, v)

    dir, w0 = e[1] - e[0], e[0] - t[0]
    r = -1 * n.dot(w0) / n.dot(dir)
    if -1e-10 > r or r > la.norm(dir) ** 2:
        return False

    w = e[0] + r * dir - t[0]
    uu, uv, vv = u.dot(u), u.dot(v), v.dot(v)
    wu, wv = w.dot(u), w.dot(v)
    D = uv * uv - uu * vv
    s, t = (uv * wv - vv * wu) / D, (uv * wu - uu * wv) / D
    if (s < 0 or s > 1) or (t < 0 or (s + t) > 1.0):
        return False

    return True

def point_line_distance(a, b, v):
    ab, av, bv = b - a, v - a, v - b
    if av.dot(ab) <= 0.0: return la.norm(av)
    if bv.dot(ab) >= 0.0: return la.norm(bv)
    return la.norm(np.cross(ab, av))/ la.norm(ab)

def point_curve_distance(c, p):
    return min(point_line_distance(u, v, p) for u, v in zip(c[:-1], c[1:]))

def closest_edge(E, p):
    f = lambda i: point_line_distance(E[i][0], E[i][1], p)
    return min(range(len(E)), key=f)

def get_angle(a, b, c):
    u, v = b - a, b - c
    y = (u / la.norm(u)).dot(v / la.norm(v))
    if abs(y) > 1:
        y = -1 if y < -1 else 1
    return np.arccos(y)
