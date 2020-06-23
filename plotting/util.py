from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tools.geometry import get_orthogonal_vectors, get_bounds
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as la
import matplotlib as mpl
import numpy as np


plt.ion()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def get_cmap(f, cmap=cm.plasma):
    if isinstance(f, dict):
        kmap = {k : i for i, k in enumerate(f.keys())}
        c = cm.ScalarMappable(Normalize(0, len(f)-1), cmap)
        return {k : c.to_rgba(i)[:3] for k, i in kmap.items()}
    f = range(f) if isinstance(f, int) else f
    c = cm.ScalarMappable(Normalize(0, len(f)-1), cmap)
    return [c.to_rgba(v)[:3] for v in f]


def plot_points(axis, P, **kw):
    return axis.scatter(P[:,0], P[:,1], P[:,2], **kw)


def plot_edge(axis, P, e, **kw):
    return axis.plot(P[e,0], P[e,1], P[e,2], **kw)


def plot_edges(axis, P, E, **kw):
    return [o for e in E for o in plot_edge(axis, P, list(e), **kw)]


def join_dict(a, b):
    for k, v in a.items():
        b[k] = v
    return b


def plot_segments(axis, P, S, cmap=cm.plasma, **kw):
    c = get_cmap(S, cmap)
    colors = {k : np.tile(c[k], (len(v),1)) for k,v in S.items()}
    fkw = lambda k: {'s' : 5, 'c' : colors[k], 'label' : str(k), **kw}
    p = [plot_points(axis, P[list(v)], **fkw(k)) for k, v in S.items()]
    axis.legend()
    return p


def plot_tree(axis, self, **kw):
    for id, p in self.group_centroids.items():
        axis.scatter(p[0], p[1], p[2], **{'c' : 'black', **kw})
    for ld, q in self.link_centroids.items():
        axis.scatter(q[0], q[1], q[2], **{'c' : 'red', **kw})
        for jd in ld:
            p = self.group_centroids[jd]
            axis.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], **{'c' : 'black', **kw})


def plot_skeleton(axis, self, **kw):
    for ld, q in self.link_centroids.items():
        for jd in ld:
            p = self.group_centroids[jd]
            e = np.vstack((self.group_centroids[jd], q))
            axis.plot(e[:,0], e[:,1], e[:,2], **{'c' : 'black', **kw})


def get_circle(p, r, u, m=100):
    n = u / la.norm(u)
    v, w = get_orthogonal_vectors(n)
    t = 2 * np.pi * np.linspace(0, 1, m)
    return np.array([p + r * (v * np.sin(x) + w * np.cos(x)) for x in t])


def plot_radii(axis, curves, radii):
    for c, r in zip(curves, radii):
        avg_r = sum(r) / len(r)
        for i, (u, v) in enumerate(zip(c[:-1], c[1:])):
            x = get_circle((u + v) / 2, avg_r, v - u)
            y = get_circle((u + v) / 2, r[i], v - u)
            axis.plot(x[:,0], x[:,1], x[:,2], c='blue', alpha=0.25)
            axis.plot(y[:,0], y[:,1], y[:,2], c='red', alpha=0.25)


def plot_poly(axis, p, idx, **kw):
    try:
        verts = [p[i] for i in idx]
        poly = Poly3DCollection(verts)
    except:
        return plot_tri(axis, p, idx, **kw)
    if 'alpha' in kw: poly.set_alpha(kw['alpha'])
    if 'c' in kw: poly.set_facecolor(kw['c'])
    if 'edge_color' in kw: poly.set_edgecolor(kw['edge_color'])
    axis.add_collection3d(poly)
    return poly


def equal_aspect(axis, bounds, mult=1.1):
    l = mult * max(abs(b - a) for a, b in bounds)
    flim = [axis.set_xlim, axis.set_ylim, axis.set_zlim]
    vals = np.array([[(sum(x) - l) / 2, (sum(x) + l) / 2] for x in bounds])
    list(map(lambda x: x[0](x[1]), zip(flim, vals)))


def get_axis(figsize=(6, 5), *args, **kw):
    fig = plt.figure(1, figsize=(6,5))
    ax = plt.subplot(111, projection='3d', *args, **kw)
    plt.tight_layout()
    return fig, ax


def plot_radii(fig, ax, obj, clear=False):
    r = obj.radius
    for e in obj.tree_edges:
        z = get_circle((e[0] + e[1]) / 2, r, e[1] - e[0])
        ax.plot(z[:,0], z[:,1], z[:,2], c='red', alpha=0.5)


def plot_skeleton(fig, ax, obj, clear=True):
    if clear: ax.cla()
    bounds = get_bounds(obj.P)
    equal_aspect(ax, bounds)
    ax.scatter(obj.P[:,0], obj.P[:,1], obj.P[:,2], s=0.5)
    for e in obj.tree_edges:
        ax.plot(e[:,0], e[:,1], e[:,2], c='black')
    plot_radii(fig, ax, obj)
    fig.suptitle(obj.name)


def plot_all(fig, ax, objs):
    ax.cla()
    P = np.vstack([obj.P for obj in objs])
    bounds = get_bounds(P)
    equal_aspect(ax, bounds)
    ax.scatter(P[:,0], P[:,1], P[:,2], s=0.1)
    for obj in objs:
        for e in obj.tree_edges:
            ax.plot(e[:,0], e[:,1], e[:,2], c='black')
        plot_radii(fig, ax, obj)


def zalpha(colors, zs):
    if len(zs) == 0: return np.zeros((0, 4))
    norm = Normalize(min(zs), max(zs))
    sats = 1 - norm(zs) * 0.7
    rgba = np.broadcast_to(mpl.colors.to_rgba_array(colors), (len(zs), 4))
    return np.column_stack([rgba[:, :3], rgba[:, 3] * sats])


def animation_axis(cmap):
    plt.ioff()
    fig, ax = get_axis(figsize=(6, 6))
    cbar = fig.colorbar(cmap, ax=ax, shrink=0.7, pad=0)
    cbar.set_label("Length", labelpad=10)
    ax.set_frame_on(False)
    for w in ('w_xaxis', 'w_yaxis', 'w_zaxis'):
        x = getattr(ax, w)
        x.set_ticklabels([''])
        x.set_ticks([])
    return fig, ax


def depth_fade(ax, points, colors):
    c = np.vstack([p.sum(axis=0) / len(p) for p in points])
    a, b = ax.azim * np.pi / 180., ax.elev * np.pi / 180.
    ca, cb, sa, sb = np.cos(a), np.cos(b), np.sin(a), np.sin(b)
    n = np.array([ca * sb, sa * cb, sb])
    return zalpha(colors, -np.dot(n, [c[:,0], c[:,1], c[:,2]]))


def zoom_box(P, bounds):
    changed = False
    for i, p in enumerate(P.T):
        extrema = (p.min(), p.max())
        for j, x in enumerate(extrema):
            if x < bounds[i, 0] - 1:
                bounds[i, 0] -= 1
                changed = True
            elif x > bounds[i, 0] + 1:
                bounds[i, 0] += 1
                changed = True
    return changed
