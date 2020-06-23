import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d # required for 3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 3d polygons

plt.ion()

'''
let write some functions to help us plot '''

def plot_edges(axis, P, E, *args, **kwargs):
    for e in E:
        axis.plot(P[e,0], P[e,1], P[e,2], *args, **kwargs)

def plot_poly(axis, P, F, *args, **kwargs):
    poly = Poly3DCollection([P[f] for f in F], *args, **kwargs)
    axis.add_collection3d(poly)
    '''
    small workaround:
        the alpha property does not work so we have to provide the alpha as
         the 4th entry of an rgba color tuple. '''
    plt.pause(0.1) # pause to let object plot
    if 'alpha' in kwargs:
        colors = poly.get_facecolor()
        for c in colors:
            c[3] = kwargs['alpha']
        poly.set_color(colors)
    '''
    same deal with edgecolor.
        honestly, 3d polygons kinda suck.'''
    if 'edgecolor' in kwargs:
        poly.set_edgecolor(kwargs['edgecolor'])
    plt.pause(0.1) # update plot after property change
    return poly


'''
can't use plt.subplots with 3d.
have to make figure and axis objects manually. '''
fig = plt.figure(1, figsize=(9,7))
ax = np.array([[plt.subplot(2,2,2*i+j, projection='3d') for j in range(1,3)] for i in range(2)])

'''
we have four axes in one figure, accessed as an array:
    ax[0,0] - top left
    ax[0,1] - top right
    ax[1,0] - bottom left
    ax[1,1] - bottom right
 just like a numpy array. '''

plt.tight_layout()

if __name__ == '__main__':
    P = np.array([[0, 1, 0],   # an array of points
                [-1, 0.5, 1],
                [0, 0, -1],
                [1, 0, 0.5]])

    E = np.array([[0, 1],   # an array of edges as indices of points in P
                [1, 2],
                [0, 2],
                [1, 3],
                [2, 3],
                [3, 0]])

    F = np.array([[0, 1, 2], # an array of faces as indices of points in P
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]])

    '''
    the scatter function adds a nice depth blur '''
    ax[0,0].scatter(P[:,0], P[:,1], P[:,2])

    '''
    lets plot the edges using our plot_edges function.
    *args and **kwargs are any additional arguments that have not been specified.
    *args collects all extra arguments not specified that are not keyword arguments as a list.
     (after the specified arguments and before the keyword arguments).
    **kwargs collects all extra keyword arguments as a dictionary.
    here we pass the argument 'black' with keyword c for color.
    Because our function passes all *args and **kwargs to plt.plot this allows us to specify
     properties of our edges such as color, linewidth, linestyle etc. '''
    plot_edges(ax[0,1], P, E, c='black')

    '''
    lets add the faces of our polyhedron '''
    ax[1,0].scatter(P[:,0], P[:,1], P[:,2], c='black')
    poly = plot_poly(ax[1,0], P, F, alpha=0.5, edgecolor='black')
