import matplotlib.pyplot as plt # import pyplot as plt
import numpy as np # we will use numpy arrays

'''
interactive on:
    automatically refresh drawing when a new element is added.
    without this you must use plt.show() after each update.
    (note: plt.show() defaults to blocking the command prompt
     plt.show(block=False) with plot without blocking) '''
plt.ion()

'''
instantiate figure and axis objects.
    first two arguments are the number of vertical
     and horizontal axes in the figure, resp. '''
fig, ax = plt.subplots(2, 2, figsize=(9,7))

'''
we have four axes in one figure, accessed as an array:
    ax[0,0] - top left
    ax[0,1] - top right
    ax[1,0] - bottom left
    ax[1,1] - bottom right
 just like a numpy array. '''


plt.tight_layout() # just makes things look nicer.

P = np.array([[0, 1],   # an array of points
            [-1, 0],
            [0, -1],
            [1, 0]])

E = np.array([[0, 1],   # an array of edges as indices of points in P
            [1, 2],
            [2, 3],
            [3, 0]])

'''
first plot each edge individually.
    each edge e in E is a 1d array with two entries.
    We can iterate over E and use the following syntax to assert
     that each entry is a pair which we can call (u, v). '''
for (u, v) in E:
    '''
    ax.plot can be used to draw lines and uses the following syntax
        to draw a line between the points (u_x, u_y) and (v_x, v_y):
            ax.plot([u_x, v_x], [u_y, v_y]).
        recall that P is a 4x2 array.
         the x-coordinate of the point at index u is in P[u, 0]
         the y-coordinate of the point at index u in in P[u, 1]. '''
    ax[0,0].plot([P[u, 0], P[v, 0]], [P[u,1], P[v,1]])

'''
pyplot has a built-in color cycle, each item you plot in a sequence will
 be plotted with a color according to this color sequence.
Lets plot the edges while specifying the color in a slightly more elegant way.'''


for e in E:
    '''
    plot has a number of additional arguments, such as color.
    full list: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
                (the docs are your best friend)'''
    ax[0,1].plot(P[e, 0], P[e, 1], color='black')
    '''
    here, each e is still an array of indices [u, v] which
     we can pass to P in order to get a *slice* of P.
        P[:, 0] is the entire first column of P (the x-coordinates as a 4x1 array)
        P[:, 1] is the entire second column of P (the y-coordinates as a 4x1 array)
    Similarly, for e = [u, v]
        P[e, 0] = P[[u, v], 0] = [P[u, 0], P[v, 0]] is the x-coordinates of our edge
        P[e, 1] = P[[u, v], 1] = [P[u, 1], P[v, 1]] is the y-coordinates of our edge '''

'''
how far can we take slicing? '''
for uv_x, uv_y in P[E]:
    '''
    P[E] is a 4x2x2 array, so each entry is a pair of pairs of coordinates '''
    ax[1,0].plot(uv_x, uv_y, c=(1,0,0)) # color, abbv c, is given as an rgb tuple

'''
we can plot everything in one command (and as one object)
 by specifying our edges as a sequence of vertices '''
S = np.array([0,1,2,3,0]) # note, we have to return to 0 to close the loop
ax[1,1].plot(P[S,0], P[S,1]) # color does not need to be specified as the path is one object
