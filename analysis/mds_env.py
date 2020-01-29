import matplotlib.pyplot as plt
import matplotlib
import diffplan
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

def plot_mds(env, *, projection=None, backend=None):
    # set to TkAgg?
    _3d = projection == '3d'
    ndim = 3 if _3d else 2

    if backend:
        be = matplotlib.get_backend()
        plt.switch_backend(backend)

    fig = plt.figure()
    if _3d:
        ax = fig.add_subplot(111, projection='3d')

    D = diffplan.compute_distance_matrix(env)
    Dmds = MDS(ndim, dissimilarity='precomputed').fit_transform(D)
    for a in range(D.shape[0]):
        for b in range(D.shape[1]):
            if a < b and D[a, b] == 1:
                plt.plot(*Dmds[[a,b]].T, 'k-')
    plt.plot(*Dmds.T, '.')
    plt.axis('equal')

    if backend:
        plt.show()
        plt.switch_backend(be)
