import torch

def graph_laplacian(env):
    '''
    https://en.wikipedia.org/wiki/Laplacian_matrix#Laplacian_matrix_for_simple_graphs
    '''
    W = torch.zeros((len(env.states)), len(env.states))
    for s in env.states:
        for a in env.actions:
            ns, r, _ = env.step(s, a)
            if s != ns:
                W[s, ns] = 1
    D = torch.diag(W.sum(0))
    return D - W

def fiedler(env):
    gl = graph_laplacian(env)
    sort_evals, sort_evecs = sorted_eig(gl)
    return sort_evecs[:, 1]

def lovasz_N(env):
    '''
    Compute `N` from Lovasz 1993. Related to Symmetric GL = I - N
    '''
    W = torch.zeros((len(env.states)), len(env.states))
    for s in env.states:
        for a in env.actions:
            ns, r, _ = env.step(s, a)
            if s != ns:
                W[s, ns] = 1
    Dinv = torch.diag(1/W.sum(0))
    return (Dinv**(1/2)) @ W @ (Dinv**(1/2))

def sorted_eig(matrix):
    evals, evecs = torch.eig(matrix, eigenvectors=True)
    sortidx = torch.argsort(-evals[:, 0]) # HACK assuming there are no complex components
    sort_evals, sort_evecs = evals[sortidx], evecs[:, sortidx]
    return sort_evals, sort_evecs
