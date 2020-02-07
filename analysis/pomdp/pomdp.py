import torch
import numpy as np

def tiger(dtype=torch.float32, p = 0.85):
    T = torch.tensor([
        # s = 0, a = 0,1,2
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.0],
        ],
        # s = 1, a = 0,1,2
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    ]).type(dtype)

    O = torch.tensor([
        # a = 0, s' = 0,1
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        # a = 1, s' = 0,1
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        # a = 2, s' = 0,1
        [
            [p, 1-p],
            [1-p, p],
        ]
    ]).type(dtype)

    R = torch.tensor([
        # s = 0
        [+10, -100, -1],
        # s = 1
        [-100, +10, -1],
    ]).type(dtype)

    return T, O, R

def optimal_tiger_fsc():
    a = torch.tensor([2, 2, 0, 2, 1])
    s = torch.tensor([
        [1, 3],
        [2, 0],
        [0, 0],
        [0, 4],
        [0, 0],
    ])
    return a, s

def env_simple(
    *,
    switch=0.1,
    coherence=0.9,
    correct=+1,
    incorrect=-1,
):
    # s, a, s'
    T=torch.tensor([
        # s = 0
        [
            [1-switch, switch],
            [1-switch, switch],
        ],
        # s = 1
        [
            [switch, 1-switch],
            [switch, 1-switch],
        ],
    ])
    # a, s', o
    O = torch.tensor([
        #a=0
        [
            #s'=0
            [coherence, 1-coherence],
            #s'=1
            [1-coherence, coherence],
        ],
        #a=1
        [
            #s'=0
            [coherence, 1-coherence],
            #s'=1
            [1-coherence, coherence],
        ]
    ])
    R = torch.tensor([
        # s = 0
        [correct, incorrect],
        # s = 1
        [incorrect, correct],
    ]).float()
    return T, O, R

def env_simple_fsc():
    return (
        torch.tensor([0, 1]),
        torch.tensor([
            [0, 1],
            [0, 1],
        ]),
    )

def random_fsc(num_nodes, T, O, R, dtype=torch.float32, action_condition=False, return_logit=False):
    num_states, num_actions, _ = T.shape
    num_obs = O.shape[2]

    fsc_action_logit = torch.tensor(np.random.uniform(0, 1, size=(num_nodes, num_actions)))
    if action_condition:
        shape = (num_nodes, num_actions, num_obs, num_nodes)
    else:
        shape = (num_nodes, num_obs, num_nodes)
    fsc_state_logit = torch.tensor(np.random.uniform(0, 1, size=shape))

    if return_logit:
        return fsc_action_logit, fsc_state_logit

    fsc_action = fsc_action_logit.softmax(1).type(dtype)
    fsc_state = fsc_state_logit.softmax(-1).type(dtype)

    assert torch.allclose(fsc_action.sum(-1), torch.ones(num_nodes).type(dtype))
    assert torch.allclose(fsc_state.sum(-1), torch.ones(shape[:-1]).type(dtype))

    return fsc_action, fsc_state

def fsc_to_stochastic(fsc_action, fsc_state, T):
    num_states, num_actions, _ = T.shape
    num_nodes, num_obs = fsc_state.shape
    a = torch.zeros((num_nodes, num_actions))
    s = torch.zeros((num_nodes, num_obs, num_nodes))
    for n in range(num_nodes):
        a[n, fsc_action[n]] = 1.
        for o in range(num_obs):
            s[n, o, fsc_state[n, o]] = 1.
    return a, s


def policy_evaluation_stoch(fsc_action, fsc_state, T, O, R, gamma, *, debug=True, max_iter=5000, vi_eps=1e-3):
    num_states, num_actions, _ = T.shape
    num_nodes, num_obs = fsc_state.shape[0], O.shape[2]

    #if len(fsc_state.shape) == 4:
    #    fsc_state = fsc_state.sum(axis=1) # marginalizing over actoin, makes it p(n'|,n,o)
    fsc_state = ensure_fsc_nao(fsc_state, num_actions, dtype)

    V = torch.zeros((num_nodes, num_states))
    for idx in range(max_iter):
        prev = torch.clone(V)
        #'''
        for n in range(num_nodes):
            # (a, s')
            val = torch.sum(
                # This was computing when we were marginalizing out actions in this step
                # (a, s', o) * (a, None, o, n') = (a, s', o, n') -(sum(ax=2))> (a, s', n')
                (O[:, :, :, None] * fsc_state[n, :, None]).sum(axis=2) *

                # (a, s', o) @ (o, n') = (a, s', n')
                #O @ fsc_state[n] *
                V.T[None, :, :], axis=2)
            # (s, a) expected future value for state/action
            Q = torch.sum(T * val[None, :, :], axis=2)
            V[n] = (R + gamma * Q) @ fsc_action[n] # HAKKK 
            #V[n] = R @ fsc_action[n] + gamma * Q.sum(1) # HAKKK  # HAKKK  # HAKKK 
        #'''
        '''
        for n in range(num_nodes):
            # (a, s')
            val = torch.sum(
                # (a, s', o) @ (o, n') = (a, s', n')
                O @ fsc_state[n] *
                V.T[None, :, :], axis=2)
            # (s, a) expected future value for state/action
            Q = torch.sum(T * val[None, :, :], axis=2)
            V[n] = (R + gamma * Q) @ fsc_action[n]
        '''
        '''
        np = torch
        for n in range(num_nodes):
            # (a, s')
            val = np.sum(
                # (a, s', o) @ (o, n') = (a, s', n')
                O @ fsc_state[n] *
                V.T[None, :, :], axis=2)
            # (s, a) expected future value for state/action
            Q = np.sum(T * val[None, :, :], axis=2)
            V[n] = (R + gamma * Q) @ fsc_action[n]
        '''
        '''
        np = torch
        for n in range(num_nodes):
            for s in range(num_states):
                # (a, s')
                val = np.sum(
                    # (a, s', o) @ (o, n') = (a, s', n')
                    O @ fsc_state[n] *
                    V.T[None, :, :], axis=2)
                # (a) expected future value for actions
                val = np.sum(T[s] * val, axis=1)
                V[n, s] = fsc_action[n] @ (R[s] + gamma * val)
        '''
        '''
        for n in range(num_nodes):
            for s in range(num_states):
                V[n, s] = sum(
                    fsc_action[n, a] * (
                        R[s, a] + gamma * sum([
                            T[s, a, ns] * O[a, ns, :] @ fsc_state[n, :, :] @ V[:, ns]
                            for ns in range(num_states)
                        ])
                    )
                    for a in range(num_actions))
        '''
        '''
        for n in range(num_nodes):
            for s in range(num_states):
                V[n, s] = sum(fsc_action[n, a] * (R[s, a] + gamma * sum([
                    T[s, a, ns] * sum(
                        O[a, ns, o] * sum(fsc_state[n, o, nn] * V[nn, ns] for nn in range(num_nodes))
                        for o in range(num_obs)
                    )
                    for ns in range(num_states)
                ])) for a in range(num_actions))
        '''
        '''
        # deterministic
        for n in range(num_nodes):
            a = fsc_action[n]
            state_vals = np.sum(O[a] * V[fsc_state[n]].T, axis=1)
            V[n] = R[:, a] + gamma * T[:, a] @ state_vals
        '''
        '''
        for n in range(num_nodes):
            for s in range(num_states):
                a = fsc_action[n]
                nn = fsc_state[n]

                state_vals = np.sum(O[a] * V[fsc_state[n]].T, axis=1)
                V[n, s] = R[s, a] + gamma * T[s, a] @ state_vals
        '''
        if torch.norm(prev-V) < vi_eps:
            if debug:
                print('Converged', idx)
            break
    return V


def ensure_fsc_nao(fsc_state, num_actions, dtype):
    '''
    Ensures FSC's observation strategy is conditional on node, action, observation (nao).
    '''
    num_obs, num_nodes = fsc_state.shape[-2:]
    if len(fsc_state.shape) == 3:
        assert torch.allclose(fsc_state.sum(-1), torch.ones((num_nodes, num_obs)).type(dtype))
        sh = fsc_state.shape
        fsc_state = fsc_state.view(-1).repeat(num_actions).view((num_actions,)+sh).transpose(0,1)
        assert fsc_state.shape == (num_nodes, num_actions, num_obs, num_nodes)
    assert torch.allclose(fsc_state.sum(-1), torch.ones((num_nodes, num_actions, num_obs)).type(dtype))
    return fsc_state


def policy_evaluation_stoch_old(fsc_action, fsc_state, T, O, R, gamma, *, debug=True, max_iter=5000, vi_eps=1e-3):
    num_states, num_actions, _ = T.shape
    num_nodes, num_obs = fsc_state.shape[0], O.shape[2]

    #if len(fsc_state.shape) == 4:
    #    fsc_state = fsc_state.sum(axis=1) # marginalizing over actoin, makes it p(n'|,n,o)
    fsc_state = ensure_fsc_nao(fsc_state, num_actions, dtype)

    V = torch.zeros((num_nodes, num_states))
    for idx in range(max_iter):
        prev = torch.clone(V)
        for n in range(num_nodes):
            for s in range(num_states):
                V[n, s] = sum(fsc_action[n, a] * (R[s, a] + gamma * sum([
                    T[s, a, ns] * sum(
                        O[a, ns, o] * sum(fsc_state[n, a, o, nn] * V[nn, ns] for nn in range(num_nodes))
                        for o in range(num_obs)
                    )
                    for ns in range(num_states)
                ])) for a in range(num_actions))
        if torch.norm(prev-V) < vi_eps:
            if debug:
                print('Converged', idx)
            break
    return V


# http://www.ifaamas.org/Proceedings/aamas2015/aamas/p1249.pdf
# https://arxiv.org/pdf/1301.6720.pdf
def policy_evaluation_sr(fsc_action, fsc_state, T, O, R, gamma, *, dtype=torch.FloatTensor):
    num_states, num_actions, _ = T.shape
    num_nodes, num_obs = fsc_state.shape[0], O.shape[2]

    fsc_action = fsc_action.type(dtype)
    fsc_state = fsc_state.type(dtype)
    T = T.type(dtype)
    O = O.type(dtype)
    R = R.type(dtype)

    '''
    T # s, a, s' -> p(s'|s,a)
    O # a, s', o -> p(o|a,s')
    fsc_action # n,a -> p(a|n)
    fsc_state # n, a, o, n' -> p(a,n'|n,o) OR n, o, n' -> p(n'|n,o)

    # deriving p(n',s'|n,s)
    p(n',s') = \sum_a p(n', a, s')

    p(n', a, s') = p(n'|a, s') p(a, s')

    p(n'|s', a) = p(n'|n, o) p(o|a,s'), from FSC transitions & observation model

    p(a,s') = p(s'|a) p(a)
    p(a) = p(a|n), from FSC
    p(s'|a) = T(s, a, s') = p(s' | s, a), from transitions
    
    '''
    #if len(fsc_state.shape) == 4:
    #    fsc_state = fsc_state.sum(axis=1) # marginalizing over actoin, makes it p(n'|,n,o)
    fsc_state = ensure_fsc_nao(fsc_state, num_actions, dtype)
    assert torch.allclose(fsc_state.sum(-1), torch.ones((num_nodes, num_actions, num_obs)).type(dtype))

    assert torch.allclose(fsc_action.sum(-1), torch.ones(num_nodes).type(dtype))
    #assert torch.allclose(fsc_state.sum(-1), torch.ones((num_nodes, num_obs)).type(dtype))
    assert torch.allclose(T.sum(-1), torch.ones((num_states, num_actions)).type(dtype))
    assert torch.allclose(O.sum(-1), torch.ones((num_actions, num_states)).type(dtype))

    # we want p(n',s'|n,s)
    Tmu = torch.zeros((num_nodes, num_states, num_nodes, num_states)).type(dtype)
    for n in range(num_nodes):
        for s in range(num_states):
            a_ns = T[s] * fsc_action[n, :, None]
            assert torch.allclose(a_ns.sum(), torch.ones(1).type(dtype))
            # (a, s', o) @ (o, n') = (a, s', n'), p(n'|a, s')
            #p_nn = O @ fsc_state[n]
            # (a, s', o) * (a, None, o, n') = (a, s', o, n') -(sum(ax=2))> (a, s', n')
            p_nn = (O[:, :, :, None] * fsc_state[n, :, None]).sum(axis=2)
            assert torch.allclose(p_nn.sum(2), torch.ones(num_actions, num_states).type(dtype)), p_nn
            Tmu[n, s] = (a_ns[:, :, None] * p_nn).sum(0).T
            assert torch.allclose(Tmu[n, s].sum(), torch.ones(1).type(dtype))

    # expected reward, (nodes, states)
    Cmu = fsc_action@R.T

    crossprod = num_nodes * num_states

    sr = (torch.eye(crossprod).type(dtype) - gamma * Tmu.view((crossprod, crossprod))).inverse()
    V = sr @ Cmu.view(crossprod)
    return V.view((num_nodes, num_states))

def make_constraints_simple(node, V, T, O, R, gamma, *, qcoef=1e-4, dtype=torch.float32):
    # http://www.ifaamas.org/Proceedings/aamas2015/aamas/p1249.pdf
    # https://arxiv.org/pdf/1301.6720.pdf
    # http://papers.nips.cc/paper/2372-bounded-finite-state-controllers.pdf

    num_states, num_actions, _ = T.shape
    num_nodes = V.shape[0]
    num_obs = O.shape[2]

    V = V.type(dtype)
    T = T.type(dtype)
    O = O.type(dtype)
    R = R.type(dtype)

    canz_shape = torch.Size((num_actions, num_obs, num_nodes))

    args = canz_shape.numel() + 1

    canz_slice = slice(0, canz_shape.numel())
    delta_slice = slice(canz_shape.numel(), canz_shape.numel()+1)
    assert args == (canz_slice.stop-canz_slice.start) + (delta_slice.stop-delta_slice.start)
    #c_a_slice = slice(1, 1 + num_actions)

    G = torch.zeros((V.shape[1] + canz_shape.numel(), args)).type(dtype)
    h = torch.cat((
        -V[node],
        torch.zeros(canz_shape.numel()).type(dtype),
    ))

    # Last spots encode c >= 0 as -c <= 0
    for idx in range(canz_shape.numel()):
        G[num_states+idx, idx] = -1

    for s in range(num_states):
        G[s, delta_slice] = 1
        canz_coef = G[s, canz_slice].view(canz_shape)

        for a in range(num_actions):
            for o in range(num_obs):
                for nn in range(num_nodes):
                    canz_coef[a, o, nn] = -(R[s, a] + gamma * (T[s, a, :] * O[a, :, o] * V[nn]).sum())

    # Q has to be PSD... so I've scaled it down considerably here
    Q = torch.eye(args).type(dtype) * qcoef
    p = torch.zeros(args).type(dtype)
    # The default optimization is an argmin, so this lets us maximize the delta!
    p[delta_slice] = -1

    # HACK HACK in the future, write these so we don't need c_a?? to be simpler??
    A = torch.zeros((1, args)).type(dtype)
    b = torch.zeros(1).type(dtype)

    # \sum_a c_a_n_z = 1
    A[0, canz_slice] = 1
    b[0] = 1

    def unpack(soln):
        canz = soln[canz_slice].view(canz_shape)
        c_a = canz.sum(axis=(1, 2))
        assert c_a.shape == (num_actions,)
        return (
            soln[delta_slice].item(),
            c_a,
            canz,
        )

    return (Q, p, G, h, A, b), unpack


def make_constraints(node, V, T, O, R, gamma, *, qcoef=1e-4, dtype=torch.float32):
    # http://www.ifaamas.org/Proceedings/aamas2015/aamas/p1249.pdf
    # https://arxiv.org/pdf/1301.6720.pdf
    # http://papers.nips.cc/paper/2372-bounded-finite-state-controllers.pdf

    num_states, num_actions, _ = T.shape
    num_nodes = V.shape[0]
    num_obs = O.shape[2]

    V = V.type(dtype)
    T = T.type(dtype)
    O = O.type(dtype)
    R = R.type(dtype)

    canz_shape = torch.Size((num_actions, num_obs, num_nodes))

    args = 1 + num_actions + canz_shape.numel()

    delta_slice = slice(0, 1)
    c_a_slice = slice(1, 1 + num_actions)
    canz_slice = slice(1 + num_actions, None)

    G = torch.zeros((V.shape[1] + num_actions + canz_shape.numel(), args)).type(dtype)
    h = torch.cat((
        -V[node],
        torch.zeros(num_actions + canz_shape.numel()).type(dtype),
    ))

    # Last spots encode c >= 0 as -c <= 0
    for idx in range(num_actions + canz_shape.numel()):
        G[num_states+idx, delta_slice.stop+idx] = -1

    for s in range(num_states):
        G[s, delta_slice] = 1
        G[s, c_a_slice] = -R[s]
        canz_coef = G[s, canz_slice].view(canz_shape)

        # (a, s', o)
        #xx = T[s, :, :, None]*O
        for a in range(num_actions):
            for o in range(num_obs):
                for nn in range(num_nodes):
                    canz_coef[a, o, nn] = -gamma * (T[s, a, :] * O[a, :, o] * V[nn]).sum()

    # Q has to be PSD... so I've scaled it down considerably here
    Q = torch.eye(args).type(dtype) * qcoef
    p = torch.zeros(args).type(dtype)
    # The default optimization is an argmin, so this lets us maximize the delta!
    p[delta_slice] = -1

    # HACK HACK in the future, write these so we don't need c_a?? to be simpler??
    A = torch.zeros((num_actions+1, args)).type(dtype)
    b = torch.zeros(A.shape[0]).type(dtype)

    # \sum_a c_a = 1
    A[num_actions, c_a_slice] = 1
    b[num_actions] = 1

    # for each a, \sum_a c_a_n_z = c_a, or \sum_a c_a_n_z - c_a = 0
    for a in range(num_actions):
        A[a, c_a_slice][a] = -1
        A[a, canz_slice].view(canz_shape)[a] = 1

    def unpack(soln):
        return (
            soln[0].item(),
            soln[c_a_slice],
            soln[canz_slice].view(canz_shape),
        )

    return (Q, p, G, h, A, b), unpack


def policy_iteration_grad(env, gamma, *, lr=0.1, grad_steps=100, progress=10, dtype=torch.float64, num_nodes=2):
    T, O, R = env

    fsc_action_logit, fsc_state_logit = random_fsc(num_nodes, T, O, R, dtype=dtype, return_logit=True)
    fsc_action_logit = fsc_action_logit.clone().detach().requires_grad_(True)
    fsc_state_logit = fsc_state_logit.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([fsc_action_logit, fsc_state_logit], lr=lr)

    value = lambda: policy_evaluation_sr(
        fsc_action_logit.softmax(-1),
        fsc_state_logit.softmax(-1),
        T, O, R, gamma, dtype=dtype
    )[0].mean()

    for idx in range(grad_steps):
        opt.zero_grad()

        loss = -value()
        loss.backward(retain_graph=True)

        if ((idx+1) % progress) == 0:
            print(idx, loss.item())

        opt.step()

    return dict(
        fsc_action_logit=fsc_action_logit,
        fsc_state_logit=fsc_state_logit,
        fsc_action=fsc_action_logit.softmax(-1),
        fsc_state=fsc_state_logit.softmax(-1),
    )