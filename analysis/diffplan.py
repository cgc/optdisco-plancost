import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import graphviz
import heapq
import itertools

class Line(object):
    def __init__(self, size=4):
        self.size = size
        self.states = range(size)
        self.states_to_idx = {s: si for si, s in enumerate(self.states)}

        self.actions = [-1, +1]

        self.start_states = [0]
        self.goal_set = set([size-1])
    def step(self, s, a):
        if (
            (s == 0 and a == -1) or
            (s == self.size-1 and a == +1)
        ):
            return s, -1, 0.0
        snext = s + a
        if snext in self.goal_set:
            return snext, +10, 1.0
        else:
            return snext, -1, 0.0

class Grid(object):
    def __init__(self, grid_string, *, step_reward=-0.1, goal_reward=+100):
        self.grid_string = grid_string
        self.npg = np.array([list(line) for line in grid_string.split('\n') if line.strip('\n')])
        self.states_features = list(zip(*np.where(self.npg!='x')))
        self.states = range(len(self.states_features))
        self.states_to_idx = {s: si for si, s in enumerate(self.states_features)}
        self.goal_set = set(zip(*np.where(self.npg=='G')))
        self.start_states = [self.states_to_idx[s] for s in list(zip(*np.where(self.npg=='S')))]
        self.actions = [
            (0, +1),
            (0, -1),
            (+1, 0),
            (-1, 0),
        ]
        self.step_reward = step_reward
        self.goal_reward = goal_reward
    def step(self, si, a):
        '''
        Return: next state, reward, termination probability.
        '''
        s = self.states_features[si]
        s_next = (s[0]+a[0], s[1]+a[1])
        if s_next not in self.states_to_idx:
            return si, self.step_reward, 0.0
        else:
            s_nexti = self.states_to_idx[s_next]
            if s_next in self.goal_set:
                return s_nexti, self.goal_reward, 1.0
            else:
                return s_nexti, self.step_reward, 0.0

'''
class Problem(object):
    def __init__(self, num_states, actions, initial_states):
        self.states = range(num_states)
        self.actions = actions
        self.initial_states = initial_states
        
    def actions(self, state):
        assert False
    def result(self, state, action):
        assert False
    def goal_test(self, state):
        assert False
'''

def softmax(z):
    return z.softmax(dim=0)
    # HACK
    e = torch.exp(z - torch.max(z))
    return e / torch.sum(e)

def env_to_torch(env):
    '''
    Converts transition, reward, and termination probability of each (state, action)
    pair into torch vectors to simplify value iteration.
    '''
    return {
        s: (
            torch.LongTensor([env.step(s, a)[0] for a in env.actions]),
            torch.Tensor([env.step(s, a)[1] for a in env.actions]),
            torch.Tensor([env.step(s, a)[2] for a in env.actions])
        )
        for s in env.states
    }

eps = torch.finfo().eps

def vi_vec_old(
    env, betas, *, max_iter=50, vi_eps=1e-5, debug=False,
    max_ent=False,
    max_ent_no_beta=False,
    max_ent_other=False,
    max_ent_other_logsoft=False,
):
    if betas.shape == torch.Size([1]):
        betas = betas.repeat(len(env.states))
    if max_iter < len(env.states):
        print('WARNING: max iter smaller than # states')
    next_states, rewards, terminations = (
        torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states]),
        torch.Tensor([[env.step(s, a)[1] for a in env.actions] for s in env.states]),
        torch.Tensor([[env.step(s, a)[2] for a in env.actions] for s in env.states]),
    )# dimensions are (# states, # actions)
    V = torch.zeros(len(env.states))
    for _ in range(max_iter):
        prev = V
        Q = rewards+(1-terminations)*V[next_states]
        policy = (betas[:, None]*Q).softmax(dim=1)
        V = torch.sum(policy*Q, dim=1)
        if max_ent_no_beta:
            V = V - torch.sum(policy*torch.log(policy+eps), axis=1)
        elif max_ent:
            V = V - torch.sum(policy*torch.log(policy+eps), axis=1) / betas
        elif max_ent_other_logsoft:
            V = V - torch.sum(policy*(betas[:, None]*Q).log_softmax(dim=1), axis=1) / betas
        elif max_ent_other: # HACK delete
            #policy = (betas[:, None]*Q).softmax(dim=1)
            z = betas[:, None]*Q
            #z = z_ - torch.max(z_, axis=1).values[:, None]
            zz = z - torch.max(z, axis=1).values[:, None]
            e = torch.exp(zz)
            #e = torch.exp(z)
            #policy = e / torch.sum(e, axis=1)
            #logpolicy = torch.log(policy+eps)
            #logpolicy_div_beta = torch.log(torch.exp(z) / torch.sum(torch.exp(z), axis=1) + eps) / betas
            #print(Q.shape, z.shape, torch.sum(torch.exp(z), axis=1).shape, betas.shape)
            #print(torch.any(torch.isnan(z)), z[30], Q[30])
            #logpolicy_div_beta = (zz - torch.log(eps+torch.sum(torch.exp(zz), axis=1)[:, None])) / betas[:, None] # !!
            logpolicy_div_beta = zz/betas[:, None] - (torch.log(eps+torch.sum(e, axis=1))/betas)[:, None] # !!
            #logpolicy_div_beta = torch.log(eps+e)/betas[:, None] - (torch.log(eps+torch.sum(e, axis=1))/betas)[:, None]

            #logpolicy_div_beta = zz/betas[:, None] - (torch.log(eps+torch.sum(e, axis=1))/betas)[:, None] # !!
            #logpolicy_div_beta = (zz - torch.log(eps+torch.sum(e, axis=1))[:, None])/betas[:, None] # !!
            V = V - torch.sum(policy*logpolicy_div_beta, axis=1)
        if torch.norm(prev-V) < vi_eps:
            break
    J = -torch.mean(V[torch.tensor(env.start_states)])
    if debug:
        Q = rewards+(1-terminations)*V[next_states]
        policy = (betas[:, None]*Q).softmax(dim=1)
        return J, V, policy
    return J

def vi_vec(
    env, betas, *, max_iter=50, vi_eps=1e-5, debug=False,
    max_ent=False,
    fixed_policy=None,
    fixed_cost=None,
    max_ent_coef=None,
):
    if betas.shape == torch.Size([1]):
        betas = betas.repeat(len(env.states))
    if max_iter < len(env.states):
        print('WARNING: max iter smaller than # states')
    next_states, rewards, terminations = (
        torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states]),
        torch.Tensor([[env.step(s, a)[1] for a in env.actions] for s in env.states]),
        torch.Tensor([[env.step(s, a)[2] for a in env.actions] for s in env.states]),
    )# dimensions are (# states, # actions)
    V = torch.zeros(len(env.states))
    for _ in range(max_iter):
        prev = V
        Q = rewards+(1-terminations)*V[next_states]
        if fixed_policy is None:
            policy = (betas[:, None]*Q).softmax(dim=1)
        else:
            policy = fixed_policy
        V = torch.sum(policy*Q, dim=1)
        if max_ent and max_ent_coef is not None:
            V = V - max_ent_coef / betas * torch.sum(policy*torch.log(policy+eps), axis=1)
        elif max_ent:
            V = V - torch.sum(policy*torch.log(policy+eps), axis=1) / betas
        elif fixed_cost is not None:
            V = V - fixed_cost
        if torch.norm(prev-V) < vi_eps:
            break
    J = -torch.mean(V[torch.tensor(env.start_states)])
    if debug:
        return J, V, policy, Q
    return J

def vi(env, betas, *, max_iter=50, vi_eps=1e-5, debug=False):
    if max_iter < len(env.states):
        print('WARNING: max iter smaller than # states')
    torch_env = env_to_torch(env)
    V = torch.zeros(len(env.states))
    for _ in range(max_iter):
        prev = V
        for s in env.states:
            next_states, rewards, terminations = torch_env[s]
            Q = rewards+(1-terminations)*V[next_states]
            policy = softmax(betas[s]*Q)
            V = V.clone()
            V[s] = policy@Q
        if torch.norm(prev-V) < vi_eps:
            break
    J = -torch.mean(V[torch.tensor(env.start_states)])
    if debug:
        return J, V
    return J


def p2p(
    env,
    betas,
    lambda_,
    *,
    default_policy=None,
    vi_eps=1e-5,
    max_iter=50,
    discount=0.99,
    simulated_policy_penalty=None,
):
    num_states = len(env.states)
    if default_policy is None:
        uniform = torch.ones(len(env.actions))/len(env.actions)
        default_policy = uniform

    state_res = env_to_torch(env)

    def _q_policy_for_state(V, s):
        next_states, rewards, terminations = state_res[s]
        Q = rewards+(1-terminations)*discount*V[next_states]
        return Q

    # First we compute the policy for each state
    state_policy = []
    for root in env.states:
        b = betas[:, root]
        V = torch.zeros(num_states, requires_grad=True)
        for _ in range(max_iter):
            Vprev = V
            for s in env.states:
                Q = _q_policy_for_state(V, s)
                policy = softmax(b[s]*Q)
                V = V.clone()
                if simulated_policy_penalty is not None:
                    Q = Q + simulated_policy_penalty*1/b[s]*torch.log(policy)
                V[s] = policy@Q
            if torch.norm(Vprev-V) < vi_eps:
                break
        # Computing policy cost
        policy_cost = 0
        for s in env.states:
            Q = _q_policy_for_state(V, s)
            policy = softmax(b[s]*Q)
            policy_cost += torch.sum(policy*torch.log(policy/default_policy))
            if s == root:
                curr_state_policy = policy
        state_policy.append((curr_state_policy, policy_cost))

    # Then we do value iteration for our final policy
    V = torch.zeros(num_states, requires_grad=True)
    for _ in range(max_iter):
        Vprev = V
        for s in env.states:
            Q = _q_policy_for_state(V, s)
            V = V.clone()
            policy, policy_cost = state_policy[s]
            V[s] = policy@Q - lambda_*policy_cost
        if torch.norm(Vprev-V) < vi_eps:
            break
    return -torch.mean(V[torch.tensor(env.start_states)])

from matplotlib.colors import LinearSegmentedColormap
def new_opacity_cmap(color):
    '''
    This function returns a colormap that linearly varies the opacity of the supplied color for inputs.
    '''
    return LinearSegmentedColormap.from_list(f'OpCo({color[:3]})', [color[:3]+(0.0,), color[:3]+(1.0,)])

def plot_grid(env, state_prop, f=None, ax=None, vmin=None, vmax=None, cmap='Greens', labels=None, vcenter=False, start_states=None, colorbar=True, goal_set=None):
    v = np.full(env.npg.shape, np.nan)
    for si, s in enumerate(env.states_features):
        v[s] = state_prop[si]
    if f is None:
        f, ax = plt.subplots()
    if vcenter:
        vmax = state_prop.abs().max()
        vmin = -vmax
    im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax)
    #s = env.states_features[sidx]
    #plt.plot(s[1], s[0], '*', c='w')
    for g in goal_set or env.goal_set:
        ax.plot(g[1], g[0], '*', c='g')
    if colorbar:
        f.colorbar(im, ax=ax)
    if labels is not None:
        for si, s in enumerate(env.states_features):
            y, x = s
            ax.annotate(labels[si], (x-0.4, y))
    if start_states is not None:
        for si in start_states:
            s = env.states_features[si]
            ax.plot(s[1], s[0], marker='o', c='k', fillstyle='none')

    for patch in env_rect(env):
        ax.add_patch(patch)

def plot_policy(env, V, policy, max_action=True, figsize=(10, 6)):
    f, ax = plt.subplots(figsize=figsize)
    plot_grid(env, V, f, ax)
    scale=0.5 # HACK b/c we can only scale to sides of square.

    kw = dict(fc='k', lw=0.5, length_includes_head=True)
    hw, hl = 0.2, 0.2
    #if max_action:
    #    # Looks bad to have arrows for other cases
    #    kw = dict(kw, head_width=0.2, head_length=0.2)

    for si in env.states:
        y, x = env.states_features[si]
        if max_action:
            maxidx = np.argmax(policy[si])
            p_a = policy[si, maxidx]
            dy, dx = env.actions[maxidx]
            ax.arrow(x, y, scale*p_a*dx, scale*p_a*dy, head_width=p_a*hw, head_length=p_a*hl, **kw)
        else:
            for aidx, a in enumerate(env.actions):
                p_a = policy[si, aidx]
                dy, dx = env.actions[aidx]
                ax.arrow(x, y, scale*p_a*dx, scale*p_a*dy, head_width=p_a*hw, head_length=p_a*hl, **kw)

def plot_beta(plt, env, betas, sidx, vmax_fixed=False):
    vmax = np.quantile(betas.detach().numpy(), 0.995) if vmax_fixed else None
    betasx = np.full(env.npg.shape, np.nan)
    for si, s in enumerate(env.states_features):
        betasx[s] = betas[si,sidx].detach().numpy()
    #plt.imshow(np.where(np.round(betasx, 2)<0.05, 0, np.round(betasx, 2)))
    plt.figure()
    plt.imshow(betasx, vmin=0, vmax=vmax, cmap='Greens')
    s = env.states_features[sidx]
    plt.plot(s[1], s[0], '*', c='w')
    g = list(env.goal_set)[0]
    plt.plot(g[1], g[0], '*', c='g')
    plt.colorbar()

def plot_betas(plt, env, betas):

    for sidx in env.start_states:
        plot_beta(plt, env, betas, sidx)

# Utilities for computing

def make_T(env, policy, terminations):
    '''
    Computing the transition matrix under the policy and the SR
    '''
    next_states = torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states])
    T = torch.zeros((len(env.states), len(env.states)))
    for s in env.states:
        # HACK have to do a loop instead of vectorization if two+ actions can lead to the same state.
        for aidx, next_s in enumerate(next_states[s]):
            T[s, next_s] += policy[s, aidx]
    sr = torch.inverse(torch.eye(len(env.states)) - T * (1-terminations[None, :]))
    return T, sr

class SubgoalGrid(Grid):
    '''
    In a Subgoal Grid, we keep transition dynamics, but replace terminations & make all actions have same cost.
    '''
    def __init__(self, grid_string, terminations, *, step_reward=-0.1):
        super().__init__(grid_string, step_reward=step_reward)
        self.terminations = terminations
    def step(self, si, a):
        s_nexti, _, _ = super().step(si, a)
        return s_nexti, self.step_reward, self.terminations[si]

# Computing option termination state conditioned on start state.

def _recursive_p_f_given_s(env, terminations, T, max_iter=None, eps=1e-5):
    max_iter = max_iter or len(env.states)*3 
    p_f_given_s = terminations.clone().repeat((len(env.states), 1))
    state_idxs = torch.arange(len(env.states))
    for it in range(max_iter):
        prev = p_f_given_s.clone()
        for s in env.states:
            p_f_given_s[s] = (
                terminations[s] * (state_idxs==s).float() +
                (1-terminations[s]) * T[s] @ p_f_given_s
            )
        if torch.norm(prev-p_f_given_s) < eps:
            pass
    return p_f_given_s

def _sr_p_f_given_s(env, terminations, T, sr, discount_initial=True):
    '''
    This is sort of a cute version that relies on the fact that expected occupancy z_{s,s'} is
    related to p^o(s'|s) by termination/(1-termination) = p^o(s'|s)/z_{s,s'}.
    However, this version breaks down for deterministic terminations.
    '''
    ident = torch.eye(len(env.states))

    sr = sr - ident # remove one from diagonal to account for undiscounted initial state.

    p_f = sr * terminations[None, :] / (1-terminations[None, :])
    if discount_initial:
        p_f = terminations[:, None] * ident + (1-terminations[:, None]) * p_f
    return p_f

def _sr_transition_p_f_given_s(env, terminations, T, sr, discount_initial=True):
    '''
    This is a pretty intuitive version that tries to count all the ways to reach a terminating
    state from a start state. SR gives expected # of visits. For each of those visits, T gives
    probability of transitioning to the terminating state. The product of SR visits, probability
    of transitioning, and probability of terminating gives the probabilty of terminating in the
    final state.
    '''
    ident = torch.eye(len(env.states))
    p_f = torch.sum(sr[:, :, None]*T[None, :, :], axis=1) * terminations
    if discount_initial:
        p_f = terminations[:, None] * ident + (1-terminations[:, None]) * p_f
    return p_f

# Computing reward under some terminations & policy.
# (Though these should be considered defunct since values from value iteration for a subgoal are equivalent.)

def _recursive_r_o(env, terminations, T, action_cost=-1, max_iter=None, eps=1e-5):
    max_iter = max_iter or len(env.states)*30
    next_states = torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states])
    r = torch.zeros(len(env.states))
    for _ in range(max_iter):
        prev = r.clone()
        for s in env.states:
            next_s = next_states[s]
            r[s] = (
                # HACK the first two versions had weird convergence issues. I wonder why?
                # HACK because this wasn't computing next_states before?
                #(1-terminations[s]) * (action_cost + T[s, next_s] @ r[next_s])
                #(1-terminations[s]) * (action_cost + r[next_s]) @ T[s, next_s]
                action_cost + (1-terminations[s]) * T[s, :] @ r
            )
        if torch.norm(prev-r) < eps:
            break
    return r

def _sr_r_o(terminations, sr, action_cost=-1, discount_initial=True):
    r_o = (sr*action_cost).sum(axis=1)
    if discount_initial:
        return action_cost + (1-terminations)*r_o
    return r_o

def option_planner_vi(
    env, betas, terminations, *,
    max_iter=None,
    vi_eps=1e-5,
    option_cost_coef=0.,
    max_ent=False,
    discount=1.0,
    option_creation_cost=None, # Returns tensor size (# states, # options)
    state_cost=None, # Returns tensor size (# states)
    option_execution_cost=None, # Returns scalar
):
    max_iter = max_iter or len(env.states) * 5
    num_options = betas.shape[0]
    # HACK clamp?

    goal_set = [env.states_to_idx[g] for g in env.goal_set]
    task_terminations = torch.tensor([s in goal_set for s in env.states]).float()

    p_f = [None]*num_options
    r_o = [None]*num_options
    policy = [None]*num_options
    Vs = [None]*num_options
    sr = [None]*num_options
    #entropy = torch.zeros(num_options)

    # Get option policies
    for o in range(num_options):
        subgoal_env = SubgoalGrid(env.grid_string, terminations[o])
        value, V, policy[o], Q = vi_vec(
            subgoal_env, betas[o],
            debug=True, max_ent=max_ent, max_iter=max_iter)
        # HACK HACK should we consider how the policy works given the tasks' terminations??
        #term = task_terminations + (1-task_terminations) * terminations[o]
        term = terminations[o]
        T, sr[o] = make_T(subgoal_env, policy[o], term)
        p_f[o] = _sr_transition_p_f_given_s(subgoal_env, term, T, sr[o], discount_initial=True)
        # HACK these are old versions
        #T, sr[o] = make_T(subgoal_env, policy[o], terminations[o])
        #p_f[o] = _sr_transition_p_f_given_s(subgoal_env, terminations[o], T, sr[o], discount_initial=True)

        # prefer V over computing r_o
        #r_o[o] = _sr_r_o(terminations[o], sr[o], action_cost=env.step_reward, discount_initial=True)
        r_o[o] = V
        #entropy[o] = -(policy[o] * torch.log(eps+policy[o])).sum()
        Vs[o] = V

    # Compute costs
    kw = dict(env=env, num_options=num_options, policy=policy, sr=sr, terminations=terminations)
    oec = option_execution_cost(**kw) if option_execution_cost else torch.zeros((len(env.states), num_options))
    sc = state_cost(**kw) if state_cost else torch.zeros((len(env.states)))
    occ = option_creation_cost(**kw) if option_creation_cost else 0.
    '''
    def option_execution_cost(*, env=None, num_options=None, policy=None, **kwargs):
        entropy = torch.zeros((len(env.states), num_options))
        for o in range(num_options):
            entropy[:, o] = -(policy[o]*torch.log(eps+policy[o])).sum(axis=1)
        return entropy
    def option_execution_cost(*, env=None, num_options=None, policy=None, sr=None, **kwargs):
        expected_visit_weighted_entropy = torch.zeros((len(env.states), num_options))
        for o in range(num_options):
            entropy = -(policy[o]*torch.log(eps+policy[o])).sum(axis=1)
            expected_visit_weighted_entropy[:, o] = sr[o] @ entropy
        return expected_visit_weighted_entropy
    def state_cost(*, **kwargs):
        return option_execution_cost(**kwargs).sum(axis=1)
    def option_creation_cost(*, **kwargs):
        return option_execution_cost(**kwargs).sum()
    '''

    # Get meta-policy
    V = torch.zeros(len(env.states))
    for idx in range(max_iter):
        prev = V
        Q = torch.zeros((len(env.states), num_options))
        for o in range(num_options):
            Q[:, o] = r_o[o] + discount * (1-task_terminations) * p_f[o]@V
        Q = Q + oec
        V = Q.max(axis=1).values + sc
        if torch.norm(V-prev) < vi_eps:
            break
    #J = -torch.mean(V[torch.tensor(env.start_states)]) - option_cost_coef * entropy.sum()
    J = -(torch.mean(V[torch.tensor(env.start_states)]) + occ)
    return J, V, Vs, Q, policy
    # A state should pay for the entropy it uses. so unused options should not be counted?

def compute_distance_matrix(env, eps=1e-5, max_iter=None):
    # Need something larger than any distance.
    large_multiplier = 10
    max_iter = max_iter or len(env.states)*large_multiplier
    next_states = torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states])

    distance = torch.zeros((len(env.states), len(env.states))) + len(env.states) * large_multiplier
    distance[torch.arange(len(env.states)), torch.arange(len(env.states))] = 0
    #distance[torch.eye(len(env.states))] = 0

    for _ in range(max_iter):
        prev = distance.clone()
        for s in env.states:
            for ss in env.states:
                nss = next_states[ss]
                distance[s, nss] = torch.min(distance[s, nss], distance[s, ss] + 1)
        if (prev-distance).norm()<eps:
            break

    return distance

def expected_bfs_cost(env, distance, termination_idx):
    # We estimate the number of nodes that would be visited
    # in breadth-first search. For a node s at distance d from goal state,
    # we know this is the number of nodes with distance<d
    # plus half the nodes with distance==d.
    bfs_cost = torch.zeros(len(env.states))
    for s in env.states:
        d = distance[s, termination_idx]
        bfs_cost[s] = (
            (distance[s]<d).sum() +
            # We add one since we always have at least one comparison.
            ((distance[s] == d).sum()+1.)/2
        )
    return bfs_cost

def compute_bfs_matrix(env, distance):
    BFS = torch.zeros((len(env.states), len(env.states)))
    for s_term in env.states:
        BFS[:, s_term] = expected_bfs_cost(env, distance, s_term)
    return BFS

def dfs(env, start_state, goal, *, shuffle=True, append_queue_entries=False):
    '''
    append_queue_entries: When true, nodes are appended to the queue, even if
    they're in the queue already. This might use more memory, but can ensure
    a promising node encountered early (but not first) isn't ignored.
    '''
    actions = np.arange(len(env.actions))
    visited = set()
    queue = [start_state]
    distance = {start_state: 0}

    while queue:
        n = queue.pop()
        if n in visited:
            continue
        visited.add(n)
        if n == goal:
            return distance[n], len(visited)
        if shuffle:
            random.shuffle(actions)
        for aidx in actions:
            nn, _, _ = env.step(n, env.actions[aidx])
            if (append_queue_entries or nn not in queue) and nn not in visited:
                queue.append(nn)
                distance[nn] = distance[n] + 1

def compute_dfs_matrix(env, *, samples=100, tqdm=lambda x: x):
    distance = torch.zeros((len(env.states), len(env.states)))
    DFS = torch.zeros((len(env.states), len(env.states)))
    for s in tqdm(env.states):
        for g in env.states:
            dist, plan_cost = 0., 0.
            for _ in range(samples):
                r = dfs(env, s, g, shuffle=True)
                dist += r[0]
                plan_cost += r[1]
            dist, plan_cost = dist/samples, plan_cost/samples
            distance[s, g] = dist
            DFS[s, g] = plan_cost
    return distance, DFS

def option_planner_bfs(
    # should truly be a cost that we intend to minimize, as it is negated below.
    env, terminations, search_cost, *,
    # standard VI arguments
    max_iter=None,
    vi_eps=1e-5,
    #C=1.,
    discount=1.,
    meta_beta=1.,
    terminal_reward=0.,
    #cost_scaling=1.,
):
    max_iter = max_iter or len(env.states) * 5
    num_options = terminations.shape[0]

    goal_set = {env.states_to_idx[g] for g in env.goal_set}
    eta = terminations.softmax(dim=1) # (options, states)

    # Get meta-policy
    V = torch.zeros(len(env.states))
    for idx in range(max_iter):
        prev = V
        Q = torch.zeros((len(env.states), num_options))
        for s in env.states:
            if s in goal_set:
                Q[s] = terminal_reward
                continue
            #cost = C*BFS[s] + D[s]
            # HACK discount should only apply to V
            #Q[s] = discount*eta@(V - cost_scaling*search_cost[s])
            Q[s] = eta@(discount*V - search_cost[s])
        V = (meta_beta*Q).logsumexp(axis=1)/meta_beta
        if torch.norm(V-prev) < vi_eps:
            break
    J = -torch.mean(V[torch.tensor(env.start_states)])
    policy = (meta_beta*Q).softmax(dim=1)
    return J, V, Q, policy, eta

def option_planner_bfs_vec(
    # should truly be a cost that we intend to minimize, as it is negated below.
    env, terminations, search_cost, *,
    # standard VI arguments
    max_iter=None,
    vi_eps=1e-5,
    discount=1.,
    meta_beta=1.,
    terminal_reward=0.,
):
    max_iter = max_iter or len(env.states) * 5
    num_options = terminations.shape[0]

    goals = torch.LongTensor([1 if env.states_features[s] in env.goal_set else 0 for s in env.states]).bool()
    eta = terminations.softmax(dim=1) # (options, states)

    # Get meta-policy
    V = torch.zeros(len(env.states))
    for _ in range(max_iter):
        prev = V
        Q = (discount*V[None, :] - search_cost)@eta.T
        Q[goals, :] = terminal_reward
        V = (meta_beta*Q).logsumexp(axis=1)/meta_beta
        if torch.norm(V-prev) < vi_eps:
            break
    J = -torch.mean(V[torch.tensor(env.start_states)])
    policy = (meta_beta*Q).softmax(dim=1)
    return J, V, Q, policy, eta

def make_transitions_under_policy(env, policy):
    next_states = torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states])
    T = torch.zeros((len(env.states), len(env.states)))
    for s in env.states:
        # HACK have to do a loop instead of vectorization if two+ actions can lead to the same state.
        for aidx, next_s in enumerate(next_states[s]):
            T[s, next_s] += policy[s, aidx]
    return T

def make_sr(env, T, terminations):
    sr = torch.inverse(torch.eye(len(env.states)) - T * (1-terminations[None, :]))
    return sr

class SubgoalGrid2(Grid):
    '''
    In a Subgoal Grid, we keep transition dynamics, but replace terminations & make all actions have same cost.
    '''
    def __init__(self, grid_string, goal, *, step_reward=-0.1):
        super().__init__(grid_string, step_reward=step_reward)
        self.goal = goal
    def step(self, si, a):
        s_nexti, _, _ = super().step(si, a)
        # HACK HACK is this the right logic?
        return s_nexti, self.step_reward, (1.0 if self.goal == s_nexti and si != s_nexti else 0.0)

def vi_vec_maxent(
    env, betas, *, max_iter=None, vi_eps=1e-5, debug=False,
    max_ent=False,
    fixed_policy=None,
    fixed_cost=None,
    max_ent_coef=None,
    step_reward=-1,
    terminal_reward=0,
):
    if betas.shape == torch.Size([1]):
        betas = betas.repeat(len(env.states))
    max_iter = max_iter or len(env.states)
    next_states, _rewards, terminations = (
        torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states]),
        torch.Tensor([[env.step(s, a)[1] for a in env.actions] for s in env.states]),
        torch.Tensor([[env.step(s, a)[2] for a in env.actions] for s in env.states]),
    )# dimensions are (# states, # actions)
    goals = torch.LongTensor([
        1 if env.states_features[s] in env.goal_set else 0
        for s in env.states
    ]).bool()
    V = torch.zeros(len(env.states))
    for _ in range(max_iter):
        prev = V
        #Q = step_reward+(1-terminations)*V[next_states]
        Q = step_reward+V[next_states]
        Q[goals] = step_reward + terminal_reward
        #policy = (betas[:, None]*Q).softmax(dim=1)
        #V = torch.sum(policy*Q, dim=1) - torch.sum(policy*torch.log(policy+eps), axis=1) / betas
        V = (betas[:, None]*Q).logsumexp(axis=1)/betas
        if torch.norm(prev-V) < vi_eps:
            break
    J = -torch.mean(V[torch.tensor(env.start_states)])
    policy = (betas[:, None]*Q).softmax(dim=1)
    return J, V, policy, Q

def compute_bfs_search_cost(env, C=1.):
    if not hasattr(env, 'D'):
        # We cache the distance matrix since it's usually a bit expensive...
        env.D = compute_distance_matrix(env)

    BFS = torch.zeros((len(env.states), len(env.states)))
    for s_term in env.states:
        BFS[:, s_term] = expected_bfs_cost(env, env.D, s_term)
    env.BFS = BFS

    return C*BFS + env.D

def compute_random_walk_search_cost(env):
    search_cost = torch.zeros((len(env.states), len(env.states)))
    uniform_policy = torch.ones((len(env.states), len(env.actions))) / len(env.actions)
    T = make_transitions_under_policy(env, uniform_policy)
    for goal in env.states:
        terminations = torch.zeros(len(env.states))
        terminations[goal] = 1.0
        sr = make_sr(env, T, terminations)
        search_cost[:, goal] = sr.sum(axis=1)
    return search_cost

def option_learner(*args, **kwargs):
    d = option_learner_grad(*args, **kwargs)
    return d['terminations'], d['res']

def option_learner_grad(
    env,
    *,
    search_cost=None,
    num_options=1,
    lr=1,
    grad_steps=10,

    opt_fn=torch.optim.Adam,
    progress=1,
    plot=True,
    discount=1.0,
    C=1.,
    seed=None,
    term_max=1,
    term_fixed=None,
    full_term_start=None,
    meta_beta=1.,
    term_start_tweaks=None,
    terminal_reward=0.,
    option_planner_bfs=option_planner_bfs_vec,
    terminations=None,
    reset=None,
    # Considers goal to be selected uniformly from env.goal_set
    goal_uniform_random=False,
    show_all_metapolicies=False,
    add_goal_options=False,
    debug=True,
):
    seed = seed or np.random.randint(2**30)
    if debug: print('seed', seed)
    r = np.random.RandomState(seed)
    term_start = r.uniform(0, term_max, size=(num_options, len(env.states)))
    for idx, val in term_start_tweaks or []: term_start[idx] = val
    terminations_opt = (
        torch.tensor(term_start.tolist(), requires_grad=True)
        if terminations is None else  
        terminations.clone().detach().requires_grad_(True))

    make_terminations = lambda env: terminations_opt
    if add_goal_options:
        assert goal_uniform_random
        '''
        goal_options = torch.zeros((len(env.goal_set), len(env.states)))
        for idx, g in enumerate(env.goal_set):
            goal_options[idx, env.states_to_idx[g]] = 100.
        make_terminations = lambda: torch.cat((terminations_opt, goal_options), axis=0)
        '''
        goal_options = torch.zeros((len(env.goal_set), len(env.states)))
        for idx, g in enumerate(env.goal_set):
            goal_options[idx, env.states_to_idx[g]] = 100.
        def make_terminations(env):
            goal_options = torch.zeros((len(env.goal_set), len(env.states)))
            for idx, g in enumerate(env.goal_set):
                goal_options[idx, env.states_to_idx[g]] = 100.
            return torch.cat((terminations_opt, goal_options), axis=0)

    opt = opt_fn([terminations_opt], lr=lr)

    cost = lambda *, env=env: option_planner_bfs(
        env, make_terminations(env), search_cost,
        terminal_reward=terminal_reward,
        discount=discount,
        meta_beta=meta_beta)

    grad_cost = lambda: cost()[0]
    if goal_uniform_random:
        goal_envs = []
        for g in env.goal_set:
            e = copy.copy(env)
            e.goal_set = {g}
            goal_envs.append(e)
        grad_cost = lambda: sum(
            cost(env=goal_env)[0]
            for goal_env in goal_envs)/len(goal_envs)

    for idx in range(grad_steps):
        opt.zero_grad()
        loss = grad_cost()
        loss.backward(retain_graph=True)
        if ((idx+1) % progress) == 0:
            if debug: print(idx, loss.item())
        opt.step()
        if reset is not None and ((idx+1) % reset) == 0:
            if debug: print(idx, 'Reset optimizer state')
            opt.state.clear()

    if goal_uniform_random:
        res = cost(env=goal_envs[0])
    else:
        res = cost()
    res = [x.detach() for x in res]
    if plot:
        J, V, Q, policy, eta = res
        #plot_grid(env, torch.argmax(policy,axis=1))
        #plt.title('Meta-policy')
        #plot_grid(env, (eps+policy[:, 0]).log() - (eps+policy[:, 1]).log(), cmap='RdBu', vcenter=True)

        fmeta, axmeta = plt.subplots(figsize=(10, 5))
        show_policy_or_eta(env, policy, colorbar=False)
        axmeta.set(title='Meta-policy')
        if show_all_metapolicies:
            for goal_env in goal_envs[1:]:
                p = cost(env=goal_env)[3].detach()
                f, ax = plt.subplots(figsize=(10, 5))
                show_policy_or_eta(goal_env, p, colorbar=False)
                ax.set(title='Meta-policy')

        f, ax = plt.subplots(figsize=(10, 5))
        show_policy_or_eta(env, eta.T, colorbar=False)
        ax.set(title=r'$\eta$')

        for o in range(num_options):
            plot_grid(env, eta[o].numpy(), vmin=0)#, vmax=1)
            plt.title(f'O {o} Eta')

    return dict(
        terminations=terminations_opt.detach(),
        res=res,
        seed=seed,
    )

def make_option_terminations(env, option_terms):
    '''
    A simple function for making ~deterministic terminations. Takes an environment and
    a list of option terminations. Each option has one termination point which has high probability.
    '''
    terminations = torch.zeros((len(option_terms), len(env.states)))
    for oidx, ot in enumerate(option_terms):
        terminations[oidx, env.states_to_idx[ot]] = 10.
    return terminations

def env_rect(env):
    lw = 0.5
    return [
        plt.Rectangle((s[1]-0.5, s[0]-0.5), 1, 1, facecolor="none", edgecolor="k", linewidth=lw)
        for si, s in enumerate(env.states_features)
    ]

def plot_graph(
    env, *, eta=None, alphas=None, z=None, labels=False, layout='neato',
    size=None,
    node_arg={}, vmin=None, vmax=None
):
    def alpha_to_hex(alpha):
        return '%02x' % (int(alpha*255))
    if z is not None:
        z = np.array(z)
        vmin = vmin or z.min()
        vmax = vmax or z.max()
        z = np.clip(z, vmin, vmax)
        alphas = (z - vmin) / (vmax - vmin)
    if eta is not None:
        alphas = eta.max(0).values
    g = graphviz.Graph()
    g.attr('graph', layout=layout, size=size and str(size))
    next_states = torch.LongTensor([[env.step(s, a)[0] for a in env.actions] for s in env.states])
    for s in env.states:
        color = '#008800'
        if env.states_features[s] in env.goal_set:
            color = '#880000'
        alpha = 1.0
        if alphas is not None:
            alpha = alphas[s]
        label = None
        if labels is True: # backward compat
            label = '|'.join([''.join(c) for c in env.states_features[s]])
        elif labels:
            label = labels[s]
        elif labels is False:
            label = ''
        default_args = dict(color='black', fillcolor=f'{color}{alpha_to_hex(alpha)}', style='filled',
               fontsize=str(8), width=str(0.25), height=str(0.1))
        if labels is False:
            default_args['shape'] = 'circle'
            default_args['width'] = default_args['height'] = str(0.1 + alpha * 0.2)
        else:
            default_args['shape'] = 'rect'
        if hasattr(env, 'pos'):
            x, y = env.pos[s]
            default_args['pos'] = f'{y},{x}!'
        g.node(str(s), label=label, **dict(default_args, **node_arg))
        '''
        shape = 'point' if labels is False else 'rect'
        g.node(str(s), color=f'{color}{alpha_to_hex(alpha)}', style='filled', label=label,
               fontsize=str(8), width=str(0.25), height=str(0.1), shape=shape)
               #fontsize=str(8), width=str(0.25), height=str(0.1), shape='circle')
        '''
        for ns in next_states[s]:
            if s >= ns:
                continue
            g.edge(str(s), str(ns.numpy()))
    return g


def show_policy_or_eta(
    env,
    policy, # can also pass in eta.T
    *,
    #threshold=1e-7,
    colorbar=True,
    cmaps = [
        new_opacity_cmap(c)
        for c in [
            plt.get_cmap('Greens')(0.7),
            plt.get_cmap('Oranges')(0.7),
            plt.get_cmap('Purples')(0.7),
            plt.get_cmap('Blues')(0.7),
            plt.get_cmap('Greys')(0.7),
            #plt.get_cmap('autumn')(1.0),
            plt.get_cmap('cool')(0.0),
            plt.get_cmap('spring')(0.0),
            plt.get_cmap('Reds')(0.7),
            plt.get_cmap('Paired')(1.0),
        ]
    ],
):
    num_options = policy.shape[1]
    v = np.full((num_options,) + env.npg.shape, np.nan)
    for si, s in enumerate(env.states_features):
        o = np.argmax(policy[si])
        #if policy[si, o] < threshold:
        #    continue
        v[o, s[0], s[1]] = policy[si, o]

    assert len(cmaps)>=num_options

    plt.gcf().set_facecolor('white')
    for o in range(num_options):
        plt.imshow(v[o], cmap=cmaps[o], vmin=0, vmax=1)
        if colorbar:
            plt.colorbar()

    ax = plt.gca()
    for patch in env_rect(env):
        ax.add_patch(patch)

    for si in env.start_states:
        s = env.states_features[si]
        ax.plot(s[1], s[0], marker='o', c='k', fillstyle='none')
    for s in env.goal_set:
        ax.scatter(s[1], s[0], marker='*', c='k')


def option_learner_enum(env, *, search_cost=None, num_options=1, option_sets=None, debug=True, tqdm=lambda x:x, **kwargs):
    option_sets = option_sets or list(itertools.combinations(range(len(env.states)), num_options))

    env = copy.copy(env)
    # HACK dealing with weird initialization of Graph() instances
    if not env.start_states:
        env.start_states = list(env.states)
    goal_set = env.goal_set or set(env.states)

    results = []

    for os in tqdm(option_sets):
        vsum = 0
        for g in goal_set:
            env.goal_set = {g}
            terms = torch.zeros((1+len(os), len(env.states)))
            terms[len(os), env.states_to_idx[g]] = 100.
            for o, oval in enumerate(os):
                terms[o, oval] = 100.
            r = option_planner_bfs_vec(env, terms, search_cost)
            vsum += -r[0].item()
        results.append(dict(options=os, value=vsum/len(goal_set)))

    if debug:
        for item in heapq.nsmallest(kwargs.get('top_results', 3), results, lambda d: -d['value']):
            print(item)

    return results
