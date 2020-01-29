import numpy as np
import itertools
import torch

class Graph(object):
    def __init__(self, graph):
        all_nodes = set()
        for node, edges in graph:
            all_nodes.add(node)
            for e in edges:
                all_nodes.add(e)
        node_count = len(all_nodes)
        nodes = range(node_count)
        assert all_nodes == set(nodes), 'Missing node #s'
        T = np.zeros((node_count, node_count), dtype=np.bool)
        for node, edges in graph:
            for e in edges:
                T[node, e] = 1
                T[e, node] = 1
        # HACK
        num_actions = T.sum(1).max()
        self.T = np.zeros((node_count, num_actions), dtype=np.int)
        for n in nodes:
            # by default self-loop
            self.T[n] = n
            for aidx, dest in enumerate(np.where(T[n])[0]):
                self.T[n, aidx] = dest
        # HACK
        self.actions = range(num_actions)
        self.states = nodes
        self.states_features = nodes # HACK
        self.states_to_idx = nodes # HACK
        self.start_states = []
        self.goal_set = {}

    def step(self, s, a):
        return self.T[s, a], None, None

def canon_state(s):
    return tuple(sorted(s, key=lambda s: (len(s), s[:1]), reverse=True))

def is_canon(s):
    '''
    >>> assert not is_canon((('D',), ('F',), ('A', 'B', 'C')))
    >>> assert is_canon(canon_state((('D',), ('F',), ('A', 'B', 'C'))))
    '''
    return (
        (len(s[0]) != len(s[1]) or s[0][:1] >= s[1][:1]) and
        (len(s[1]) != len(s[2]) or s[1][:1] >= s[2][:1])
    )

def is_valid_hanoi(s):
    '''
    >>> assert is_valid_hanoi(['CBA', '', 'FED'])
    >>> assert not is_valid_hanoi(['CBA', '', 'FDE'])
    '''
    return all(
        c[idx-1] > c[idx]
        for c in s
        for idx in range(1, len(c))
    )

def is_valid_height_limit(s, col_heights):
    '''
    >>> assert is_valid_height_limit(('ABC',), (3,))
    >>> assert not is_valid_height_limit(('ABCD',), (3,))
    '''
    return all(
        len(c) <= ch
        for c, ch in zip(s, col_heights)
    )

def block_states(N=3, canonicalize=True, hanoi=False, height_limits=None):
    '''
    >>> assert len(block_states(N=3, canonicalize=False)) == 60
    >>> assert len(block_states(N=3, canonicalize=True)) == 13
    '''
    letters = [chr(ord('A')+idx) for idx in range(N)]
    block_perm = list(itertools.permutations(letters))

    states = []

    for col1 in range(N+1):
        for col2 in range(col1, N+1):
            sizes = [col1 - 0, col2 - col1, N - col2]
            # This is a filter for canonical states that's fast and easy.
            # HACK though might consider modifying the loops to make this true
            if canonicalize:
                if not(sizes[0] >= sizes[1] >= sizes[2]):
                    continue
            for blocks in block_perm:
                cols = tuple([tuple(blocks[:col1]), tuple(blocks[col1:col2]), tuple(blocks[col2:])])
                if canonicalize and not is_canon(cols):
                    continue
                if hanoi and not is_valid_hanoi(cols):
                    continue
                if height_limits and not is_valid_height_limit(cols, height_limits):
                    continue
                states.append(cols)
    return states

class Blocks(object):
    '''
    >>> b = Blocks(3, canonicalize=True)
    >>> b.states_features[b.step(b.states_to_idx[(('C', 'B'), ('A',), ())], (0, 1))[0]]
    (('A', 'B'), ('C',), ())
    >>> b = Blocks(3, canonicalize=False)
    >>> b.states_features[b.step(b.states_to_idx[(('C', 'B'), ('A',), ())], (0, 1))[0]]
    (('C',), ('A', 'B'), ())
    >>> b = Blocks(3, canonicalize=False, hanoi=True)
    >>> b.states_features[b.step(b.states_to_idx[(('C', 'B'), ('A',), ())], (0, 1))[0]]
    (('C', 'B'), ('A',), ())
    '''
    def __init__(self, num_blocks, *, canonicalize=True, hanoi=False, height_limits=None):
        # HACK do not change this variable
        num_columns = 3
        ####
        self.num_blocks = num_blocks
        self.canonicalize = canonicalize
        self.hanoi = hanoi
        self.height_limits = height_limits
        self.states_features = block_states(N=num_blocks, canonicalize=canonicalize, hanoi=hanoi, height_limits=height_limits)
        self.states = range(len(self.states_features))
        self.start_states = self.states # HACK
        self.states_to_idx = {s: si for si, s in enumerate(self.states_features)}
        goal_state = {(tuple(chr(ord('A')+idx) for idx in range(num_blocks)[::-1]), (), ())}
        self.goal_set = set(goal_state)
        self.actions = [
            (srcidx, destidx)
            for srcidx in range(num_columns)
            for destidx in range(num_columns)
            if srcidx != destidx
        ]
        self.step_reward = -1
    def step(self, si, a):
        '''
        Return: next state, reward, termination probability.
        '''
        s = self.states_features[si]
        srcidx, destidx = a

        if not s[srcidx]:
            return si, self.step_reward, 0.0

        s_next = tuple(
            col[:-1] if colidx == srcidx else
            col+tuple(s[srcidx][-1]) if colidx == destidx else
            col
            for colidx, col in enumerate(s)
        )
        if self.hanoi and not is_valid_hanoi(s_next):
            return si, self.step_reward, 0.0
        if self.height_limits and not is_valid_height_limit(s_next, self.height_limits):
            return si, self.step_reward, 0.0
        if self.canonicalize:
            s_next = canon_state(s_next)
        s_nexti = self.states_to_idx[s_next]
        if s_next in self.goal_set:
            return s_nexti, self.step_reward, 1.0
        else:
            return s_nexti, self.step_reward, 0.0
        # HACK goal??

def blocks_state_match_count(a, b):
    '''
    >>> assert blocks_state_match_count((('A', 'B', 'C'),()), (('A', 'B', 'C'),())) == 3
    >>> assert blocks_state_match_count((('A', 'B'), (), ('C',)), (('A',), ('C', 'B'), ())) == 1
    >>> assert blocks_state_match_count((('A', 'B', 'C'),()), (('C', 'B', 'A'),())) == 0
    '''
    same = 0
    for ca, cb in zip(a, b):
        for ita, itb in zip(ca, cb):
            if ita != itb:
                break
            same += 1
    return same

def compute_blocks_distance_heuristic(env, tqdm=lambda x: x):
    # HACK consider height limits?
    assert not env.canonicalize
    d = torch.zeros((len(env.states), len(env.states)))
    for s in tqdm(env.states):
        for g in env.states:
            d[s, g] = env.num_blocks - blocks_state_match_count(env.states_features[s], env.states_features[g])
    return d
