import search
import astar
import prior_envs
import random

class TestEnv(object):
    def __init__(self, shortcut_cost=-2):
        self.states = [0, 1, 2, 3]
        self.actions = [-1, 0, 1]
        self.shortcut_cost = shortcut_cost
    def step(self, s, a):
        if a == +1 and s != 3:
            return s+a, -1, None
        if a == -1 and s != 0:
            return s+a, -1, None
        if a == 0 and s in (0, 3):
            ns = 0 if s == 3 else 3
            return ns, self.shortcut_cost, None
        return s, -1, None


def test_deterministic_search_weighted():
    start, goal = 0, 3
    kw = dict(shuffle_actions=False)

    env = TestEnv(-5)

    res = search.deterministic_search(env, start, goal, algorithm='ucs', **kw)
    print(res)
    assert res['visited'] == set([0, 1, 2])
    assert res['path'] == [0, 1, 2, 3]

    res = search.deterministic_search(env, start, goal, algorithm='bfs', **kw)
    print(res)
    assert res['visited'] == set([0])
    assert res['path'] == [0, 3]


def test_deterministic_search_f2d():
    start, goal = 0, 7
    kw = dict(shuffle_actions=False)

    env = prior_envs.f2d
    h = astar.make_env_position_heuristic(env, goal)

    res = search.deterministic_search(env, start, goal, algorithm='a*', heuristic=h, **kw)
    print(res)
    assert res['visited'] == set([0, 3, 6])
    assert len(res['path']) == 4

    # FIFO means we look at everything en route
    res = search.deterministic_search(env, start, goal, queue_order='fifo', algorithm='a*', heuristic=h, **kw)
    print(res)
    assert res['visited'] == set([0, 1, 3, 4, 6])
    assert len(res['path']) == 4

    res = search.deterministic_search(env, start, goal, algorithm='bfs', **kw)
    print(res)
    assert res['visited'] == set([0, 1, 2, 3, 4, 5, 6, 9])
    assert len(res['path']) == 4

    res = search.deterministic_search(env, start, goal, algorithm='dfs', **kw)
    print(res)
    assert len(res['path']) == 4

    # DFS can be suboptimal
    random.seed(2)
    res = search.deterministic_search(env, start, goal, algorithm='dfs')
    print(res)
    assert len(res['path']) == 6
