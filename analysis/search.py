import collections
import random
import heapq

def reconstruct_path(start, end, camefrom):
    p = [end]
    while p[-1] != start:
        p.append(camefrom[p[-1]])
    return p[::-1]

def bfs(env, start, goal_test, *, shuffle_actions=True, visited_init=frozenset()):
    if not callable(goal_test):
        goal = goal_test
        goal_test = lambda state: state == goal
    queue = collections.deque()
    queue.append(start)
    actions = list(range(len(env.actions)))
    visited = set(visited_init)
    camefrom = {}
    while queue:
        current = queue.popleft()
        if goal_test(current):
            p = reconstruct_path(start, current, camefrom)
            return dict(
                visited=frozenset(visited),
                frontier=frozenset(queue),
                path=p,
                cost=len(p)-1,
            )
        visited.add(current)
        if shuffle_actions:
            random.shuffle(actions)
        for aidx in actions:
            a = env.actions[aidx]
            ns, _, _ = env.step(current, a)
            if ns not in visited and ns not in queue:
                camefrom[ns] = current
                queue.append(ns)

def dfs(env, start_state, goal, *, shuffle=True, append_queue_entries=False, return_path=False, visited_init=frozenset()):
    '''
    append_queue_entries: When true, nodes are appended to the queue, even if
    they're in the queue already. This might use more memory, but can ensure
    a promising node encountered early (but not first) isn't ignored.
    '''
    actions = list(range(len(env.actions)))
    visited = set(visited_init)
    queue = [start_state]
    distance = {start_state: 0}
    come_from = {}

    while queue:
        n = queue.pop()
        if n in visited:
            continue
        if n == goal:
            r = dict(
                distance=distance[n],
                plan_cost=len(visited),
            )
            if return_path:
                r['path'] = reconstruct_path(start_state, n, come_from)
                # HACK
                r['visited'] = visited
            return r
        visited.add(n)
        if shuffle:
            random.shuffle(actions)
        for aidx in actions:
            nn, _, _ = env.step(n, env.actions[aidx])
            if (append_queue_entries or nn not in queue) and nn not in visited:
                queue.append(nn)
                distance[nn] = distance[n] + 1
                if return_path:
                    come_from[nn] = n

def deterministic_search(
    env, start, goal_test, *,
    shuffle_actions=True,
    visited_init=frozenset(),
    queue_order=None,
    heuristic=None,
    algorithm=None,
    debug=False,
):
    cost_prioritization = True
    assert algorithm in (None, 'a*', 'dfs', 'bfs', 'ucs')
    if algorithm == 'a*':
        queue_order = queue_order or 'lifo'
        assert heuristic is not None
    elif algorithm == 'ucs':
        # Slight preference for lifo since it should minimize heap usage.
        queue_order = queue_order or 'lifo'
    elif algorithm == 'dfs':
        queue_order = 'lifo'
        cost_prioritization = False
    elif algorithm == 'bfs':
        queue_order = 'fifo'
        cost_prioritization = False

    if cost_prioritization:
        def not_in_queue(next_state, f_next_state):
            return all(f_next_state < f for (f, _, g, state) in queue if state == next_state)
    else:
        def not_in_queue(next_state, f_next_state):
            return not any(state == next_state for (f, _, g, state) in queue)

    if not callable(goal_test):
        goal = goal_test
        goal_test = lambda state: state == goal

    if not cost_prioritization:
        f_fn = lambda state, g: 0
    elif heuristic is None:
        f_fn = lambda state, g: g
    else:
        f_fn = lambda state, g: g + heuristic(state)

    assert queue_order in ('fifo', 'lifo')
    inc = +1 if queue_order == 'fifo' else -1

    queue_ct = 0
    queue = [] # [(f(state), queue #, g(state), state)]
    heapq.heappush(queue, (f_fn(start, 0), queue_ct, 0, start))
    queue_ct += inc

    actions = list(range(len(env.actions)))
    visited = set(visited_init)
    camefrom = {}

    while queue:
        f, _, g, current = heapq.heappop(queue)
        if debug:
            print('Current node', current, 'f', f, 'g', g)
        if current in visited: # Need to do this since we might queue state twice for A* or DFS
            continue
        if goal_test(current):
            if debug:
                for el in queue:
                    print(queue)
            return dict(
                cost=g,
                visited=frozenset(visited),
                frontier=frozenset((el[-1] for el in queue)),
                path=reconstruct_path(start, current, camefrom),
            )

        visited.add(current)
        if shuffle_actions:
            random.shuffle(actions)
        for aidx in actions:
            a = env.actions[aidx]
            ns, rew, _ = env.step(current, a)
            g_ns = g - rew
            f_ns = f_fn(ns, g_ns)

            if debug:
                print('\tSuccessors', ns, 'f', f_ns, 'g', g_ns, 'will add', ns not in visited and not_in_queue(ns, f_ns))
            if ns not in visited and not_in_queue(ns, f_ns):
                camefrom[ns] = current
                heapq.heappush(queue, (f_ns, queue_ct, g_ns, ns))
                queue_ct += inc
