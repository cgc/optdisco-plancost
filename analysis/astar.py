from collections import deque, defaultdict
import itertools
import heapq
import numpy as np
import random
import torch

# From https://en.wikipedia.org/wiki/A*_search_algorithm
def a_star_cost(
    env,
    start,
    goal,
    heuristic_cost_estimate,
    *,
    dist_between=1,
    #start=None,
    #goal_test=None,
    shuffle_actions=True,
    return_path=False,
    next_state=None,
    #return_all_equal_cost_paths=False,
    #depth_limit=None,
    queue='lifo',
    seed=None,
):
    r = random.Random(seed)

    if next_state is None:
        next_state = np.array([
            [env.step(s, a)[0] for a in env.actions]
            for s in env.states
        ])
    actions = np.arange(len(env.actions))

    # The set of nodes already evaluated
    closedSet = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    openSet = {start}

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    if return_path:
        cameFrom = {}

    # For each node, the cost of getting from the start node to that node.
    gScore = defaultdict(lambda: float('inf'))  # map with default value of Infinity

    # The cost of going from start to start is zero.
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = defaultdict(lambda: float('inf'))  # map with default value of Infinity

    heap_entry = 0
    prioritized_nodes = []  # This is a heap

    if queue=='lifo':
        inc_val = -1
    else:
        assert queue == 'fifo'
        inc_val = +1

    def set_fscore(node, f):
        nonlocal heap_entry
        fScore[node] = f
        # HACK
        # by adding +heap_entry, we ensure FIFO
        # for LIFO we can add -heap_entry
        # for other kinds of orderings, we can add random.random()
        # TODO figure out why LIFO seems so important for some incidental efficiencies.
        heapq.heappush(prioritized_nodes, (f, heap_entry, node))
        heap_entry += inc_val

    # For the first node, that value is completely heuristic.
    set_fscore(start, heuristic_cost_estimate[start, goal])

    #if return_all_equal_cost_paths:
    #    solutions = []

    nodes_visited = 0

    # bookkeeping...
    currf = -1
    count = 0

    res = None

    while openSet:
        f, _, current = heapq.heappop(prioritized_nodes)
        if current in closedSet:
            # HACK this can happen when we have set the f-score for a state
            # twice (thus adding it to queue) before we process it. Since we
            # always process things in order of f-score, we can assume a
            # second visit to this node can be ignored as it's about a path
            # that is less optimal.
            continue

        if res is not None and res['final_score'] != fScore[current]:
            break

        # Ok we start with some bookkeeping... Keeping track of things with same f score as current thing
        if currf == f:
            count += 1
        else:
            # reset
            currf = f
            count = 1
        nodes_visited += 1

        if current == goal:
            '''
            if return_all_equal_cost_paths:
                # If we have other solutions, we make sure this one has equal cost.
                if solutions and gScore[solutions[0]] != gScore[current]:
                    # Otherwise, we simply stop looking at solutions. below we return the solutions we found.
                    break
                solutions.append(current)
            else:
            '''
            '''
            # HACK we are going to pop things until we find everything with this cost...
            fmatch_count_remaining = 0
            while prioritized_nodes:
                other_f, _, other = heapq.heappop(prioritized_nodes)
                if other_f != currf:
                    break
                fmatch_count_remaining += 1
            return dict(
                final_score=f,
                fmatch_count_prior=count,
                fmatch_count_remaining=fmatch_count_remaining,
                nodes_visited=nodes_visited,
                path=reconstruct_path(cameFrom, current) if return_path else None,
            )
            '''
            assert res is None, 'HACK assume 1 goal for now?'
            res = dict(
                final_score=f,
                fmatch_count_prior=count,
                nodes_visited=nodes_visited,
                path=reconstruct_path(cameFrom, current) if return_path else None,
            )

        openSet.remove(current)
        closedSet.add(current)

        if shuffle_actions:
            #actions = np.random.permutation(len(env.actions))
            r.shuffle(actions)
        for aidx in actions:
            a = env.actions[aidx]
            #neighbor, _, _ = env.step(current, a)
            neighbor = next_state[current, aidx]
            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            # The distance from start to a neighbor
            tentative_gScore = gScore[current] + dist_between#(current, neighbor)

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor)
            elif tentative_gScore >= gScore[neighbor]:
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            if return_path:
                cameFrom[neighbor] = (a, current)
            gScore[neighbor] = tentative_gScore
            set_fscore(neighbor, gScore[neighbor] + heuristic_cost_estimate[neighbor, goal])

    '''
    if return_all_equal_cost_paths and solutions:
        return [reconstruct_path(cameFrom, s) for s in solutions]
    '''
    res['fmatch_count_remaining'] = count - res['fmatch_count_prior']
    res['fmatch_count_total'] = count
    res['nodes_visited_final'] = nodes_visited
    return res


def reconstruct_path(cameFrom, current):
    actions = []
    states = [current]
    while current in cameFrom.keys():
        action, current = cameFrom[current]
        actions.append(action)
        states.append(current)
    states.reverse()
    actions.reverse()
    return actions, states


# From https://en.wikipedia.org/wiki/A*_search_algorithm
def A_Star(
    problem,
    heuristic_cost_estimate,
    start=None,
    goal_test=None,
    dist_between=None,
    shuffle=True,
    return_all_equal_cost_paths=False,
    depth_limit=None,
):
    if start is None:
        start = problem.initial
    if goal_test is None:
        goal_test = problem.goal_test
    if dist_between is None:
        dist_between = lambda current, neighbor: 1
    else:
        assert depth_limit is None, (
            'Error: Current depth-limited search is implemented by assuming cost function is 1 so as to use g(state).')

    # The set of nodes already evaluated
    closedSet = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    openSet = {start}

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = {}

    # For each node, the cost of getting from the start node to that node.
    gScore = defaultdict(lambda: float('inf'))  # map with default value of Infinity

    # The cost of going from start to start is zero.
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = defaultdict(lambda: float('inf'))  # map with default value of Infinity

    heap_entry = 0
    prioritized_nodes = []  # This is a heap

    def set_fscore(node, f):
        nonlocal heap_entry
        fScore[node] = f
        # HACK
        # by adding +heap_entry, we ensure FIFO
        # for LIFO we can add -heap_entry
        # for other kinds of orderings, we can add random.random()
        # TODO figure out why LIFO seems so important for some incidental efficiencies.
        heapq.heappush(prioritized_nodes, (f, -heap_entry, node))
        heap_entry += 1

    # For the first node, that value is completely heuristic.
    set_fscore(start, heuristic_cost_estimate(problem, start))

    if return_all_equal_cost_paths:
        solutions = []

    while openSet:
        f, _, current = heapq.heappop(prioritized_nodes)
        if current in closedSet:
            # HACK this can happen when we have set the f-score for a state
            # twice (thus adding it to queue) before we process it. Since we
            # always process things in order of f-score, we can assume a
            # second visit to this node can be ignored as it's about a path
            # that is less optimal.
            continue
        if goal_test(current):
            if return_all_equal_cost_paths:
                # If we have other solutions, we make sure this one has equal cost.
                if solutions and gScore[solutions[0]] != gScore[current]:
                    # Otherwise, we simply stop looking at solutions. below we return the solutions we found.
                    break
                solutions.append(current)
            else:
                return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        closedSet.add(current)

        # Depth limit is the number of nodes deep we will search. We skip nodes any deeper than limit.
        # So for limit of 3, we will visit nodes 3 actions in from the root.
        if depth_limit is not None and gScore[current] >= depth_limit:
            assert gScore[current] == depth_limit, 'A* depth-limited should not descend past depth limit.'
            if return_all_equal_cost_paths:
                # Then we start gathering with logic akin to the above for equal cost paths.
                # If we have other solutions, we make sure this one has equal fScore.
                # We compare fScore since that's how we prioritize nodes and since our
                # only sensible way to assess distance to the goal involves our heuristic
                # of the cost.
                # HACK determine if we simply prefer to compare the heuristic cost instead of fScore.
                # HACK gScore above is really equal to fScore as h() is 0. so this is possibly equivalent logic??
                if solutions and fScore[solutions[0]] != fScore[current]:
                    # Otherwise, we simply stop looking at solutions. below we return the solutions we found.
                    break
                solutions.append(current)
                continue  # We want to ensure we avoid searching any deeper.
            else:
                return reconstruct_path(cameFrom, current)

        actions = problem.actions(current)
        if shuffle:
            random.shuffle(actions)
        for a in actions:
            neighbor = problem.result(current, a)
            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            # The distance from start to a neighbor
            tentative_gScore = gScore[current] + dist_between(current, neighbor)

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor)
            elif tentative_gScore >= gScore[neighbor]:
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = (a, current)
            gScore[neighbor] = tentative_gScore
            set_fscore(neighbor, gScore[neighbor] + heuristic_cost_estimate(problem, neighbor))

    if return_all_equal_cost_paths and solutions:
        return [reconstruct_path(cameFrom, s) for s in solutions]


def compute_manhattan_heuristic(env):
    h = np.full((len(env.states), len(env.states)), np.nan)
    for si, s in enumerate(env.states_features):
        for gi, g in enumerate(env.states_features):
            if si > gi:
                continue
            # manhattan? or euclidean?
            #c = np.sqrt((s[0]-g[0])**2 + (s[1]-g[1])**2)
            c = abs(s[0]-g[0]) + abs(s[1]-g[1])
            h[si, gi] = c
            h[gi, si] = c
    return h



def compute_astar_matrix(env, heuristic_cost_estimate, *, samples=100, tqdm=lambda x: x):
    next_state = np.array([
        [env.step(s, a)[0] for a in env.actions]
        for s in env.states
    ])

    cost = torch.zeros((len(env.states), len(env.states)))
    distance = torch.zeros((len(env.states), len(env.states)))
    for s in tqdm(env.states):
        for g in env.states:
            ds = [
                a_star_cost(env, s, g, heuristic_cost_estimate, next_state=next_state)
                for _ in range(samples)
            ]
            cost[s, g] = 1.*sum(d['nodes_visited'] for d in ds)/samples
            distance[s, g] = ds[0]['final_score']
    return distance, cost


if __name__ == '__main__':
    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
