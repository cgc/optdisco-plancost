import collections

def bfs(env, start, goal):
    queue = collections.deque()
    queue.append(start)
    actions = list(range(len(env.actions)))
    visited = set()
    camefrom = {}
    while queue:
        current = queue.popleft()
        if current == goal:
            p = [current]
            while p[-1] != start:
                p.append(camefrom[p[-1]])
            return dict(
                visited=visited,
                frontier=set(queue),
                path=(
                    None,
                    p[::-1]
                ),
            )
        visited.add(current)
        for aidx in actions:
            a = env.actions[aidx]
            ns, _, _ = env.step(current, a)
            if ns not in visited and ns not in queue:
                camefrom[ns] = current
                queue.append(ns)

