import envs

c0 = set(range(5))
c1 = set(range(5, 10))
c2 = set(range(10, 15))
graph = [
    (0, c0-{0}),
    (1, c0-{1}),
    (2, c0-{2}),
    (3, c0-{3, 4}),
    (4, c0-{3, 4}),

    (5, c1-{5}),
    (6, c1-{6}),
    (7, c1-{7}),
    (8, c1-{8, 9}),
    (9, c1-{8, 9}),

    (10, c2-{10}),
    (11, c2-{11}),
    (12, c2-{12}),
    (13, c2-{13, 14}),
    (14, c2-{13, 14}),

    # cross-community
    (3, [8]),
    (4, [13]),
    (9, [14]),
]
f2a = envs.Graph(graph)

# fig. 2C

graph = [
    (0, [1, 2, 3]),
    (2, [1, 3]),
    (4, [1, 3, 5]),
    (5, [6, 8]),
    (7, [6, 8]),
    (9, [6, 7, 8]),
]

f2c = envs.Graph(graph)

# must be x (horiz), y (vert)
f2c.pos = [
    (0, 5),
    (2, 5),
    (1, 4),
    (0, 3),
    (2, 3),
    (3, 2),
    (5, 2),
    (4, 1),
    (3, 0),
    (5, 0),
]

# fig. 2D

def grid(idx=0, w=3, h=3):
    xy_to_node = lambda x,y: idx + x+y*w
    nodes = []
    for x in range(w):
        for y in range(h):
            neighbors = []
            if 0 <= x-1: neighbors.append(xy_to_node(x-1,y))
            if x+1 < w: neighbors.append(xy_to_node(x+1,y))
            if 0 <= y-1: neighbors.append(xy_to_node(x,y-1))
            if y+1 < h: neighbors.append(xy_to_node(x,y+1))
            nodes.append((xy_to_node(x,y), neighbors))
    return nodes

graph = grid(0) + grid(10) + [
    (9, [2, 8, 10, 16])
]
f2d = envs.Graph(graph)

# fig. 2F

f2f = envs.Blocks(3, hanoi=True, canonicalize=False)
f2f.goal_set = set(f2f.states_features)

# hack towards balageur

bala2016_hack = envs.Graph([
    (1, [0, 2]),
    (3, [2, 4]),
    (5, [4, 6]),
    (7, [6, 8]),

    (10, [9, 11]),
    (12, [11, 3]),
    (14, [3, 13]),
])