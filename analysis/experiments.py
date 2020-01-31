from ipypb import track as tqdm
import prior_envs
import copy
import numpy as np
import random
import inspect
import diffplan
import joblib
import os

experiment_envs = dict(
    f2c=prior_envs.f2c,
    f2d=prior_envs.f2d,
    f2f=prior_envs.f2f,
)

def do_experiment(env_name, num_options, samples=100, cost='bfs'):
    cost_name = dict(
        bfs='',
        astar='-astar',
    )[cost]
    dump_name = f'{env_name}-o{num_options}{cost_name}.bin'
    if os.path.exists(dump_name):
        print(dump_name, 'exists')
        return
    print(dump_name)

    seeds = np.random.randint(2**30, size=2)
    np.random.seed(seeds[0])
    random.seed(seeds[1])
    print('Seeds', seeds)

    env = copy.copy(experiment_envs[env_name])

    if cost=='bfs':
        D = diffplan.compute_distance_matrix(env)
        BFS = diffplan.compute_bfs_matrix(env, D)
        c = (D + BFS).float()
    elif cost=='astar':
        h = envs.compute_blocks_distance_heuristic(env)
        a_star_distance, a_star_cost = astar.compute_astar_matrix(env, h, samples=100, tqdm=tqdm)
        c = (a_star_distance + a_star_cost).float()

    env.goal_set = set(env.states_features)
    env.start_states = env.states

    opts = dict(
        reset=400, progress=20, grad_steps=400,
        debug=False,
        lr=0.05, num_options=num_options, term_max=1.,
        goal_uniform_random=True, plot=False, add_goal_options=True,
    )

    results = []
    for _ in tqdm(range(samples)):
        r = diffplan.option_learner_grad(env, search_cost=c, **opts)
        results.append(r)

    joblib.dump(dict(
        options=opts,
        results=results,
        seeds=seeds,
    ), dump_name)

if __name__ == '__main__':
    do_experiment('f2c', 1)
    do_experiment('f2c', 2)
    do_experiment('f2d', 1)
    do_experiment('f2f', 1)
    do_experiment('f2f', 3)
    do_experiment('f2f', 6)
    do_experiment('f2f', 1, cost='astar')
    do_experiment('f2f', 3, cost='astar')
    do_experiment('f2f', 6, cost='astar')
