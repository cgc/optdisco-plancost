import diffplan
import torch

def test_it():
    env = diffplan.Grid('S G')
    terms = torch.tensor([
    #    [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ]).float()
    sc = torch.tensor([
        [0, 1, 3],
        [1, 0, 1],
        [3, 1, 0],
    ]).float() + torch.eye(3)
    J, V, Q, policy, eta = diffplan.option_planner_hardmax(env, terms, sc)
    assert policy[0].argmax() == 0 and J==2
    sc[1, 2] = 5
    J, V, Q, policy, eta = diffplan.option_planner_hardmax(env, terms, sc)
    assert policy[0].argmax() == 1 and J==3
