#!/usr/bin/env python

import numpy as np
import argparse
from datetime import datetime as dt

WORLD = np.array([
    ["G", "_", "_", "_", "_", "X", "X"],
    ["G", "_", "_", "_", "_", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["X", "X", "X", "X", "X", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_"],
    ["S", "_", "_", "_", "X", "X", "X"]
])
STATES = range(WORLD.size)  # 1D array from 0 to 28
STATE2WORLD = {
    0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(0,4), 5:(0,5), 6:(0,6),
    7:(1,0), 8:(1,1), 9:(1,2), 10:(1,3), 11:(1,4), 12:(1,5), 13:(1,6),
    14:(2,0), 15:(2,1), 16:(2,2), 17:(2,3), 18:(2,4), 19:(2,5), 20:(2,6),
    21:(3,0), 22:(3,1), 23:(3,2), 24:(3,3), 25:(3,4), 26:(3,5), 27:(3,6),
    28:(4,0), 29:(4,1), 30:(4,2), 31:(4,3), 32:(4,4), 33:(4,5), 34:(4,6),
    35:(5,0), 36:(5,1), 37:(5,2), 38:(5,3), 39:(5,4), 40:(5,5), 41:(5,6),
    42:(6,0), 43:(6,1), 44:(6,2), 45:(6,3), 46:(6,4), 47:(6,5), 48:(6,6)
}
START = 42  # state index of start
GOALS = [0, 7]  # state index of goals
WALLS = [  # state index of walls
    5, 6,
    14, 15,
    21, 22, 23, 24, 25,
    28, 29,
    46, 47, 48
]
V = 0  # velocity

ACTIONS = [  # set of all actions
    0, 1, 2,  # move RIGHT: V-1, V+0, V+1
    3, 4, 5,  # move UP:    V-1, V+0, V+1
    6, 7, 8  # move LEFT:   V-1, V+0, V+1
]
CRASH = -10.  # reward for hitting a wall
WIN = 100.  # reward for reaching a goal
STEP = -1.  # reward for moving

PI = np.zeros((len(STATES), len(ACTIONS)))  # policy: <state, action> -> <float>
Q = np.zeros((len(STATES), len(ACTIONS)))  # <state, action> -> <float>


def reset():
    """
    reset grid world and velocities
    """
    global WORLD
    WORLD = np.array([
        ["G", "_", "_", "_", "_", "X", "X"],
        ["G", "_", "_", "_", "_", "_", "_"],
        ["X", "X", "_", "_", "_", "_", "_"],
        ["X", "X", "X", "X", "X", "_", "_"],
        ["X", "X", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_"],
        ["S", "_", "_", "_", "X", "X", "X"]
    ])
    global V
    V = 0


def choose_action(s, epsilon):
    """
    Choose an action from state s according to epsilon-greedy policy
    :param s: current state
    :param epsilon: proba of choosing a non-optimal action
    :return: action to take from s
    """
    # Update PI(s, a) for all actions a for that state s:
    # action probabilities = epsilon/(|A|-1) for all actions by default
    # over |A|-1 because 1 of them will be optimal and have proba 1-epsilon
    global PI
    PI[s, :] = [epsilon / (len(ACTIONS)-1.)] * len(ACTIONS)

    # Get the best action for that state (greedy w.r.t. Q):
    best_a = 0
    best_q_val = -np.inf
    for i, q_val in enumerate(Q[s,:]):
        if q_val > best_q_val:
            best_q_val = q_val
            best_a = i

    # Change default proba of best action to be 1-epsilon
    PI[s, best_a] = 1. - epsilon
    print "best action:", best_a
    assert np.sum(PI[s, :]) == 1.

    # sample from ACTIONS with proba distribution PI[s, :]
    return np.random.choice(ACTIONS, p=PI[s, :])


def move(s, a, beta):
    """
    Perform action a in state s, and observe r in s'
    :param s: current state
    :param a: action to take from state s
    :param beta: proba of no velocity update
    :return: next state and observed reward
    """
    # update velocity with probability 1-beta
    global V
    if np.random.choice(2, p=[beta, 1-beta]) == 1:
        if a in [0, 3, 6] and V > 0: V -= 1
        elif a in [2, 5, 8] and V < 3: V += 1
    else:
        print "velocity not updated!"

    r_border = range(6, 49, 7)  # states on the right border
    l_border = range(0, 49, 7)  # states on the left border
    t_border = range(7)  # states on the top border

    units = range(V)
    # move RIGHT of V units:
    if a < len(ACTIONS) / 3:
        for i in units:
            WORLD[STATE2WORLD[s+i]] = '~'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s+i in r_border or s+i+1 in WALLS:
                reset()
                return START, CRASH
        # nothing special: draw where I end up & return
        WORLD[STATE2WORLD[s+V]] = 'O'
        return s+V, STEP

    # move UP of V units:
    elif a < 2*len(ACTIONS) / 3:
        for i in units:
            WORLD[STATE2WORLD[s-i*7]] = '|'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s-i*7 in t_border or s-(i+1)*7 in WALLS:
                reset()
                return START, CRASH
        # nothing special: draw where I end up & return
        WORLD[STATE2WORLD[s-V*7]] = 'O'
        return s-V*7, STEP

    # move LEFT of V units:
    elif a < len(ACTIONS):
        for i in units:
            WORLD[STATE2WORLD[s-i]] = '~'  # draw my path gradualy in the world
            # goal: draw where I end up & return
            if s-i-1 in GOALS:
                WORLD[STATE2WORLD[s-i-1]] = 'O'
                return s-i-1, WIN
            # crash: reset world and velocities, return to start state
            elif s-i in l_border or s-i-1 in WALLS:
                reset()
                return START, CRASH
        # nothing special: draw where I end up & return
        WORLD[STATE2WORLD[s-V]] = 'O'
        return s-V, STEP

    return s, STEP  # should never happen


def main():
    def my_float(x):  # Custom type for argparse arguments: gamma, alpha, epsilon, beta
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % x)
        return x

    parser = argparse.ArgumentParser(description='MDP SARSA vs Expected SARSA.')
    parser.add_argument(
        'method',
        choices=["SARSA", "ExpectedSARSA"],
        help="The algorithm to use for solving a simple grid world MDP."
    )
    parser.add_argument(
        '-n', '--n_episodes', type=int, default=100,
        help="number of episodes to train the agent for."
    )
    parser.add_argument(
        '-g', '--gamma', type=my_float, default=0.99,
        help="discount factor."
    )
    parser.add_argument(
        '-a', '--alpha', type=my_float, default=0.1,
        help="learning rate."
    )
    parser.add_argument(
        '-e', '--epsilon', type=my_float, default=0.1,
        help="epsilon in epsilon-greedy policy, ie: Stochasticity of policy."
    )
    parser.add_argument(
        '-b', '--beta', type=my_float, default=0.1,
        help="probapility of no velocity update. ie: Stochasticity of environment."
    )
    args = parser.parse_args()
    print args

    n_steps = []  # number of steps for each episode
    rewards = []  # total reward for each episode

    start = dt.now()
    ep = 0
    while ep < args.n_episodes:
        print "\nEpisode", ep+1, "/", args.n_episodes, "..."
        reset()  # reset grid world and velocities before the start of each episode.
        steps = 0  # keep track of the number of steps to finish an episode
        reward = 0  # keep track of the total reward for that episode
        s = START
        a = choose_action(s, args.epsilon)
        while s not in GOALS:
            print WORLD
            print "state:", s, "V:", V, "action:", a
            s_next, r = move(s, a, args.beta)
            steps += 1
            reward += r
            print "next state:", s_next, "reward:", r
            if args.method == "SARSA":  # need to pick next action BEFORE the update!
                a_next = choose_action(s_next, args.epsilon)
                Q[s, a] = Q[s, a] + args.alpha * (r + args.gamma * Q[s_next, a_next] - Q[s, a])
            else:  # no need to pick the next action, we take the expectation of Q!
                Q[s, a] = Q[s, a] + args.alpha * (r + args.gamma * np.sum(PI[s, :]*Q[s, :]) - Q[s, a])
                a_next = choose_action(s_next, args.epsilon)
            s = s_next
            a = a_next
        ep += 1
        n_steps.append(steps)
        rewards.append(reward)
    print WORLD

    print "took:", (dt.now() - start).total_seconds(), "seconds."
    print "number of steps for each episode:", n_steps
    print "average number of steps:", np.average(n_steps)
    print "reward of each episode:", rewards
    print "average return:", np.average(rewards)


if __name__ == '__main__':
    main()
