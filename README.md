# SARSA-ExpectedSARSA
RL - SARSA and ExpectedSARSA implementation for a race grid world MDP

## Description
Start in `S`, move `V` steps at a time (where `V` is the velocity) in some direction.

If you hit a wall `X`, go back to `S` and `V` is reset to 0
### Grid world - Race track:
```
G | _ | _ | _ | _ | X | X |
G | _ | _ | _ | _ | _ | _ |
X | X | _ | _ | _ | _ | _ |
X | X | X | X | X | _ | _ |
X | X | _ | _ | _ | _ | _ |
_ | _ | _ | _ | _ | _ | _ |
S | _ | _ | _ | X | X | X |
```
### Actions:
- move RIGHT & Velocity -1 -- 0
- move RIGHT & Velocity +0 -- 1
- move RIGHT & Velocity +1 -- 2
- move UP & Velocity -1 -- 3
- move UP & Velocity +0 -- 4
- move UP & Velocity +1 -- 5
- move LEFT & Velocity -1 -- 6
- move LEFT & Velocity +0 -- 7
- move LEFT & Velocity +1 -- 8

### Rewards:
- GOAL = +100
- WALL = -10
- STEP = -1

## Run
`python main.py <algo> <flags>`

Algo:
- "SARSA"
- "ExpectedSARSA"

Flags:
- `-n #` number of episodes to train for (default=100)
- `-g #` gamma, discount factor (default=0.99)
- `-a #` alpha, learning rate (default=0.10)
- `-e #` epsilon, policy stochasticity, proba of chosing a random non-greddy action (default=0.10)
- `-b #` beta, environement stochasticity, proba of not updating `V` no matter the action (default=0.10)

