import gym
from gym import spaces
import numpy as np


class RiverCrossingEnv(gym.Env):
    """ Small 3x5 Gridworld with a river in the middle row."""

    def __init__(self):
        self.height = 3
        self.width = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width)
        ))
        self.moves = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
        }
        self.current_state = (2, 0)

    def reset(self):
        """
        Resets the environment to the first state of the environment.
        Returns
        -------
        state : State
            First state of the episode.
        """
        # TODO: Implement this
        self.current_state = (2, 0)
        return (2, 0)

    def step(self, action):
        """
        Performs a single step of the environment given an action.
        i.e. samples s',r from the p(s',r | s,a) distribution.
        Parameters
        ----------
        action: int
            The action performed by the agent
        Returns
        -------
        next_state : state
            State resulting from the transition
        reward : float
            Reward for this transition
        done : bool
            Whether next_state is a terminal state.
        info : dict
            ignore this
        """
        y = max(0, min(2, self.current_state[0] + self.moves[action][0]))
        x = max(0, min(4, self.current_state[1] + self.moves[action][1]))

        if y == 1:
            sidestep_chance = np.random.uniform(0, 1)
            if sidestep_chance < 0.2:
                x = min(4, x + 1)
            elif sidestep_chance < 0.7:
                x = min(4, x + 2)
            else:
                x = min(4, x + 3)

        next_state = (y, x)

        reward = 0
        if next_state == (0, 4):
            reward = 1
        elif next_state == (1, 4):
            reward = -1

        done = False
        if next_state == (0, 4) or next_state == (1, 4):
            done = True

        info = dict()

        self.current_state = next_state

        return next_state, reward, done, info

    def render(self):
        """
        Optional.
        Prints the environment for visualization.
        """
        print("="*25)
        for y in range(3):
            row = ""
            for x in range(5):
                if self.current_state != (y, x):
                    row += "|   |"
                else:
                   row += "| x |"
            print(row)
            print("="*25)
        pass
