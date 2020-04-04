"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np


class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """

    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")


class Mixture_Bandit_NonStat(Bandit):
    """ A Mixture_Bandit_NonStat is a 2-component Gaussian Mixture
    reward distribution (sum of two Gaussians with weights w and 1-w in [O,1]).

    The two means are selected according to N(0,1) as before.
    The two weights of the gaussian mixture are selected uniformely.
    The Gaussian mixture in non-stationary: the means AND WEIGHTS move every
    time-step by an increment epsilon~N(m=0,std=0.01)"""

    def __init__(self):
        Bandit.__init__(self)

        self.mean_1 = np.random.normal(0, 1)
        self.mean_2 = np.random.normal(0, 1)

        self.weight_1 = np.random.uniform(0, 1)
        self.weight_2 = np.random.uniform(0, 1)

    def reset(self):
        self.mean_1 = np.random.normal(0, 1)
        self.mean_2 = np.random.normal(0, 1)

        self.weight_1 = np.random.uniform(0, 1)
        self.weight_2 = np.random.uniform(0, 1)

    def update(self):
        self.mean_1 += np.random.normal(0, 0.01)
        self.mean_2 += np.random.normal(0, 0.01)
        self.weight_1 += np.random.normal(0, 0.01)
        self.weight_2 += np.random.normal(0, 0.01)

    def pull(self) -> float:
        return self.weight_1 * self.mean_1 + self.weight_2 * self.mean_2


class KBandit_NonStat:
    """ Set of K Mixture_Bandit_NonStat Bandits.
    The Bandits are non stationary, i.e. every pull changes all the
    distributions.

    This k-armed Bandit has:
    * an __init__ method to initialize k
    * a reset() method to reset all Bandits
    * a pull(lever) method to pull one of the Bandits; + non stationarity
    """

    def __init__(self, k, **kwargs):
        self.k = k
        self.bandits = [Mixture_Bandit_NonStat() for i in range(k)]
        self.best_action = 0

    def reset(self):
        for bandit in self.bandits:
            bandit.reset()
        self.best_action = 0

    def pull(self, lever) -> float:
        result = self.bandits[lever].pull()

        for bandit in self.bandits:
            bandit.update()

        self.best_action = np.argmax([bandit.pull() for bandit in self.bandits])

        return result
