# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np
from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        print("using discount {}".format(discount))
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.policies = util.Counter()

        delta = 0.01
        iteration = 1
        # initialize the policies arbitrarily
        for state in mdp.getStates():
            actions = mdp.getPossibleActions(state)
            if len(actions) >= 1:
                self.policies[state] = mdp.getPossibleActions(state)[0]
            else:
                self.policies[state] = None

        policy_loop = 0
        while True:
            policy_loop += 1
            while True:
                # policy evaluation
                iteration += 1
                difference = 0
                for state in mdp.getStates():
                    old_value = self.values[state]
                    action = self.policies[state]
                    if action is None:
                        continue
                    self.values[state] = self.computeQValueFromValues(state, action)
                    difference = max(difference, abs(old_value-self.values[state]))
                if difference < delta or iteration == iterations:
                    break

            if iteration == iterations:
                break

            # policy imporvement
            stable = True
            iteration += 1
            for state in mdp.getStates():
                old_policy = self.policies[state]
                self.policies[state] = self.computeActionFromValues(state)
                if old_policy != self.policies[state]:
                    stable = False
            if stable or iteration == iterations:
                break

        print("It took a total of {} iterations to converge.".format(iteration))
        print("It took a total of {} total policy iteration loops to converge.".format(policy_loop))


        # TODO: Implement Policy Iteration.
        # Exit either when the number of iterations is reached,
        # OR until convergence (L2 distance < delta).
        # Print the number of iterations to convergence.
        # To make the comparison FAIR, one iteration is a single sweep over states.
        # Compute the number of steps until policy convergence, but do not stop
        # the algorithm until values converge.

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # TODO: Implement this function according to the doc

        states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for new_state, probability in states_and_probs:
            reward = self.mdp.getReward(state, action, new_state)
            sum += probability * (reward + self.discount * self.values[new_state])

        return sum


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # TODO: Implement according to the doc
        actions = self.mdp.getPossibleActions(state)

        if len(actions) == 0:
            return None

        sums_per_action = np.zeros(len(actions), dtype=np.float)
        for i in range(len(actions)):
            sum_for_action = self.computeQValueFromValues(state, actions[i])
            if sum_for_action is not None:
                sums_per_action[i] = sum_for_action
        return actions[np.argmax(sums_per_action)]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
