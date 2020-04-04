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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
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

        delta = 0.01
        final_iteration = iterations
        for iteration in range(iterations):
            difference = 0
            for state in self.mdp.getStates():
                old_value = self.values[state]
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0:
                    continue

                q_per_action = np.zeros(len(actions), dtype=np.float)
                for i in range(len(actions)):
                    q_per_action[i] = self.computeQValueFromValues(state, actions[i])
                best_value = q_per_action[np.argmax(q_per_action)]
                if best_value is not None:
                    self.values[state] = best_value
                else:
                    continue

                difference = max(difference, abs(old_value - self.values[state]))

            print(self.values)

            if difference < delta:
                final_iteration = iteration + 1
                break

        print("It took a total of {} iterations to converge.".format(final_iteration))
        # TODO: Implement Value Iteration.
        # Exit either when the number of iterations is reached,
        # OR until convergence (L2 distance < delta).
        # Print the number of iterations to convergence.

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
        for i in range(len(states_and_probs)):
            new_state, probability = states_and_probs[i]
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
