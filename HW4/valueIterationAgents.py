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
              mdp.getStates() Ex: ['TERMINAL_STATE', (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
              mdp.getPossibleActions(state) Ex: ('north', 'west', 'south', 'east')
              mdp.getTransitionStatesAndProbs(state, action) Given the state and action, get the probabilities of next possible states, Ex: [((1, 2), 0.2), ((2, 2), 0.8)]
              mdp.getReward(state, action, nextState) Get the reward of the state
              mdp.isTerminal(state) Check if the state is terminal
        """
        self.mdp = mdp
        self.discount = discount # discount rate
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        tmp = util.Counter()
        for i in range(iterations+1):
            self.values = tmp.copy()
            for state in self.mdp.getStates():
                tmp[state] = self.mdp.getReward(state, None, None) + (self.discount * self.getQValue(state, self.getAction(state)))

    def getValue(self, state): #utility
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if action == None:
            return 0
        val = 0
        for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            val += prob * self.values[nextState]
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None
        actions = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            actions[action] = self.computeQValueFromValues(state, action)
        return actions.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
