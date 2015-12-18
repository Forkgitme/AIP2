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
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here

        "*** YOUR CODE HERE ***"
        """reken voor elke iteratie voor elke state de values uit"""
        while self.iterations > 0:
          """use util.Counter for easyness, om keys met values bij te houden (soort dictionary) zie util"""
          v = util.Counter()
          """als er geen legal actions zijn (terminal state) continue"""
          for st in mdp.getStates():
            if mdp.isTerminal(st):
              continue
            """hou een lijst bij met possible values van possible actions"""
            possibleValues = []
            for act in mdp.getPossibleActions(st):
              possibleValues.append(self.getQValue(st, act))
            """neem de value van de beste actie"""
            v[st] = max(possibleValues)
          self.iterations -= 1
          self.values = v


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
        "*** YOUR CODE HERE ***"
        sucs = self.mdp.getTransitionStatesAndProbs(state, action)
        Qval = 0
        """"hier moest geen haakjes (st, prob) wordt anders gezien als tuple en niet individuele dingen"""
        for st, prob in sucs:
            Qval += prob * (self.mdp.getReward(state, action, st) + (self.values[st] * self.discount))
        return Qval


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        """als er geen legal actions is (terminal state) return None (zie tekst boven)"""
        if self.mdp.isTerminal(state):
          return None

        """gebruik hier ook counter voor easyness om keys met values op te slaan (acties = key, value = qvalue)"""
        actFromVal = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for a in actions:
            qVal = self.getQValue(state, a)
            actFromVal[a] = qVal
        """return key (actie) met de hoogste value"""
        return actFromVal.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
