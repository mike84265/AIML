# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from searchAgents import mazeDistance
        from util import manhattanDistance
        x, y = newPos
        foodEnumeration = 0
        ghostEnumeration = 0
        newFoodList = newFood.asList()
        nearFood = [food for food in newFoodList if food[0] in range(x-2,x+3) and food[1] in range(y-2,y+3)]
        if len(nearFood) > 0:
            for food in nearFood:
                foodEnumeration += 250 / mazeDistance(newPos, food, successorGameState)
        if currentGameState.hasFood(*newPos):
            foodEnumeration += 1000

        
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                ghostPos = ghost.getPosition()
                ghostPosInt = int(ghostPos[0]), int(ghostPos[1])
                ghostEnumeration += 5000 / (mazeDistance(newPos, ghostPosInt, successorGameState) ** 2) if not ghostPosInt == newPos else 999999
        if foodEnumeration == 0 and not action == Directions.STOP:
            newFoodList.sort(key=lambda x: manhattanDistance(newPos, x))
            if manhattanDistance(newPos, newFoodList[0]) <= 6:
                foodEnumeration += 250 / mazeDistance(newPos, newFoodList[0], successorGameState)
            else:
                if abs(food[0]-newPos[0]) > abs(food[1]-newPos[1]):
                    if food[0] > newPos[0] and action == Directions.EAST: foodEnumeration += 500
                    elif food[0] < newPos[0] and action == Directions.WEST: foodEnumeration += 500
                else:
                    if food[1] > newPos[1] and action == Directions.NORTH: foodEnumeration += 500
                    elif food[1] < newPos[1] and action == Directions.SOUTH: foodEnumeration += 500
                
        ret = foodEnumeration - ghostEnumeration if ghostEnumeration > 300 else foodEnumeration
        # print 'Action = ', action, 'ret = ', ret, 'foodEnumeration = ', foodEnumeration, 'ghostEnumeration = ', ghostEnumeration
        return ret

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agentIndex):
            legalMoves = state.getLegalActions(agentIndex)
            if depth == 0 or state.isWin() or state.isLose() :
                return self.evaluationFunction(state)
            if Directions.STOP in legalMoves: legalMoves.remove(Directions.STOP)
            if agentIndex == 0:
                ret = -999999
                for action in legalMoves:
                    tmp = minimax(state.generateSuccessor(agentIndex, action), depth-1, 1)
                    ret = tmp if tmp > ret else ret
            else:
                ret = 999999
                for action in legalMoves:
                    nextAgent = agentIndex+1 if agentIndex < state.getNumAgents()-1 else 0
                    tmp = minimax(state.generateSuccessor(agentIndex, action), depth-1, nextAgent)
                    ret = tmp if tmp < ret else ret
            return ret

        PacmanLegalMoves = gameState.getLegalActions()
        if Directions.STOP in PacmanLegalMoves: PacmanLegalMoves.remove(Directions.STOP)
        scores = [minimax(gameState.generatePacmanSuccessor(action), gameState.getNumAgents()*(self.depth-1), 1) for action in PacmanLegalMoves] 
        bestScore = max(scores)
        print bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        print PacmanLegalMoves[chosenIndex]
        return PacmanLegalMoves[chosenIndex] 

        from copy import deepcopy
        removeStop = False
        states = []
        legalMoves = gameState.getLegalActions()
        if removeStop: legalMoves.remove(Directions.STOP)
        states = [ (action, gameState.generatePacmanSuccessor(action)) for action in legalMoves]
        for d in range(1, self.depth):
            currentStates = deepcopy(states)
            del states[:]
            for s in currentStates:
                legalMoves = s[1].getLegalActions()
                if len(legalMoves) > 0 and removeStop: legalMoves.remove(Directions.STOP)
                for action in legalMoves:
                    states.append((s[0], s[1].generatePacmanSuccessor(action)))
        if len(states) > 0:
            scores = [self.evaluationFunction(s[1]) for s in states]
            bestScore = max(scores)
            print bestScore
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return states[chosenIndex][0]
        else:
            return Directions.STOP


        # Choose one of the best actions
        # scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

