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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        foodPos = newFood.asList()
        foodCount = len(foodPos)
        closestDistance = float('inf')
        for i in range(foodCount):
            man_distance = manhattanDistance(foodPos[i], newPos) + foodCount * 100
            distance = man_distance
            closestDistance = min(distance, closestDistance)
        if foodCount == 0:
            closestDistance = 0
        score = -closestDistance

        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        num_agents = gameState.getNumAgents()
        score = []

        def mini_max_algo(s, iter_count):
            if iter_count >= self.depth * num_agents or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iter_count % num_agents != 0:
                result = float('inf')
                l = s.getLegalActions(iter_count % num_agents)
                List = [x for x in l if x != 'Stop']
                for a in List:
                    sdot = s.generateSuccessor(iter_count % num_agents, a)
                    result = min(result, mini_max_algo(sdot, iter_count + 1))
                return result
            else:
                result = -float('inf')
                l = s.getLegalActions(iter_count % num_agents)
                List = [x for x in l if x != 'Stop']
                for a in List:
                    sdot = s.generateSuccessor(iter_count % num_agents, a)
                    result = max(result, mini_max_algo(sdot, iter_count + 1))
                    if iter_count == 0:
                        score.append(result)
                return result

        mini_max_algo(gameState, 0)
        l = gameState.getLegalActions(0)
        List = [x for x in l if x != 'Stop']
        return List[score.index(max(score))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        num_agents = gameState.getNumAgents()
        score = []

        def alpha_beta_algo(s, iter_count, alpha, beta):
            if iter_count >= self.depth * num_agents or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iter_count % num_agents != 0:
                result = float('inf')
                l = s.getLegalActions(iter_count % num_agents)
                List = [x for x in l if x != 'Stop']
                for a in List:
                    sdot = s.generateSuccessor(iter_count % num_agents, a)
                    result = min(result, alpha_beta_algo(sdot, iter_count + 1, alpha, beta))
                    beta = min(beta, result)
                    if beta < alpha:
                        break
                return result
            else:
                result = -float('inf')
                l = s.getLegalActions(iter_count % num_agents)
                List = [x for x in l if x != 'Stop']
                for a in List:
                    sdot = s.generateSuccessor(iter_count % num_agents, a)
                    result = max(result, alpha_beta_algo(sdot, iter_count + 1, alpha, beta))
                    alpha = max(alpha, result)
                    if iter_count == 0:
                        score.append(result)
                    if beta < alpha:
                        break
                return result

        alpha_beta_algo(gameState, 0, -float('inf'), float('inf'))
        l = gameState.getLegalActions(0)
        List = [x for x in l if x != 'Stop']
        return List[score.index(max(score))]


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

        num_agents = gameState.getNumAgents()
        action_score = []

        def expect_minimax_algo(s, iter_count):
            if iter_count >= self.depth * num_agents or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iter_count % num_agents != 0:
                successor_score = []
                l = s.getLegalActions(iter_count % num_agents)
                List = [x for x in l if x != 'Stop']
                for a in List:
                    sdot = s.generateSuccessor(iter_count % num_agents, a)
                    result = expect_minimax_algo(sdot, iter_count + 1)
                    successor_score.append(result)
                average_score = sum([float(x) / len(successor_score) for x in successor_score])
                return average_score
            else:
                result = -float('inf')
                l = s.getLegalActions(iter_count % num_agents)
                List = [x for x in l if x != 'Stop']
                for a in List:
                    sdot = s.generateSuccessor(iter_count % num_agents, a)
                    result = max(result, expect_minimax_algo(sdot, iter_count + 1))
                    if iter_count == 0:
                        action_score.append(result)
                return result

        expect_minimax_algo(gameState, 0)
        l = gameState.getLegalActions(0)
        List = [x for x in l if x != 'Stop']
        return List[action_score.index(max(action_score))]

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
