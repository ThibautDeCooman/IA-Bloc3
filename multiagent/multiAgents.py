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
        score = successorGameState.getScore()
        
        #Gagner est excellent
        if successorGameState.isWin():
            return 999999999
            
        #Distance avec chaque nourriture, on retient la plus proche
        foodDistances = [manhattanDistance(x, newPos) for x in newFood.asList()]
        closestFood = min(foodDistances)
        
        score += 1./closestFood
        score += 10
        
        #On calcule la distance avec les fantomes et on s'arrange pour toujours etre a une certaine distance (ici 1) d'eux
        ghostDistances = [manhattanDistance(newPos, newGhostStates[ghost].getPosition()) for ghost in range(len(newGhostStates))]
        for distance in ghostDistances:
            if distance <= 1:
                score -= 10
        
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
        bestAction = None
        bestCost = -float("inf")
        for action in gameState.getLegalActions(0):
            newState = self.result(gameState, action, 0)
            cost = self.min_value(newState, 0, 1)
            if cost > bestCost:
                bestCost = cost
                bestAction = action
                
        return bestAction
        
    def terminal_test(self, state, depth):
        return state.isWin() or state.isLose() or depth >= self.depth
        
    def utility(self, state):
        return self.evaluationFunction(state)
        
    def result(self, state, action, agentIndex):
        return state.generateSuccessor(agentIndex, action)
        
    def max_value(self, state, depth):
        if self.terminal_test(state, depth):
            return self.utility(state)
        
        v = -float("inf")
        # Pour chaque action de Pacman
        for action in state.getLegalActions(0):
            newState = self.result(state, action, 0)
            # On lance sur le premier fantome
            v = max(v, self.min_value(newState, depth, 1))
        return v
        
    def min_value(self, state, depth, ghostIndex):
        if self.terminal_test(state, depth):
            return self.utility(state)
            
        v = float("inf")
        for action in state.getLegalActions(ghostIndex):
            newState = self.result(state, action, ghostIndex)
            
            if ghostIndex < state.getNumAgents() - 1:
                # On passe au fantome suivant
                v = min(v, self.min_value(newState, depth, ghostIndex+1))
            else:
                # On passe a Pacman
                v = min(v, self.max_value(newState, depth + 1))
            
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float("inf")
        beta = float("inf")
        bestAction = None
        bestCost = -float("inf")
        for action in gameState.getLegalActions(0):
            newState = self.result(gameState, action, 0)
            cost = self.min_value(newState, 0, 1, alpha, beta)
            if cost > bestCost:
                bestCost = cost
                bestAction = action
            alpha = max(alpha, cost)
                
        return bestAction
        
    def terminal_test(self, state, depth):
        return state.isWin() or state.isLose() or depth >= self.depth
        
    def utility(self, state):
        return self.evaluationFunction(state)
        
    def result(self, state, action, agentIndex):
        return state.generateSuccessor(agentIndex, action)
        
    def max_value(self, state, depth, alpha, beta):
        if self.terminal_test(state, depth):
            return self.utility(state)
        
        v = -float("inf")
        # Pour chaque action de Pacman
        for action in state.getLegalActions(0):
            newState = self.result(state, action, 0)
            # On lance sur le premier fantome
            v = max(v, self.min_value(newState, depth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
            
        return v
        
    def min_value(self, state, depth, ghostIndex, alpha, beta):
        if self.terminal_test(state, depth):
            return self.utility(state)
            
        v = float("inf")
        for action in state.getLegalActions(ghostIndex):
            newState = self.result(state, action, ghostIndex)
            
            if ghostIndex < state.getNumAgents() - 1:
                # On passe au fantome suivant
                v = min(v, self.min_value(newState, depth, ghostIndex+1, alpha, beta))
            else:
                # On passe a Pacman
                v = min(v, self.max_value(newState, depth + 1, alpha, beta))

            if v < alpha:
                return v
            beta = min(beta, v)
            
        return v

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
        bestAction = None
        bestCost = -float("inf")
        for action in gameState.getLegalActions(0):
            newState = self.result(gameState, action, 0)
            cost = self.chance_value(newState, 0, 1)
            if cost > bestCost:
                bestCost = cost
                bestAction = action
                
        return bestAction
        
    def terminal_test(self, state, depth):
        return state.isWin() or state.isLose() or depth >= self.depth
        
    def utility(self, state):
        return self.evaluationFunction(state)
        
    def result(self, state, action, agentIndex):
        return state.generateSuccessor(agentIndex, action)
        
    def max_value(self, state, depth):
        if self.terminal_test(state, depth):
            return self.utility(state)
        
        v = -float("inf")
        # Pour chaque action de Pacman
        for action in state.getLegalActions(0):
            newState = self.result(state, action, 0)
            # On lance sur le premier fantome
            v = max(v, self.chance_value(newState, depth, 1))
        return v
        
    def min_value(self, state, depth, ghostIndex):
        if self.terminal_test(state, depth):
            return self.utility(state)
            
        if ghostIndex < state.getNumAgents() - 1:
            # On passe au fantome suivant
            return self.chance_value(state, depth, ghostIndex+1)
        else:
            # On passe a Pacman
            return self.max_value(state, depth + 1)

    def chance_value(self, state, depth, ghostIndex):
        if len(state.getLegalActions(ghostIndex)) == 0:
            return self.utility(state)

        value = 0
        for action in state.getLegalActions(ghostIndex):
            newState = self.result(state, action, ghostIndex)
            value += self.min_value(newState, depth, ghostIndex)

        value /= float(len(state.getLegalActions(ghostIndex)))
        return value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    
    # Gagner est excellent
    if currentGameState.isWin():
        return 999999999
        
    # Distance avec chaque nourriture, on retient la plus proche
    foodDistances = [manhattanDistance(x, newPos) for x in newFood.asList()]
    closestFood = min(foodDistances)
    
    # On donne une assez bonne importance a manger
    score += 1./closestFood * 5
    
    # On calcule la distance avec les fantomes
    ghostDistances = [manhattanDistance(newPos, newGhostStates[ghost].getPosition()) for ghost in range(len(newGhostStates))]
    for i in range(len(ghostDistances)):
        distance = ghostDistances[i]
        # Si on pense qu'on peut manger le fantome, on force Pacman a se rendre vers le fantome
        if newScaredTimes[i] >= distance:
            score += 1./(distance+1) * 100
        # Sinon, on fuit le fantome
        elif distance <= 5:
            score -= 10

    return score

# Abbreviation
better = betterEvaluationFunction

