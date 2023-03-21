from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    def __init__(self, state: GameState):
        self.position = state.getPacmanPosition()
        self.food = state.getFood()
        self.ghostPositions = state.getGhostPositions()
        self.ghostStates = state.getGhostStates()
        self.scaredTimes = [ghostState.scaredTimer for ghostState in state.getGhostStates()]

    def __hash__(self):
        return hash((self.position, tuple(self.ghostPositions), self.food))

    def __eq__(self, other):
        return (self.position == other.position and
                self.food == other.food and
                self.ghostPositions == other.ghostPositions)


class QLearnAgent(Agent):
    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.qValues = util.Counter()
        self.counts = util.Counter()
        self.lastState = None
        self.lastAction = None
        self.lastStateFeatures = None



    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        return endState.getScore() - startState.getScore()

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        return self.qValues[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        legalActions = [a for a in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST] if (state, a) in self.qValues]
        return max([self.getQValue(state, action) for action in legalActions]) if legalActions else 0

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        qValue = self.getQValue(state, action)
        nextMaxQValue = self.maxQValue(nextState)
        self.qValues[(state, action)] = qValue + self.alpha * (reward + self.gamma * nextMaxQValue - qValue)

    def updateCount(self, state: GameStateFeatures, action: Directions):
        self.counts[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        return self.counts[(state, action)]

    def explorationFn(self, utility: float, counts: int) -> float:
        return utility + 1 / (1 + counts)
    
    def getEpisodesSoFar(self) -> int:
        return self.episodesSoFar

    def getNumTraining(self) -> int:
        return self.numTraining

    def incrementEpisodesSoFar(self) -> None:
        self.episodesSoFar += 1

    def setAlpha(self, alpha: float) -> None:
        self.alpha = alpha

    def setEpsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def getAction(self, state: GameState) -> Directions:
        if self.lastState is not None:
            reward = self.computeReward(self.lastState, state)
            self.learn(self.lastStateFeatures, self.lastAction, reward, GameStateFeatures(state))

        legalActions = state.getLegalActions()
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            qValues = [(self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action)), action) for action in legalActions]
            maxQValue = max(qValues, key=lambda x: x[0])[0]
            action = random.choice([a for q, a in qValues if q == maxQValue])

        self.lastState = state
        self.lastAction = action
        self.lastStateFeatures = stateFeatures
        self.updateCount(stateFeatures, action)

        return action

    def final(self, state: GameState):
        if self.lastState is not None:
            reward = self.computeReward(self.lastState, state)
            self.learn(self.lastStateFeatures, self.lastAction, reward, GameStateFeatures(state))

        self.lastState = None
        self.lastAction = None
        self.lastStateFeatures = None

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

