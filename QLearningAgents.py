from pacman import Directions
from game import Agent
import random
import game
import util

##############
# My agent uses Q-Learning. The q values are stored in a dictionary, indexed by (state,value) pairs.
# The 'state' is the state object used by the game.
# This object is of type 'GameState'. It encodes Pacman's position, food locations, ghosts and Pacman's score.
# This class has methods (specifically__hash__() and __eq()__) which allow it to be used this way
##############

class extendedDictionary():
    # This is basically a dictionary, with the added benefit that it returns 0 if the key isn't found
    def __init__(self):
        self.dict = {}

    def __getitem__(self,x):
        if x in self.dict.keys(): return self.dict[x]
        else: return 0.0

    def __setitem__(self,x,data):
        self.dict[x] = data

    def __len__(self):
        return len(self.dict)

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # My code begins here:
        # We create a dictionary of q values, to be indexed by (state,value) pairs
        self.Q_table = util.Counter()

        # We create a dictionary of N_sa counts (i.e. how many times we have had (state,action)
        self.N_sa = util.Counter()

        # We initialise variables to persist the previous state, previous action and previous reward between moves
        self.s = None
        self.a = None
        self.r = 0

        print "Final version for submission."

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # New functions

    def calculateMaxQ(self, state):
        # This function takes a state object and returns the maximum Q value for any states reachable from that state

        # For the state passed to the function, we lookup all of the state action pairs for each of the legal actions.
        # If the value isn't available for an action, we will get 0 back.
        listOfNeighbouringQValues = [self.Q_table[(state,a)] for a in state.getLegalPacmanActions()]

        # We return the maximum Q value in the list, unless it was empty in which case we just return 0
        return max(listOfNeighbouringQValues) if len(listOfNeighbouringQValues) > 0 else 0

    def selectAMove(self, state):
        # This function chooses the best moves according to the values in the Q-values table

        # Create a dictionary, keyed by actions from the list of legal actions, with the qvalue for each action
        movesWithQValues = { action : self.Q_table[(state, action)] for action in state.getLegalPacmanActions() }

        # Return whichever action from the current state has the highest q value
        return max(movesWithQValues,key=movesWithQValues.get)


    def updateQ(self,state,reward):
        # This is the update rule for qlearning

        if self.s != None: # Check this isn't the very first action, in which case there is no previous state

            # Increment N[(s,a)] - switched off, see comment below on the update rule
            # self.N_sa[(self.s,self.a)] = self.N_sa[(self.s,self.a)] + 1

            # We apply the update rule for q-learning.
            q = self.Q_table[(self.s,self.a)]
            alpha = self.alpha
            Nsa = self.N_sa[(self.s,self.a)]
            gamma = self.gamma
            MaxQ =self.calculateMaxQ(state)

            # The full q-learning update rule is below commented out
            #self.Q_table[(self.s, self.a)] = q + alpha * Nsa * (reward + gamma * MaxQ - q)

            # However, this agent actually performs better when we do not adjust alpha by N[(s,a)], so I have used the simpler update rule below
            self.Q_table[(self.s, self.a)] = q + alpha * (reward + gamma * MaxQ - q)



    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # Identify the legal moves
        legal = state.getLegalPacmanActions()

        # We calculate the reward. We define this as the difference in score since the last move.
        # This incentivises eating food which increases scores and disincentivises dying which decreases scores.
        reward = state.getScore() - self.r

        # We update the Q value associated with the previous (state,action) pair.
        # We pass the current state so it can calculate the maximum Q and we pass the reward which is required in the update rule
        self.updateQ(state,reward)

        # Epsilon-greedy action selection.
        # We explore if a random number between 0 and 1 is less than epsilon, otherwise exploit using the Q value table.
        if random.uniform(0,1) < self.epsilon: pick = random.choice(legal)
        else: pick = self.selectAMove(state)

        # We persist the current state, action and reward so it can be used to update the Q value tables in the next move
        self.s = state
        self.a = pick
        self.r = reward

        # Take the selected action
        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # We also have to update the Q table here because the getAction() method isn't called after Pacman wins/dies.
        # This is very important given it's the main thing we care about!
        # Just as before, we calculate the reward and update the q value for the previous (state,action) pair
        reward = state.getScore() - self.r
        self.updateQ(state, reward)

        # Reset the three things we persist between moves for the update rule once the game is over, before the next begins.
        self.r = 0
        self.s = None
        self.a = None

        # From the framework code given to us:
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)




