from pacman import Directions
from game import Agent
import api
import util

import copy # used for copying my map objects
import math

class bcolors:
    # These colors are used by my pretty printing method of my map class to make the visuals nicer. They are not
    # part of my substantial functionality. I found them on a stackoverflow thread.

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class map:

    # This class is a map object which is used by the reward map and the iterated utility map objects I create. I did
    #  create it myself rather than use the one from the labs.

    def __init__(self, state):
        # The constructor identifies the size of the map through the corners, and then calls a number of functions
        # that begin to populate the detail of the map
        TR = self.getCorners(api.corners(state))[3]  # Capture the TR coordinate for map building
        width = TR[0] + 2
        height = TR[1] + 2
        self.width = width
        self.height = height
        self.matrix = [[0 for x in range(width)] for y in range(height)]

        # We call these functions to populate the map with walls, food, capsules and ghosts
        self.assignWalls(api.walls(state))
        self.assignFood(api.food(state))
        self.assignCapsules(api.capsules(state))
        self.assignGhosts(api.ghosts(state))

        # We call this function to create a list of locations that should be iterated. (i.e. not walls)
        self.identifyLocationsToIterate()

    def getLocation(self,location):
        #This function simply returns the value at a location of a map
        return self.matrix[location[1]][location[0]]

    def setLocation(self,location,value):
        #This setter function sets the value at the location of a map
        self.matrix[location[1]][location[0]] = value

    def assignWalls(self,walls):
        #This function simply iterates through the walls,recording them in the map
        for wall in walls: self.setLocation(wall,"#")

    def assignFood(self,food):
        #This function simply iterates through the capsules, recording them in the map
        for foodLoc in food: self.setLocation(foodLoc,".")

    def assignCapsules(self,capsules):
        #This function simply iterates through the capsules, recording them in the map
        for capsule in capsules: self.setLocation(capsule,"o")

    def assignGhosts(self,ghosts):
        # This function is more complex than the other assignment functions because I reward neighbours of ghosts as well as ghosts.

        # Pacman adjusts how many neighbours of ghosts he cares about depending on the following metric which is a function of map size.
        numberOfNeighboursToAssign = round(math.sqrt(self.width*self.height)/5)

        for ghost in ghosts:
            ghostLoc = (int(ghost[0]),int(ghost[1])) #  Ghost locations arrive as floats so convert them to integers

        # What follows is a sequence of nested for loops. Each loop checks the next layer of neighbours. Rewards are
        # set as consecutive letters of the alphabet, i.e. 'G' for a ghost, 'H' for a neighbour,
        # 'I' for a neighbour's neighbour, and so on. We only record a reward if the map is big enough and we
        # haven't already recorded a higher magnitude reward. This works up to 3 neighbours.

            for neighbour1 in self.getNeighbours(ghostLoc):
                for neighbour2 in self.getNeighbours(neighbour1):
                    for neighbour3 in self.getNeighbours(neighbour2):
                        if numberOfNeighboursToAssign >= 3 and str(self.getLocation(neighbour3)) not in "GHI": self.setLocation(neighbour3,"J")
                    if numberOfNeighboursToAssign >= 2 and str(self.getLocation(neighbour2)) not in "GH": self.setLocation(neighbour2,"I")
                if numberOfNeighboursToAssign >= 1 and str(self.getLocation(neighbour1)) not in "G": self.setLocation(neighbour1, "H")
            self.setLocation(ghostLoc,"G")

        return True


    def identifyLocationsToIterate(self):
        # This function identifies which locations we should iterate once iteration begins. It is a very generalised
        # way to identify locations that aren't walls.

        self.locationsToIterate = []
        for y in range(self.height):
            for x in range(self.width):
                locationValue = self.getLocation((x,y))
                if not str(locationValue) == "#": self.locationsToIterate.append((x,y))

    def __str__(self):
        # This is a colorful map printout. It's a depiction of Pacman's map. It doesn't
        # change Pacman's decisions, it's just used for visualising things.

        prettyMap = ""
        for y in range(self.height):
            for x in range(self.width):
                character = str(self.getLocation((x,self.height - y - 1)))
                if character == "#": character = bcolors.FAIL + character*5 + bcolors.ENDC
                elif character == "o": character = bcolors.OKBLUE + character + bcolors.ENDC
                elif character == ".": character = "  " + bcolors.OKGREEN + character + bcolors.ENDC + "  "
                elif character in "GHIJK": character = "  " + bcolors.WARNING + character + bcolors.ENDC + "  "
                else: character = self.roundchar(float(character))

                prettyMap = prettyMap + " | " + character
            prettyMap = prettyMap + " |\n"

        return prettyMap

    def roundchar(self, num):
        # This is only by the __str__ function to round the values in the map and increases readability
        return '{:.5s}'.format('{:+0.1f}'.format(num))

    def populateRewards(self):
        # Iterate through the map, identifying features in the map and assigning rewards appropriately. Note that if
        # there is no feature we apply a small negative reward (-0.04) to discourage long cycles which don't progress
        # the game.

        for y in range(self.height):
            for x in range(self.width):
                character = str(self.getLocation((x, y)))
                if character == "o": self.setLocation((x, y),+3.00)
                if character == ".": self.setLocation((x, y),+1.00)
                if character == "0": self.setLocation((x, y),-0.04)
                if character == "G": self.setLocation((x, y), -25)  # Corresponds to ghosts
                if character == "H": self.setLocation((x, y), -20)  # Neighbours of ghosts
                if character == "I": self.setLocation((x, y), -15)  # 2nd Neighbours of ghosts
                if character == "J": self.setLocation((x, y), -10)  # 3rd Neighbours of ghosts
                if character == "K": self.setLocation((x, y), -5)   # 4th Neighbours of ghosts

        return True

    def getCorners(self, corners):
        # This method finds the extremities of the map. Credit for this is to Simon Parsons as I used this method
        # after the lab in which it was provided because it's more robust than my own.

        # Setup variable to hold the values
        minX = 100
        minY = 100
        maxX = 0
        maxY = 0

        # Sweep through corner coordinates looking for max and min
        # values.
        for i in range(len(corners)):
            cornerX = corners[i][0]
            cornerY = corners[i][1]

            if cornerX < minX:
                minX = cornerX
            if cornerY < minY:
                minY = cornerY
            if cornerX > maxX:
                maxX = cornerX
            if cornerY > maxY:
                maxY = cornerY

        return [(1, 1), (maxX - 1, 1), (1, maxY - 1), (maxX - 1, maxY - 1)]

    def getLegalMoves(self,location):
        # This method checks the given location and returns all legal moves from that location. It does this by
        # checking all of the candidate moves, and if that move doesn't take Pacman in to a wall then appends that
        # move as a legal move.

        directionsList = [Directions.WEST, Directions.EAST, Directions.NORTH, Directions.SOUTH]
        legal = []
        for move in directionsList:
            if self.isInMap(self.updateLocation(location,move)):
                if not self.getLocation(self.updateLocation(location, move)) == "#": legal.append(move)
        return legal

    def isInMap(self,location):
        # This method simply returns whether Pacman's location is inside the map or not (based on the
        # width/height). It's used by the getLegalMoves function.
        if location[0] < 0 or location[1] < 0 or location[0] >= self.width or location[1] >= self.height: return False
        else: return True

    def updateLocation(self, location, move):
        # This method returns an agent's new location, given a location and a move from a location
        if move == Directions.NORTH:
            return location[0], location[1] + 1
        elif move == Directions.SOUTH:
            return location[0], location[1] - 1
        elif move == Directions.WEST:
            return location[0] - 1, location[1]
        elif move == Directions.EAST:
            return location[0] + 1, location[1]
        elif move == Directions.STOP:
            return location[0], location[1]
        else:
            return False

    def getNeighbours(self,location):
        # This method returns a list of 'neighbours' for a location, i.e. other locations that can be accessed with
        #  one legal move
        return [self.updateLocation(location,move) for move in self.getLegalMoves(location)]

    def rotateMove(self,move,clockwise):
        # This method is a simple dictionary lookup. It takes a move, and returns the move rotated from the given
        # move. A second boolean parameter specifies the direction, where clockwise = True. E.g. North, clockwise ->
        # East. This is used when the agent calculates expected utility under the non-deterministic action model.

        if clockwise: dict = {Directions.NORTH: Directions.EAST, Directions.EAST: Directions.SOUTH, Directions.SOUTH: Directions.WEST, Directions.WEST: Directions.NORTH, Directions.STOP: Directions.STOP}

        else: dict = {Directions.NORTH: Directions.WEST,Directions.WEST:Directions.SOUTH,Directions.SOUTH:Directions.EAST,Directions.EAST:Directions.NORTH,Directions.STOP:Directions.STOP}

        if move in dict: return dict[move]
        else: return False

class MDPAgent(Agent):

    def __init__(self):
        # My agent's constructor simply persists the gamma value I have selected in my design. The map is
        # reconstructed every move so there is little value in storing any details of the map itself.
        self.gamma = 0.61

    def getAction(self, state):

        # Create the rewards map, identify the locations we will iterate and populate the rewards
        rewardMap = map(state)
        rewardMap.identifyLocationsToIterate()
        rewardMap.populateRewards()

        # Create a utility map, copied from the rewards map. It is a copy because this is the one we will iterate
        # whilst keeping the rewards map constant. Copy is an imported library, and deepcopy ensures we get a new
        # copy with no pointers to the old object.
        utilityMap = copy.deepcopy(rewardMap)

        continueIterating = True
        iterationCount = 0

        while continueIterating:

            iterationCount = iterationCount + 1

            oldUtilityMap = copy.deepcopy(utilityMap) # We temporarily store the old map so we can check for convergance after iteration
            utilityMap = self.singleIteration(rewardMap,utilityMap,api.whereAmI(state)) #This is the iteration.

            if self.converged(oldUtilityMap,utilityMap): continueIterating = False # check for convergence


        # Once iteration is complete, we identify the possible moves from our current location, use a separate
        # function to identify the best move, and attempt to make that move.
        legal = utilityMap.getLegalMoves(api.whereAmI(state)) #
        bestMove = self.getBestMove(api.whereAmI(state),legal,utilityMap)

        return api.makeMove(bestMove, legal)

    def getBestMove(self, location, legal, map):
        # This method simply identifies the best move for Pacman once iteration is complete. It takes pacman's
        # location, the legal moves at that location, and Pacman's map with the stable iteration values populated.

        legal.append(Directions.STOP) # Adding STOP back in, as my function doesn't return it

        # Create a dictionary of the utility of the location as keys, with the moves as value so we can identify the best move
        utilities = {map.getLocation(map.updateLocation(location,move)):move for move in legal}
        return utilities[max(utilities)]

    def converged(self,oldmap,newmap):
        # This method checks convergence of two maps it is passed. Convergence is defined as the absolute
        # difference in the value of every location being less than or equal to 0.0001. This seems very strict but
        # actually the additional granularity is achieved in very few additional iterations. It stops pacman randomly
        # wondering in areas with few features.
        for location in oldmap.locationsToIterate:
            if abs(oldmap.getLocation(location) - newmap.getLocation(location)) > 0.0001: return False

        return True

    def singleIteration(self,rewardMap,map,pacmanLocation):
        # This is the substance of the MDP solver. It applies the bellman equation to each location.

        # We again copy the map. This is because the bellman equation references the
        # values from the previous iteration. We have to be careful not to reference values we've updated in the
        # current iteration. Deep copying the map ensures this.
        newMap = copy.deepcopy(map)

        # We iterate through each location
        # Bellman equation: U_i+1 = R + gamma^distance * max [expected utility of neighbours]
        for location in newMap.locationsToIterate:

            # We calculate the reward for the location
            R_s = rewardMap.getLocation(location)

            # We identify the maximum utility of each neighbour and store it in a list
            neighbours = [self.expectedUtility(location,move,map) for move in map.getLegalMoves(location)]

            # Gamma is calculated from the perspective of pacman. The further away a location is, the less pacman
            # cares about it. Gamma_factors uses the manhattandistance for computational ease.
            gamma_factor = self.gamma**util.manhattanDistance(location,pacmanLocation)

            # We update our new map with the calculated result
            newMap.setLocation(location, R_s + gamma_factor*max(neighbours))
        return newMap

    def expectedUtility(self, currentLocation, intendedAction, map):
        # The expected utility takes a location, an 'intended' action and the map. It returns the expected utility of that move.

        # An edge case is that we stop, in which case we get the current location's utility with certainty.
        if intendedAction == Directions.STOP: return map.getLocation(currentLocation)

        # We need to know which moves are legal, to understand where we might stay still rather than move 'left' or 'right'
        legal = map.getLegalMoves(currentLocation)

        # Calculate the utility of most likely outcome
        utility_main = map.getLocation(map.updateLocation(currentLocation, intendedAction))

        # 10% chance we accidentally go 'left' instead of straight on, but if that's not a legal move we stay still
        if map.rotateMove(intendedAction,False) in legal:
            utility_left = map.getLocation(map.updateLocation(currentLocation, map.rotateMove(intendedAction, False)))

        else: utility_left = map.getLocation(currentLocation)

        # 10% chance we accidentally go 'right' instead of straight on, but if that's not a legal move we stay still
        if map.rotateMove(intendedAction,True) in legal:
            utility_right = map.getLocation(map.updateLocation(currentLocation, map.rotateMove(intendedAction, True)))

        else: utility_right = map.getLocation(currentLocation)

        # We return the average of the payoffs, multiplied by the probability of them occurring.
        return 0.8*utility_main + 0.1*utility_left + 0.1*utility_right


