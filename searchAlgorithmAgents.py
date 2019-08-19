# partialAgent.py
# parsons/15-oct-2017
#
# Version 1
#
# The starting point for CW1.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class PartialAgent(Agent):

    # Create a variable to hold the last action
    def __init__(self):
        self.plan = []
        self.targets = []
        self.currentTarget = []
        self.mode = "happy"
        self.listOfIntersections = []

        #Configuration parameters
        self.stopsAtIntersections = False
        self.sizeOfHeatmap = 5 # Recommend either 3, 5 or 7

    def final(self,state):
        # Resets things
        self.plan = []
        self.targets = []
        self.currentTarget = []
        self.mode = "happy"
        self.listOfIntersections = []

    def getAction(self, state):

        # # Saving the current location and legal actions. I also remove Directions from the legal directions set as
        # I generally want pacman to keep moving. I will add this back in for a limited set of scenarios later.
        TR = self.getCorners(api.corners(state))[3] #Capture the TR coordinate for map building
        myLocation = api.whereAmI(state)
        legal = self.getLegalActions(myLocation, api.walls(state))
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Generates a list of intersections if we haven't yet done it. These will serve as waypoints for pacman to
        # encourage him to explore the whole map. They are also used as stop points.
        if not self.listOfIntersections: self.listOfIntersections = self.identifyIntersections(api.walls(state),
                                                                                               TR[0] + 1, TR[1] + 1)

        # Adds the corners and intersections to the targets lists
        if not self.targets:
            self.targets = self.listOfIntersections + self.getCorners(api.corners(state))

        # Next two lines identify ghosts and removes any ghosts from the list that we can't access (i.e. are fully
        # enclosed by walls)
        ghosts = api.ghosts(state)
        ghosts[:] = [ghost for ghost in ghosts if self.planARoute(myLocation, ghost, api.walls(state))]

        # Updates the list of targets with any additional food and capsules found. We always do this in case we see
        # some food on our way to something else.
        self.targets = self.extendTargetListNoDupes(self.targets, api.food(state) + api.capsules(state))

        # If we've reached a target, remove it from a target list
        if myLocation in self.targets: self.targets.remove(myLocation)

        # Create a 'heatmap' of the map, with 'heat' surrounding the ghosts.
        ghostHeatmap = self.generateHeatMap(TR[0] + 1, TR[1] + 1, ghosts)

        # Find out if we should be happy or worried - we are happy if we are not in the heatmap of a ghost
        self.mode = "worried" if ghostHeatmap[myLocation] > 0 else "happy"

        # The code now branches between the two behaviours for the ghost.
        if self.mode == "worried":  # avoid ghosts!

            # Abandon the current plan
            self.plan = []

            # find a route from me to the nearest ghost, and if possible do the opposite of the first step on that
            # plan If that's not possible, do something random that isn't step towards the ghost. If that's not
            # possible we stop. Note that I round the location of the ghosts incase they appear at non integer
            # locations

            # ghostPlans is a list of plans to each visible ghost
            ghostPlans = [self.planARoute(myLocation, self.roundLocation(ghost), api.walls(state)) for ghost in
                          ghosts]

            # ghostPlansDict is a dictionary, keyed by the distance of the ghost, with the plans as the values
            ghostPlansDict = {len(plan): plan for plan in ghostPlans}

            # closestGhostPlan is the plan of one of the closest ghosts
            closestGhostPlan = ghostPlansDict[min(ghostPlansDict)]

            # Identify a move that would take us closer to the closest ghost so we can avoid it
            aMoveTowardsTheGhostWouldBe = closestGhostPlan[-1]

            # Check whether we can just go in opposite direction
            if self.oppositeMove(aMoveTowardsTheGhostWouldBe) in legal:
                return api.makeMove(self.oppositeMove(aMoveTowardsTheGhostWouldBe), legal)
            # If can't go in opposite direction, then find another direction to go in that isn't towards the ghost
            elif len(legal) > 1:
                if aMoveTowardsTheGhostWouldBe in legal:
                    legal.remove(aMoveTowardsTheGhostWouldBe)
                return api.makeMove(random.choice(legal), legal)
            # Move of last resort - stop!
            else:
                return api.makeMove(Directions.STOP, legal)

        # This is the other major branch - i.e. Pacman's state is happy
        else:
            # Pacman stops at intersections to see more food if configured
            if myLocation in self.listOfIntersections:
                self.listOfIntersections.remove(myLocation)
                if self.stopsAtIntersections: return api.makeMove(Directions.STOP, legal)

            # If we have a plan, do that, if not we will review targets
            if self.plan:
                return api.makeMove(self.plan.pop(), legal)

            # If we don't have a plan, we need to make one.
            else:
                # Calculate the manhattan distance to all the targets
                distances = {}
                for target in self.targets:
                    distance = util.manhattanDistance(target, api.whereAmI(state))
                    if distance in distances:
                        distances[distance].append(target)
                    else:
                        distances[distance] = [target]

                # Find the closest target by manhattan distance
                if distances: self.currentTarget = distances[min(distances)][0]

                # Plan a route to that target
                self.plan = self.planARoute(myLocation, self.currentTarget, api.walls(state))

                # Next line deals with a very specific edge case - that the target is entirely enclosed by walls.
                # We can't plan to it, so the planning function returns false. Instead we remove the target and take
                # a random action.
                if not self.plan:
                    if self.currentTarget in self.targets: self.targets.remove(self.currentTarget)
                    self.plan = [random.choice(legal)]

                # Start to follow our plan!
                return api.makeMove(self.plan.pop(), legal)

    def roundLocation(self, location):
        # I created this function because very occasionally ghosts appear at non-integer locations when they come
        # back to life when they were moving at half speed. Pacman has no way to access those locations so cannot
        # plan a route to them. Instead I plan a route to their location rounded (e.g. (7.5,7.5) is rounded to (8,8))

        return round(location[0]), round(location[1])

    def extendTargetListNoDupes(self, target, newItems):
        # This method takes two lists and returns a new list which is a combination of both, without duplicates. We
        # use this to maintain a list of targets.
        if not target:
            return newItems
        elif not newItems:
            return target
        else:
            return target + list(set(newItems) - set(target))

    def identifyIntersections(self, walls, width, height):
        # This method identifies all locations where there is at least three moves that pacman could make. These
        # serve as useful way points for pacman to visit because from here he can see down the corridors when he stops.

        matrixList = [(x, y) for x in range(width) for y in range(height)]
        for location in matrixList:
            if location in walls: matrixList.remove(location)
        numberOfLegalMovesDict = {location: len(self.getLegalActions(location, walls)) for location in matrixList}
        intersections = [location for location, numberofmoves in numberOfLegalMovesDict.iteritems() if
                         numberofmoves > 2]
        return intersections

    def generateHeatMap(self, width, height, ghosts):
        # This method generates a heatmap, like a mental map for pacman, with 1s around ghosts that it knows about.
        matrixList = [(x, y) for x in range(width) for y in range(height)]
        heatmap = dict(zip(matrixList, ([0] * (width * height))))

        for ghost in ghosts:
            for i in range(self.sizeOfHeatmap):
                for j in range(self.sizeOfHeatmap):
                    heatmap[(ghost[0] + i - (self.sizeOfHeatmap-1)/2, ghost[1] + j - (self.sizeOfHeatmap-1)/2)] = 1

        return heatmap

    def oppositeMove(self, move):
        # This method simply returns the opposite move from what it has been passed
        if move == Directions.NORTH:
            return Directions.SOUTH
        elif move == Directions.SOUTH:
            return Directions.NORTH
        elif move == Directions.WEST:
            return Directions.EAST
        elif move == Directions.EAST:
            return Directions.WEST
        else:
            return False

    def getCorners(self, corners):
        # This method finds the extremities of the map. Credit for this is to Simon Parsons as I used this method
        # after the lab in which it was provided because it's more robust than my own. Mine is left below in comments.

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

    # self.corners = dict(zip(["BL", "TR", "TL", "BR"], api.corners(state)))
    # self.corners["BL"] = (self.corners["BL"][0] + 1, self.corners["BL"][1] + 1)
    # self.corners["TL"] = (self.corners["TL"][0] + 1, self.corners["TL"][1] - 1)
    # self.corners["BR"] = (self.corners["BR"][0] - 1, self.corners["BR"][1] - 1)
    # self.corners["TR"] = (self.corners["TR"][0] - 1, self.corners["TR"][1] + 1)

    def planARoute(self, myLocation, target, walls):
        # These two functions are general pathfinding algorithm that Pacman can use to find the route to any given
        # location (food, capsules, ghosts). It takes an origin (myLocation), destination (target), and the walls which
        # need to be navigated around. It creates a dictionary, the key of which is a location (x,y) and the value of
        # which is the location which leads us there. It then creates a queue of locations, and explores each location
        # that can be reached with one move from there, adding these to the dictionary. It does this until it has
        # found the target. Once it has found the target it then uses the dictionary, working backwards from the target
        # until it finds the starting location
        dictionary = {}
        locationsToExplore = util.Queue()
        locationsToExplore.push(myLocation)
        dictionary[myLocation] = None
        while not locationsToExplore.isEmpty():
            currentLocation = locationsToExplore.pop()

            legal = self.getLegalActions(currentLocation, walls)
            for move in legal:
                if self.updateLocation(currentLocation, move) not in dictionary:
                    dictionary[self.updateLocation(currentLocation, move)] = currentLocation
                    if self.updateLocation(currentLocation, move) == target:
                        plan = self.getPath(dictionary, target)
                        plan.reverse()
                        return plan
                    locationsToExplore.push(self.updateLocation(currentLocation, move))

        # If no plan has been found it returns False
        return False

    def getPath(self, dictionary, target):
        # This function takes the dictionary and the target we are looking for, and traces back from the target through
        # the dictionary returning a sequence of moves that will get you from the target to the origin
        plan = []
        location = target
        while location != None:
            if dictionary[location] != None: plan.insert(0, self.twoLocsToMove(dictionary[location], location))
            location = dictionary[location]
        return plan

    def getLegalActions(self, location, walls):
        # Returns the legal action that can be undertaken given a location and walls
        directionsList = [Directions.WEST, Directions.EAST, Directions.NORTH, Directions.SOUTH]
        legal = []
        for move in directionsList:
            if self.updateLocation(location, move) not in walls: legal.append(move)

        return legal

    def twoLocsToMove(self, location1, location2):
        # Returns a move that would get you from location1 to location2, if it exists, otherwise returns false
        if (location2[0] - location1[0] == -1) and (location2[1] - location1[1] == 0):
            return Directions.WEST
        elif (location2[0] - location1[0] == 1) and (location2[1] - location1[1] == 0):
            return Directions.EAST
        elif (location2[1] - location1[1] == 1) and (location2[0] - location1[0] == 0):
            return Directions.NORTH
        elif (location2[1] - location1[1] == -1) and (location2[0] - location1[0] == 0):
            return Directions.SOUTH
        else:
            return Directions.STOP

    def updateLocation(self, location, move):
        # Returns an agent's new location, given a location and a move from that location
        if move == Directions.NORTH:
            return location[0], location[1] + 1
        elif move == Directions.SOUTH:
            return location[0], location[1] - 1
        elif move == Directions.WEST:
            return location[0] - 1, location[1]
        elif move == Directions.EAST:
            return location[0] + 1, location[1]
        else:
            return False
