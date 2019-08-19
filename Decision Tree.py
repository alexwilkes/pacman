from __future__ import division
from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import math
import numpy as np
from uuid import uuid4
import copy

# My classifier is a decision tree with post-pruning to reducing over-fitting.
# The decision tree is built recursively, selecting the attribute that minimises average gini score.
# The data is split 80/20, with 80% used to build the tree, and the remaining data is held back to implement post-pruning.
# Post-pruning is implemented by iteratively pruning nodes at the bottom of the tree whenever pruning improves accuracy on the 20%.
# No external machine learning libraries (e.g. SKLearn) have been used.
# My code includes tree learning, tree pruning, prediction and train/test splitting

def getMinValueOfDictionary(d):
    # Utility function: given a dictionary it will return a list of the keys associated with the min value.
    minValue = min(d.values())
    keys = [k for k, v in d.items() if v == minValue]
    return keys

def modalValue(array):
    # Utility function: returns the most common item in a list
    return max(set(array), key=list(array).count)

def gini(c):
    # The gini measure for identifying the best attribute to split on
    # The function takes a 1D numpy array and returns a float which is the gini measure for that array

    # Count the frequency of each class and store a total so we can calculate the probabilities
    y_counts = np.unique(c, return_counts=True)[1]
    size = np.sum(y_counts)

    # Calculate the probabilities, and store a list of the sum of the squares of them
    probabilities = [value / size for value in y_counts]
    H = [p ** 2 for p in probabilities]

    # Return 1 - sum(probabilities)
    return 1 - np.sum(H)


def splitBranch(combineddata, attribute):
    # Takes a combined numpy array and returns two, split on the passed attribute (always assumed to only contain 0s or 1s).

    # First we filter the data, splitting on whether the attribute is 0 or 1
    split1 = combineddata[combineddata[:, attribute] == 0]
    split2 = combineddata[combineddata[:, attribute] == 1]

    # Then we remove the attribute we used as we don't use it again
    split1 = np.delete(split1, attribute, axis=1)
    split2 = np.delete(split2, attribute, axis=1)

    # Return a tuple with the two new datasets
    return (split1, split2)


def expectedGini(combineddata, attribute):
    # This function calculates the expected gini index across the two resultant datasets for a branch.
    # It assumes both the data and the target are labelled 0/1.
    # It will not work for non-binary attributes

    class1, class2 = splitBranch(combineddata, attribute)

    class1_size = len(class1) / (len(class1) + len(class2))
    class2_size = len(class2) / (len(class1) + len(class2))

    return class1_size * gini(class1[:, -1]) + class2_size * gini(class2[:, -1])


def bestAttribute(dataset):
    # This method takes a 2D numpy array and returns the attribute on which a binary split will result in the lowest average gini index

    columns = range(len(dataset[0]))  # Simply creates a list of attribute IDs of the required length

    # Create a dictionary, keyed by the attributes, with the entropy/gini for each
    expectedGinis = {columns[t]: expectedGini(dataset, t) for t in range(len(columns) - 1)}

    # There may be more than one 'best attribute' where they have the same expected gini value, in which case we select one of the highest at random
    bestattribute = random.choice(getMinValueOfDictionary(expectedGinis))

    return bestattribute


def shouldTerminate(dataset):
    # This method supports the control of the iteration that builds the decision tree. It makes this decision based on the dataset passed to it.
    # We terminate if either all the samples have the same class (case 1) OR if we have run out of examples (case 2) OR we have run out of attributes (case 3)

    # Split the combined dataset in to the X's and the labels
    classes = dataset[:, -1]
    attributes = dataset[:, 0:-1]

    # Case 1
    doITerminate = all(classes[0] == thisClass for thisClass in classes)
    if doITerminate: return True

    # Case 2
    if len(attributes) == 0:
        return True

    # Case 3
    else:
        doITerminate = all(all(attributes[0] == att) for att in attributes)
        return doITerminate


def plurality(classes):
    # This method takes a list of classes, and returns the most common one. It is used for classification at leaf nodes.
    return max(set(classes), key=list(classes).count)

def classification(dataset, parentdataset):
    # If the classes for the data at the leaf node are all the same, then we return that class
    if len(np.unique(dataset[:, -1])) == 1:
        return dataset[0, -1]

    # If we have more than one class in the dataset, we take the most common result in the dataset (plurality)
    elif len(np.unique(dataset[:, -1])) > 1:
        return plurality(dataset[:, -1])

    # If we have nothing in the dataset (no examples), then we take plurality from the parent
    else:
        return plurality(parentdataset[:, -1])


class Node:
    # The objects in my tree are nodes which can be either 'branchingNode' or 'leafNode.'  This class is a parent class for both which contains pruning logic.
    # Nodes are referenced in the tree by a uuid which is generated by the uuid library on demand.

    def pruneTreeAt(self, uuid):
        # This method will return a new tree, but pruned at the node supplied. If passed a node ID that doesn't exist, it will simply return the existing tree.

        if self.id == uuid:
            return LeafNode(self.dataset, id=self.id)

        elif self.isLeaf:
            return self

        else:
            return BranchingNode(self.attributeToSplitOn, self.leftTree.pruneTreeAt(uuid),
                                 self.rightTree.pruneTreeAt(uuid), dataset=self.dataset, id=self.id)


class BranchingNode(Node):
    # Branching nodes are those in the tree that are not leaves (i.e. have children).

    def __init__(self, attributeToSplitOn, leftTree, rightTree, dataset=None, isRoot=False, id=None):
        self.isRoot = isRoot # This is true only for the first node created
        self.isLeaf = False
        self.id = str(uuid4()) if id is None else id # We give uuids to every node, so we can reference them later during pruning. If a node id is passed in to the constructor, e.g. after pruning it will preserve the old ID.
        self.dataset = dataset # Store the dataset used to branch on this node. It only gets used in the case of pruning, when the branching node gets converted in to a leaf node.

        # Record which attribute was selected at this point and allocate the two subtrees
        self.attributeToSplitOn = attributeToSplitOn # Store the attribute that was used to branch on this node

        # Store the child nodes of this node
        self.leftTree = leftTree
        self.rightTree = rightTree

    def predict(self, x):
        # Once the tree exists, predict will recursively call itself, traversing the tree until it finds a leaf node, which will return a prediction
        # It takes parameter 'x' which is a feature vector of data

        value = x[self.attributeToSplitOn]  # Get the value of the feature vector at the relevant attribute
        x = np.delete(x, self.attributeToSplitOn)  # Because we delete attributes as we built the tree, we must also delete them from the feature vector to ensure alignment

        # The branching statement below traverses left or right down the tree depending on 'value'
        if value == 0:
            return self.leftTree.predict(x)
        elif value == 1:
            return self.rightTree.predict(x)

    def isPrunable(self):
    # Prunable nodes are those with two leaf children. This method is used by getPrunableNodes()
        return self.leftTree.isLeaf and self.rightTree.isLeaf


class LeafNode(Node):
    # Leaf nodes are those nodes at the bottom of the tree with no children.

    def __init__(self, dataset, parentDataset=None, id=None):
        self.isLeaf = True
        self.id = str(uuid4()) if id is None else id # Similar to branching nodes, we assign uuids to each node.

        # We store the predicted class associated with the classification method. We must pass both the dataset, and the parent's dataset. The latter is only used if the former is empty.
        self.predictedClass = classification(dataset, parentDataset)

    # Predict will be called by the branching node above the leaf node. We can simply return the value we stored in the attribute when creating the tree.
    def predict(self, x):
        # This method returns the predicted class that the constructor stored in the leaf node
        return self.predictedClass

    def isPrunable(self):
        # Leaf nodes are never prunable
        return False

# The decision tree is build recursively. The method checks for the terminating conditions, and otherwise recursively calls itself to split the data using the best attribute.
def generateTree(dataset, parentDataset=None, isRoot=False):
    # First we check for termination conditions
    if (shouldTerminate(dataset)):
        return LeafNode(dataset, parentDataset=parentDataset)

    # Otherwise we choose the attribute to split on, split the dataset and recursively call the function on the two childen.
    else:
        attributeToSplitOn = bestAttribute(dataset)
        splitDataset = splitBranch(dataset, attributeToSplitOn)

        # As well as the two new datasets, we pass the current dataset incase they need to classify using plurality on their parent
        return BranchingNode(attributeToSplitOn, generateTree(splitDataset[0], dataset),
                             generateTree(splitDataset[1], dataset), dataset=dataset, isRoot=isRoot)

class Tree:
    # This class is a wrapper for the node objects, with some additional methods that get called on the entire tree

    # We use this method if we are rebuilding an existing tree (e.g. after pruning)
    def fromNode(self, node):
        self.tree = node

    # We use this method to preprocess the data and then generate the tree.
    def fit(self, data, target):

        # Convert the data to numpy arrays if it isn't already
        data = np.array(data)
        self.target = np.array(target)

        # This will drop any columns for which all values are the same. These don't contain any information.
        # This means the classifier can be safely extended to e.g. random forests where we don't select from all attributes at every node.
        # If we had multiple completely sparse attributes, then random forests fails when all the attributes it considers at a node are sparse.
        self.data = data[:, np.invert(np.all(data == data[0, :], axis=0))]
        self.droppedAttributes = np.arange(data.shape[1])[np.all(data == data[0, :], axis=0)]

        # We combine the two arrays in to one big one, call the generateTree method and store the result
        combineddata = np.concatenate((self.data, self.target[:, None]), axis=1)
        self.tree = generateTree(combineddata, isRoot=True)

    def predict(self, x):
        # This method passes through to the top node of the tree
        return self.tree.predict(x)

    def test(self, X_test, y_test):
        # This method returns the accuracy of the tree on a new dataset. This is required for pruning.
        predictions = [self.predict(x) for x in X_test] # Create predictions for all values in the dataset

        predictions_vs_actuals = np.unique((y_test - predictions) == 0, return_counts=True) # Count the number of correct predictions and store in a dictionary
        predictions_dict = dict(zip(predictions_vs_actuals[0], predictions_vs_actuals[1]))

        total = np.sum(predictions_vs_actuals)

        # Next line manages an edge case where no predictions were correct. Without this the following calculation would have a key error.
        if True not in predictions_dict.keys(): predictions_dict[True] = 0

        # Return the accuracy score
        accuracy = predictions_dict[True] / total
        return accuracy

def getPrunableNodes(node):
    # This function is used for pruning. Given a tree node, it returns all the nodes that could be pruned below it.

    prunableNodes = [] # Create an empty list to store the results of the inner function

    # This inner function is used because we search the tree recursively. If we returned within the outer function, it would stop searching too early.
    def getPrunableNodes_inner(node):
        if node.isPrunable():
            prunableNodes.append(node.id)
        if not node.isLeaf:
            getPrunableNodes_inner(node.leftTree)
            getPrunableNodes_inner(node.rightTree)

    getPrunableNodes_inner(node)
    return prunableNodes

def pruneTree(tree, X_test, y_test):
    # This function implements post-pruning on a tree to reduce overfitting. It takes a tree, and some new data which we held back.
    # It starts at the bottom of the tree, checking 'prunable' nodes. It will compare accuracy before and after pruning a node, and if pruning increases accuracy it will retain it.
    # The function returns a newly pruned tree.

    # We create an initial baseline for accuracy
    treeAccuracy = tree.test(X_test, y_test)

    # This control variable manages the while loop
    continueIterating = True

    while continueIterating:
        continueIterating = False

        # Fetch all prunable nodes on the current tree
        prunableNodes = getPrunableNodes(tree.tree)

        # Iterate through the prunable nodes
        for prunableNode in prunableNodes:

            # Create a new tree with the node pruned
            prunedTree = Tree()
            prunedTree.fromNode(tree.tree.pruneTreeAt(prunableNode))
            prunedTreeAccuracy = prunedTree.test(X_test, y_test)

            # Compare accuracy to the baseline. If 'as good' or better, then we adopt the new tree
            if prunedTreeAccuracy >= treeAccuracy:
                tree = copy.deepcopy(prunedTree)

                continueIterating = True  # Because we changed the tree, there may be new prunable nodes to check.

    return tree

def train_test_split(data, target, test_proportion):
    # This function splits the data in to two, according to the proportions passed in the parameter

    # If necessary, convert the data to numpy objects
    data = np.array(data)
    target = np.array(target)

    # We create a list of random integers which will be the integers of the rows we select in to each group.
    shuffledInts = np.random.permutation(len(data))

    # We chop the list of shuffled integers in the proportions passed in the parameter
    testIndices = shuffledInts[:int(test_proportion * len(data))]
    trainIndices = shuffledInts[int(test_proportion * len(data)):]

    # We return four numpy arrays as a tuple
    return (data[trainIndices], data[testIndices], target[trainIndices], target[testIndices])

# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes. This function and the code that reads the file in to data and target is not mine. Line 383 onwards is.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # I use my train_test_split function to split the data and target in to a training dataset, and hold back 20% which is then used for pruning
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, 0.2)

        # I create a tree object and fit using the training data
        dtree = Tree()
        dtree.fit(X_train, y_train)

        # Prune the tree using the held back data and store the result.
        self.prunedTree = pruneTree(dtree, X_test, y_test)
        
    # Tidy up when Pacman dies (not my code)
    def final(self, state):

        print "I'm done!"

    # Turn the numbers from the feature set into actions (not my code):
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # Get the current feature vector and identify the legal moves
        features = api.getFeatureVector(state)
        legal = api.legalActions(state)

        # We use the predict method of our tree to predict the best move.
        move = self.prunedTree.predict(features)
        move = self.convertNumberToMove(move)
        print move

        return api.makeMove(move, legal)

