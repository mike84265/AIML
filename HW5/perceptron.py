# perceptron.py
# -------------
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


# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).

        ## Something you might need to use:
        ## trainingData[i]: a feature vector, an util.Counter()
        ## trainingLabels[i]: label for each trainingData[i]
        ## self.weights[label]: weight vector for a label (class), an util.Counter()
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                ymax, maxScore = None, 0
                for label in self.legalLabels:
                    score = trainingData[i] * self.weights[label]
                    if score >= maxScore:
                        ymax, maxScore = label, score
                if ymax != trainingLabels[i]:
                    self.weights[trainingLabels[i]] += trainingData[i]
                    self.weights[ymax] -= trainingData[i]

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label

        ## Something you might need to use:
        ## self.weights[label]: weight vector for a label (class), an util.Counter()
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        sortedWeight = sorted(self.weights[label].items(), key=lambda x: x[1], reverse=True)
        # print 'sortedWeight = ', sortedWeight
        # print 'weights = ', self.weights
        for feature in sortedWeight[:100]:
            featuresWeights.append(feature[0])

        return featuresWeights
