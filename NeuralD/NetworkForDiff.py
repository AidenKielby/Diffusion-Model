import math
import random
import pickle
import os
import numpy as np
import re
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        # this part should be self explanitory
        self.inputNeurons = inputNeurons
        self.hiddenLayers = hiddenLayers
        self.neuronsPerHiddenLayer = neuronsPerHiddenLayer
        self.outputNeurons = outputNeurons
        self.neurons = self.initNeurons(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
        self.weights = self.initWeights(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
        self.biases = self.initBiases(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
        self.preActivations = [[] for x in range(self.hiddenLayers + 2)]

    def sigmoid(self, x):
        x = max(min(x, 500), -500)  # prevents overflow
        return 1 / (1 + math.exp(-x))

    def leakyReLU(self, x):
        return x if x > 0 else x * 0.1

    def leakyReLU_derivative(self, x):
        return 1 if x > 0 else 0.1

    def softmax(self, x):
        e_x = [math.exp(i) for i in x]
        total = sum(e_x)
        if total == 0:
            print(e_x)
        return [i / total for i in e_x]

    def addInputs(self, inputs: list[float]):
        if len(inputs) == len(self.neurons[0]):
            self.neurons[0] = inputs

    def initNeurons(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        neurons = []
        l1 = [0 for i in range(inputNeurons)]
        neurons.append(l1)

        for layer in range(hiddenLayers):
            l = [0 for i in range(neuronsPerHiddenLayer)]
            neurons.append(l)

        ln = [0 for i in range(outputNeurons)]
        neurons.append(ln)

        return neurons

    def initWeights(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        # weights -> layer -> toNeuron(index) -> from neuron(index)

        globalInitialWeightValue = 1

        weights = []

        inputWeights = []
        for i in range(neuronsPerHiddenLayer):
            inputWeights.append([random.uniform(-1, 1) * math.sqrt(2 / inputNeurons) for i in range(inputNeurons)])
        weights.append(inputWeights)

        for i in range(hiddenLayers-1):
            layerWeights = []
            for i in range(neuronsPerHiddenLayer):
                layerWeights.append([random.uniform(-1,1) * math.sqrt(1 / neuronsPerHiddenLayer) for i in range(neuronsPerHiddenLayer)])
            weights.append(layerWeights)

        outputWeights = []
        for i in range(outputNeurons):
            outputWeights.append([random.uniform(-1,1) * math.sqrt(1 / neuronsPerHiddenLayer) for i in range(neuronsPerHiddenLayer)])
        weights.append(outputWeights)

        return weights

    def initBiases(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        # due to the input layer not having a bias, all other layers must be acceses as if it was for the layer brfore it
        # for example, for hidden layer number 1 (index 1 for all neurons), you would have to do index 0
        biases = []

        for i in range(hiddenLayers):
            layerBiases = [random.uniform(-0.01, 0.01) for _ in range(neuronsPerHiddenLayer)]
            biases.append(layerBiases)

        outputLayerBiases = [0 for i in range(outputNeurons)]
        biases.append(outputLayerBiases)

        return biases

    def forwardPass(self):
        self.preActivations = [[] for x in range(self.hiddenLayers + 2)]

        for layerIndex in range(self.hiddenLayers):
            layer = self.neurons[layerIndex+1]
            for neuronIndex in range(len(layer)):
                sum = 0
                for fromNeuronIndex in range(len(self.neurons[layerIndex])):
                    fromNeuron = self.neurons[layerIndex][fromNeuronIndex]
                    sum += fromNeuron * self.weights[layerIndex][neuronIndex][fromNeuronIndex]
                sum += self.biases[layerIndex][neuronIndex]
                sum = max(min(sum, 100), -100)

                # store raw_sum for derivative use
                self.preActivations[layerIndex].append(sum)
                self.neurons[layerIndex+1][neuronIndex] = self.leakyReLU(sum)

        layerIndex = self.hiddenLayers+1
        layer = self.neurons[layerIndex]
        #if len(self.neurons[-1]) == 1:
        for neuronIndex in range(len(layer)):
            sum = 0
            for fromNeuronIndex in range(len(self.neurons[layerIndex-1])):
                fromNeuron = self.neurons[layerIndex-1][fromNeuronIndex]
                sum += fromNeuron * self.weights[layerIndex-1][neuronIndex][fromNeuronIndex]
            sum += self.biases[layerIndex-1][neuronIndex]
            sum = max(min(sum, 100), -100)

            # store raw_sum for derivative use
            self.preActivations[layerIndex].append(sum)
            self.neurons[layerIndex][neuronIndex] = self.sigmoid(sum)
        """else:
            sums = []
            for neuronIndex in range(len(layer)):
                total = 0
                for fromNeuronIndex in range(len(self.neurons[layerIndex - 1])):
                    total += self.neurons[layerIndex - 1][fromNeuronIndex] * self.weights[layerIndex - 1][neuronIndex][
                        fromNeuronIndex]
                total += self.biases[layerIndex - 1][neuronIndex]
                sums.append(total)
            self.neurons[layerIndex] = self.softmax(sums)"""

    def cross_entropy(output, target):
        return -sum(t * math.log(o + 1e-9) for o, t in zip(output, target))

    def MSE(self, networkOutput, actualAnswer):
        sum = 0
        for i in range(len(networkOutput)):
            sum += math.pow((networkOutput[i] - actualAnswer[i]), 2)
        return (1 / (len(networkOutput))) * sum

    def BCE(self, output, target):
        return -sum(t * math.log(o + 1e-9) + (1 - t) * math.log(1 - o + 1e-9) for o, t in zip(output, target))

    def backpropagate(self, correctOutput: list[float], learningRate: float):
        loss = self.BCE(self.neurons[-1], correctOutput)
        allLayerWeightGradients = []
        allErrorTerms = []

        outputErrorTerms = []
        outputNeurons = self.neurons[-1]
        outputWeightGradients = []
        prevNeurons = self.neurons[len(self.neurons) - 2]
        outputErrorTerms = [o - t for o, t in zip(outputNeurons, correctOutput)]
        # find the gradient in the output neuron set
        for neuronIndex in range(len(outputNeurons)):

            layerGradient = []
            for prevNeuronIndex in range(len(prevNeurons)):
                layerGradient.append(outputErrorTerms[neuronIndex] * prevNeurons[prevNeuronIndex])
            outputWeightGradients.append(layerGradient)

        allLayerWeightGradients.insert(0, outputWeightGradients)
        allErrorTerms.insert(0, outputErrorTerms)

        layerContributions = []
        lastErrorTerms = outputErrorTerms[:]
        # find the contributions of all hidden layers based off previous layers & error terms
        for layerIndex in range(self.hiddenLayers):
            trueLayerIndex = (self.hiddenLayers) - layerIndex

            layer = self.neurons[trueLayerIndex]
            layerContribution = []
            # contribution
            for neuronIndex in range(len(layer)):
                sum = 0
                for forwardNeuronIndex in range(len(self.neurons[trueLayerIndex+1])):
                    sum += (self.weights[trueLayerIndex][forwardNeuronIndex][neuronIndex] *
                            lastErrorTerms[forwardNeuronIndex])
                layerContribution.append(sum)
            layerContributions.insert(0, layerContribution)

            newErrorTerms = []
            layerGrandients = []
            prevNeurons = self.neurons[trueLayerIndex - 1]
            # error terms
            for neuronIndex in range(len(layer)):
                pre_act = self.preActivations[trueLayerIndex - 1][neuronIndex]
                derivative = self.leakyReLU_derivative(pre_act)
                newErrorTerms.append(layerContribution[neuronIndex] * derivative)

                layerGradient = []
                for prevNeuronIndex in range(len(prevNeurons)):
                    layerGradient.append(newErrorTerms[neuronIndex] * prevNeurons[prevNeuronIndex])
                layerGrandients.append( layerGradient)
            allLayerWeightGradients.insert(0, layerGrandients)
            allErrorTerms.insert(0, newErrorTerms)
            lastErrorTerms = newErrorTerms[:]

        #  update the stuff
        # first, the biases cuz they easy
        for layerIndex in range(len(self.biases)):
            layer = self.biases[layerIndex]
            for biasIndex in range(len(layer)):
                bias = layer[biasIndex]
                bias_grad = allErrorTerms[layerIndex][biasIndex]
                bias_grad = max(min(bias_grad, 5.0), -5.0)  # clip
                bias -= learningRate * bias_grad
                bias = max(min(bias, 3.0), -3.0)  # keep bias bounded
                self.biases[layerIndex][biasIndex] = bias

        # weights
        for layerIndex in range(len(self.weights)):
            layer = self.weights[layerIndex]
            for toNeuronIndex in range(len(layer)):
                toNeuronWeights = layer[toNeuronIndex]
                for weightIndex in range(len(toNeuronWeights)):
                    weight = toNeuronWeights[weightIndex]
                    grad = allLayerWeightGradients[layerIndex][toNeuronIndex][weightIndex]
                    grad = max(min(grad, 5.0), -5.0)  # clip gradient
                    weight -= learningRate * grad
                    weight = max(min(weight, 3.0), -3.0)  # clip weight
                    self.weights[layerIndex][toNeuronIndex][weightIndex] = weight


if __name__ == "__main__":

    net = NeuralNetwork(2500, 2, 64, 1)