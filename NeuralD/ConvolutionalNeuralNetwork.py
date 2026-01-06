import math
import random
import pickle
import os
import numpy as np
import re
import matplotlib.pyplot as plt

class CNN:
    # each "layer" specified counts for one conv and one pooling layer (does not inclue input and output layers)
    # kernel size might need to be odd, dont really care to check weather or not thats true
    def __init__(self, inputSize: tuple[int, int], outputSize: tuple[int, int], kernelSize: int, poolerSize: int, layers: int, kernelsPerLayer: int): 
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.kernelSize = kernelSize
        self.poolerSize = poolerSize
        self.numLayers = layers
        self.kernelsPerLayer = kernelsPerLayer

        self.inputs = []
        self.outputs = []

        self.kernelLayers = [self.initKernels() for _ in range(layers)]
        self.layers = [] # both pooling and conv layers
        self.poolerLayers = [self.initPoolers() for _ in range(layers)]
        self.biases = [[random.random() for _ in range(kernelsPerLayer)] for _ in range(layers)]

        self.distanceFromKernelCenter = int(kernelSize/2) # divide by 2, always round down

    def initKernel(self, kernelSize: int):
        return [[random.random() for _ in range(kernelSize)] for _ in range(kernelSize)]

    def initKernels(self):
        return [self.initKernel(self.kernelSize) for _ in range(self.kernelsPerLayer)]
    
    def initPooler(self, poolerSize: int):
        return [[1 for _ in range(poolerSize)] for _ in range(poolerSize)]

    def initPoolers(self):
        return [self.initPooler(self.poolerSize)]

    def leakyReLU(self, x):
        return x if x > 0 else x * 0.1

    def leakyReLU_derivative(self, x):
        return 1 if x > 0 else 0.1
    
    def addInputs(self, inputs: tuple[int, int]):
        self.inputs = inputs

    def dotProduct(self, input1: list[list[int]], input2: list[list[int]]):
        flattenedIn1 = [item for sublist in input1 for item in sublist]
        flattenedIn2 = [item for sublist in input2 for item in sublist]

        return sum([flattenedIn1[index]*flattenedIn2[index] for index in range(len(flattenedIn1))])
    
    def pad(self, image, pad):
        h = len(image)
        w = len(image[0])
        new_h = h + 2 * pad
        new_w = w + 2 * pad

        padded = [[0 for _ in range(new_w)] for _ in range(new_h)]

        for y in range(h):
            for x in range(w):
                padded[y + pad][x + pad] = image[y][x]

        return padded
    
    def avrAll(self, matrices):
        if not matrices:
            return []

        rows = len(matrices[0])
        cols = len(matrices[0][0])

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for matrix in matrices:
            for i in range(rows):
                for j in range(cols):
                    result[i][j] += matrix[i][j]

        n = len(matrices)
        for i in range(rows):
            for j in range(cols):
                result[i][j] /= n

        return result
    
    def addValue(self, matrix, value):
        rows = len(matrix)
        cols = len(matrix[0])
        # Create new matrix with added value
        return [[matrix[i][j] + value for j in range(cols)] for i in range(rows)]   
    
    def applyLeakyReLU(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        # Create new matrix with added value
        return [[self.leakyReLU(matrix[i][j]) for j in range(cols)] for i in range(rows)]   

    def forwardPass(self):
        # input layer -> first hidden layer
        layer = []
        self.layers = []
        pad = self.distanceFromKernelCenter
        padded_input = self.pad(self.inputs, pad)
        for kernelIndex in range(len(self.kernelLayers[0])):
            kernelOutputForLayer = []
            for yIndex in range(pad, len(padded_input) - pad):
                yLayer = []

                yList = padded_input[yIndex]
                
                for xIndex in range(pad, len(yList) - pad):

                    kernelInput = [
                        [
                            padded_input[yIndex-(yPos-self.distanceFromKernelCenter)][xIndex-(xPos-self.distanceFromKernelCenter)] for xPos in range(self.kernelSize)
                        ] for yPos in range(self.kernelSize)
                    ]

                    result = self.dotProduct(kernelInput, self.kernelLayers[0][kernelIndex])
                    result += self.biases[0][kernelIndex]
                    result = self.leakyReLU(result)
                    yLayer.append(result)

                kernelOutputForLayer.append(yLayer)
                

            layer.append(kernelOutputForLayer)
        
        self.layers.append(layer)
        # should now have a layer list that contains the result of each kernel convolving across the input (layer[kernelsResult[y[x]]])
        # now for all middle layers and output

        for layerIndex in range(1, self.numLayers+1):
            layer = []
            pad = self.distanceFromKernelCenter
            for kernelIndex in range(len(self.kernelLayers[layerIndex-1])):
                kernelOutputForLayer = []
                for kernalOutputOfPrevLayerIndex in range(len(self.layers[layerIndex-1])):
                    padded_input = self.pad(self.layers[layerIndex-1][kernalOutputOfPrevLayerIndex], pad)
                    prevLayersForAvr = []
                    for yIndex in range(pad, len(padded_input) - pad):
                        yLayer = []

                        yList = padded_input[yIndex]
                        
                        for xIndex in range(pad, len(yList) - pad):

                            kernelInput = [
                                [
                                    padded_input[yIndex-(yPos-self.distanceFromKernelCenter)][xIndex-(xPos-self.distanceFromKernelCenter)] for xPos in range(self.kernelSize)
                                ] for yPos in range(self.kernelSize)
                            ]

                            result = self.dotProduct(kernelInput, self.kernelLayers[layerIndex-1][kernelIndex])
                            yLayer.append(result)

                        prevLayersForAvr.append(yLayer)
                    
                    kernelOutputForLayer.append(prevLayersForAvr)
                    
                avr = self.avrAll(kernelOutputForLayer)
                avr = self.addValue(avr, self.biases[layerIndex-1][kernelIndex])
                layer.append(self.applyLeakyReLU(avr))
            self.layers.append(layer)

    def getOutputs(self):
        return self.layers[-1]
    

if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    test_input = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    network = CNN(
        inputSize=(5,5),
        outputSize=(5,5),
        kernelSize=3,
        poolerSize=2,
        layers=1,
        kernelsPerLayer=2
    )

    # Add input
    network.addInputs(test_input)

    # Forward pass
    network.forwardPass()

    # Get outputs
    outputs = network.getOutputs()

    # Print outputs
    for k_index, kernel_output in enumerate(outputs):
        print(f"Kernel {k_index} output:")
        for row in kernel_output:
            print(["{0:.2f}".format(val) for val in row])
        print()
    

    