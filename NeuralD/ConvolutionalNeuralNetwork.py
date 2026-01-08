import math
import random
import pickle
import os
import numpy as np
import re
import matplotlib.pyplot as plt

class CNN:
    # each "layer" specified counts for one conv and one pooling layer (does not inclue input layer or output layer)
    # kernel size might need to be odd, dont really care to check weather or not thats true
    def __init__(self, inputSize: tuple[int, int], outputSize: tuple[int, int], kernelSize: int, poolerSize: int, layers: int, kernelsPerLayer: int): 
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.kernelSize = kernelSize
        self.poolerSize = poolerSize
        self.numLayers = layers + 1
        self.kernelsPerLayer = kernelsPerLayer

        self.inputs = []
        self.outputs = []

        self.kernelLayers = [self.initKernels() for _ in range(layers + 1)]
        self.layers = [] # both pooling and conv layers
        self.preActivation = [] # post activation function is stored in layers. i think. i am so tired idk anymore
        self.poolerLayers = [self.initPoolers() for _ in range(layers + 1)]
        self.biases = [[0 for _ in range(kernelsPerLayer)] for _ in range(layers + 1)]

        self.distanceFromKernelCenter = int(kernelSize/2) # divide by 2, always round down

    def initKernel(self, kernelSize: int):
        return [[random.uniform(-0.1, 0.1) for _ in range(kernelSize)] for _ in range(kernelSize)]

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
    
    def leakyReLU_derivative_matrix(self, matrix):
        out = []
        for y in matrix:
            layer = []
            for x in y:
                layer.append(self.leakyReLU_derivative(x))
            out.append(layer)
        return out
    
    def addInputs(self, inputs: tuple[int, int]):
        self.inputs = inputs

    def dotProduct(self, input1, input2):
        return float(np.sum(np.array(input1, dtype=np.float32) * np.array(input2, dtype=np.float32)))
    
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
        layerPreAct = []

        self.layers = []
        self.preActivation = []

        pad = self.distanceFromKernelCenter
        padded_input = self.pad(self.inputs, pad)
        for kernelIndex in range(len(self.kernelLayers[0])):
            kernelOutputForLayer = []
            kernelOutputForLayerPreAct = []
            for yIndex in range(pad, len(padded_input) - pad):
                yLayer = []
                yLayerForPreAct = []

                yList = padded_input[yIndex]
                
                for xIndex in range(pad, len(yList) - pad):

                    kernelInput = [
                        [
                            padded_input[yIndex-(yPos-self.distanceFromKernelCenter)][xIndex-(xPos-self.distanceFromKernelCenter)] for xPos in range(self.kernelSize)
                        ] for yPos in range(self.kernelSize)
                    ]

                    result = self.dotProduct(kernelInput, self.kernelLayers[0][kernelIndex])
                    result += self.biases[0][kernelIndex]
                    yLayerForPreAct.append(result)
                    result = self.leakyReLU(result)
                    yLayer.append(result)

                kernelOutputForLayer.append(yLayer)
                kernelOutputForLayerPreAct.append(yLayerForPreAct)
                

            layer.append(kernelOutputForLayer)
            layerPreAct.append(kernelOutputForLayerPreAct)
        
        self.layers.append(layer)
        self.preActivation.append(layerPreAct)
        # should now have a layer list that contains the result of each kernel convolving across the input (layer[kernelsResult[y[x]]])
        # now for all middle layers and output

        for layerIndex in range(1, self.numLayers):
            layer = []
            layerPreAct = []
            pad = self.distanceFromKernelCenter
            for kernelIndex in range(self.kernelsPerLayer):
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

                            result = self.dotProduct(kernelInput, self.kernelLayers[layerIndex][kernelIndex])
                            yLayer.append(result)

                        prevLayersForAvr.append(yLayer)
                    
                    kernelOutputForLayer.append(prevLayersForAvr)
                    
                avr = self.avrAll(kernelOutputForLayer)
                avr = self.addValue(avr, self.biases[layerIndex][kernelIndex])
                layerPreAct.append(avr)
                layer.append(self.applyLeakyReLU(avr))
            self.layers.append(layer)
            self.preActivation.append(layerPreAct)

    def getOutputs(self):
        return self.layers[-1]
    
    def compute_prev_gradient(self, frontGradient, kernel):
        kernel = np.asarray(kernel, dtype=np.float32)
        flipped_kernel = np.flip(np.flip(kernel, 0), 1)
        
        pad = kernel.shape[0] - 1
        padded_gradient = np.pad(frontGradient, ((pad, pad), (pad, pad)), mode='constant')
        
        prevGradient = np.zeros((frontGradient.shape[0] + pad, frontGradient.shape[1] + pad))
        
        for y in range(prevGradient.shape[0] - kernel.shape[0] + 1):
            for x in range(prevGradient.shape[1] - kernel.shape[1] + 1):
                patch = padded_gradient[y:y+kernel.shape[0], x:x+kernel.shape[1]]
                prevGradient[y, x] += np.sum(patch * flipped_kernel)
        
        # crop to match previous layer size
        crop = (kernel.shape[0] - 1) // 2
        prevGradient = prevGradient[crop: -crop, crop: -crop]
        
        return prevGradient
    
    def backpropagate(self, correctOutput: list[list[float]], learningRate: float):
        # loss of output layer is vector of outpits - vector of expected outputs
        expectedOutput = np.array(correctOutput, dtype=np.float32)          # (H, W)
        actualOutput = np.array(self.layers[-1][0], dtype=np.float32)          # (1, H, W)
        deriv = np.array(self.leakyReLU_derivative_matrix(self.preActivation[-1][0]), dtype=np.float32)  # (H, W)
        outputGrad2D = (actualOutput - expectedOutput) * deriv
        outputGrad2D = np.clip(outputGrad2D, -10.0, 10.0)

        # output kernel gradients
        pad = self.distanceFromKernelCenter
        kernel_grads = []
        for i in range(len(self.layers[-2])):
            kernel_grad = np.zeros((self.kernelSize, self.kernelSize))
            padded_input = self.pad(self.layers[-2][i], pad)
            for outputIndexY in range(len(self.layers[-1][0])): # this should go thru every output not flattened
                yOutput = outputGrad2D[outputIndexY]
                for outputIndexX in range(len(yOutput)):
                    kernelOutput = yOutput[outputIndexX]

                    kernelInput = np.array([
                        [
                            padded_input[outputIndexY-(yPos-self.distanceFromKernelCenter)][outputIndexX-(xPos-self.distanceFromKernelCenter)] for xPos in range(self.kernelSize)
                        ] for yPos in range(self.kernelSize)
                    ])

                    kernel_grad += kernelInput * kernelOutput
            #kernel_grad /= (len(self.layers[-1][0]) * len(self.layers[-1][0][0]) * len(self.layers[-2]))
            grad_norm = np.linalg.norm(kernel_grad)
            if grad_norm > 5.0:
                kernel_grad *= (5.0 / grad_norm)
            kernel_grads.append(kernel_grad)
         
        self.kernelLayers[-1][0] = (np.array(self.kernelLayers[-1][0]) - (np.sum(np.array(kernel_grads), axis = 0) * learningRate)).tolist()
        self.biases[-1][0] -= learningRate * np.sum(outputGrad2D)

        frontGradient = outputGrad2D
        for layerIndex in range(2, len(self.layers)):
            
            frontGradSum = 0
            for kernelIndex in range(len(self.kernelLayers[-layerIndex])):
                prevGradient = self.compute_prev_gradient(frontGradient, self.kernelLayers[-layerIndex][kernelIndex])
                prevGradient = np.clip(prevGradient, -10.0, 10.0)
                frontGradSum += prevGradient
                pad = self.distanceFromKernelCenter
                kernel_grads = []

                for i in range(len(self.layers[-layerIndex-1])):
                    kernel_grad = np.zeros((self.kernelSize, self.kernelSize))
                    padded_input = self.pad(self.layers[-layerIndex-1][i], pad)

                    for outputIndexY in range(len(self.layers[-layerIndex][kernelIndex])): # this should go thru every output not flattened
                        yOutput = prevGradient[outputIndexY]

                        for outputIndexX in range(len(yOutput)):
                            kernelOutput = yOutput[outputIndexX]

                            kernelInput = np.array([
                                [
                                    padded_input[outputIndexY-(yPos-self.distanceFromKernelCenter)][outputIndexX-(xPos-self.distanceFromKernelCenter)] for xPos in range(self.kernelSize)
                                ] for yPos in range(self.kernelSize)
                            ])

                            kernel_grad += kernelInput * kernelOutput

                    #kernel_grad /= (len(self.layers[-layerIndex][kernelIndex]) * len(self.layers[-layerIndex][kernelIndex][0]) * len(self.layers[-layerIndex-1]))
                    grad_norm = np.linalg.norm(kernel_grad)
                    if grad_norm > 5.0:
                        kernel_grad *= (5.0 / grad_norm)
                    kernel_grads.append(kernel_grad)
                
                self.kernelLayers[-layerIndex][kernelIndex] = (np.array(self.kernelLayers[-layerIndex][kernelIndex]) - (np.sum(np.array(kernel_grads), axis = 0) * learningRate)).tolist()
                self.biases[-layerIndex][kernelIndex] -= learningRate * np.sum(prevGradient)

            frontGradient = np.clip(frontGradSum / max(1, len(self.kernelLayers[-layerIndex])), -10.0, 10.0)
    

def train_on_symbols(network: CNN, iterations=1000, lr=0.0001):
    # X image: 1s on diagonals
    x_img = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
    ]
    
    # Check mark: roughly 1s in a check pattern
    check_img = [
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    training_data = [
        (x_img, x_img),       # train to reproduce X
        (check_img, check_img) # train to reproduce check
    ]
    
    for it in range(iterations):
        for input_img, target_img in training_data:
            network.addInputs(input_img)
            network.forwardPass()
            network.backpropagate(target_img, learningRate=lr)
        if (it+1) % (iterations//10) == 0:
            print(f"Training iteration {it+1}/{iterations} complete")

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    network = CNN(
        inputSize=(5, 5),
        outputSize=(5, 5),
        kernelSize=3,
        poolerSize=1,
        layers=1,
        kernelsPerLayer=1
    )

    train_on_symbols(network, iterations=3000, lr=0.01)

    # Test outputs
    for symbol_name, symbol_img in [("X", [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]]),
                                    ("Check", [[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,0]])]:
        network.addInputs(symbol_img)
        network.forwardPass()
        output = network.getOutputs()
        print(f"{symbol_name} output:")
        for k_index, kernel_output in enumerate(output):
            print(f"Kernel {k_index}:")
            for row in kernel_output:
                print(["{0:.2f}".format(val) for val in row])
            print()