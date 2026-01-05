from NetworkForDiff import NeuralNetwork
from PIL import Image
import DealWithNoise
import random
import os
import pickle
import numpy as np


imageSize = (40, 40) # (x, y)

network = NeuralNetwork(imageSize[0]*imageSize[1] + 1, 10, 80, imageSize[0]*imageSize[1]) # input neurons, hidden layer number, neurons per hidden layer, output neurons
loadName = "diffusionModel_10.80" # diffusionModelN for when predicting noise, otherwise just diffusionModel. check whats after the underscore for version
saveName = "diffusionModel_10.80" # diffusionModelN for when predicting noise, otherwise just diffusionModel. check whats after the underscore for version

iterations = 4000
saveEvery = 250

def trainNetwork(network: NeuralNetwork, imagesPath: str):
    images = getImagePaths(imagesPath)

    total = iterations * len(images)
    last_percent = -1
    for iteration in range(iterations):
        for i in range(len(images)):
            img = Image.open(images[i]).convert("L")
            img = img.resize((imageSize[0], imageSize[1]))
            #img.show()
            #break
            #t = iteration/iterations
            t = random.random()
            noise = DealWithNoise.getNoise(imageSize[0], imageSize[1])
            newImg = DealWithNoise.addNoise(imageToMatrix(img), t, noise)
            newImg = np.clip(newImg, 0, 1)

            flatNewImg = [item for sublist in newImg for item in sublist]
            flatNewImg = [2*x - 1 for x in flatNewImg]  # scale [0,1] → [-1,1]
            flatNoise = [item for row in noise for item in row]
            flatOriginal = [item for row in imageToMatrix(img) for item in row]

            flatNewImg.append(t)

            network.addInputs(flatNewImg)
            network.forwardPass()
            network.backpropagate(flatOriginal, 0.005)

            completed = iteration * len(images) + (i + 1)
            percent = int((completed / total) * 100)
            if percent != last_percent:
                print(f"Progress: {percent}%")
                last_percent = percent

        if (iteration+1) % saveEvery == 0:
            with open( saveName + "_iteration#" + str(iteration) + ".pkl", "wb") as f:
                pickle.dump(network, f)

    with open( saveName +".pkl", "wb") as f:
            pickle.dump(network, f)

def useNetwork(path: str):
    with open(path + ".pkl", "rb") as f:
        net = pickle.load(f)
    
    startingImage = DealWithNoise.getNoise(imageSize[0], imageSize[1])
    flatImg = [item for sublist in startingImage for item in sublist]
    timesteps = 100
    for i in range(timesteps):
        flatImg.append((timesteps-i)/timesteps)
        net.addInputs(flatImg)
        net.forwardPass()
        result = net.neurons[-1]
        #noUse = flatImg.pop() # comment out when model is not predictiong noise
        #result = DealWithNoise.removeNoiseFlat(flatImg, (timesteps-i)/timesteps, result) # comment out when model is not predictiong noise
        result = np.clip(result, 0, 1)
        flatImg = DealWithNoise.addFlatNoise(result, 0.1, DealWithNoise.getFlatNoise(imageSize[0], imageSize[1]))
        #flatImg = np.clip(flatImg, 0, 1)
    
    resultImg = []
    # flat list back to list[y[x]]
    x = []
    total = 0
    for y in range (imageSize[1]):
        for xIndex in range (imageSize[0]):
            x.append(flatImg[total])
            total += 1
        resultImg.append(x)
        x = []
    
    return resultImg
        

def getImagePaths(folderPath):
    files = []

    for filename in os.listdir(folderPath):
        fullPath = os.path.join(folderPath, filename)

        if os.path.isfile(fullPath):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                files.append(fullPath)

    return files

def imageToMatrix(img):
    w, h = img.size
    pixels = img.load()

    matrix = [[0.0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            matrix[y][x] = pixels[x, y] / 255.0

    return matrix

def matrixToImage(matrix):
    h = len(matrix)
    w = len(matrix[0])

    # Convert to 0–255 uint8
    array = np.array(matrix) * 255
    array = array.astype(np.uint8)

    img = Image.fromarray(array, mode='L')  # 'L' = 8-bit grayscale
    return img

if __name__ == "__main__":
    a = input("start over or load? ")
    if a == "start over":
        trainNetwork(network, "NeuralD\\husky")
    elif a == "load":
        b = input("train or use? ")
        if b == "use":
            img = useNetwork(loadName)
            usableImg = matrixToImage(img)
            usableImg.save("output.png")
            usableImg.show()
        elif b == "train":
            with open(loadName + ".pkl", "rb") as f:
                net = pickle.load(f)
            trainNetwork(net, "NeuralD\\husky")