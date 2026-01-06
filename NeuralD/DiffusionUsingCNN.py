from NetworkForDiff import NeuralNetwork
from PIL import Image
import DealWithNoise
import random
import os
import pickle
import numpy as np


imageSize = (40, 40) # (x, y)

network = NeuralNetwork(imageSize[0]*imageSize[1] + 1, 10, 80, imageSize[0]*imageSize[1]) # input neurons, hidden layer number, neurons per hidden layer, output neurons
loadName = "diffusionModel_5.80" # diffusionModelN for when predicting noise, otherwise just diffusionModel. check whats after the underscore for version
saveName = "diffusionModel_10.80" # diffusionModelN for when predicting noise, otherwise just diffusionModel. check whats after the underscore for version

iterations = 10
saveEvery = 10

limit = 30

def trainNetwork(network: NeuralNetwork, imagesPath: str):
    images = getImagePaths(imagesPath)
    total = iterations * len(images)
    last_percent = -1
    for iteration in range(iterations):
        for i, path in zip(range(limit), images):
            img = Image.open(path).convert("L").resize(imageSize)
            img_arr = imageToMatrix(img)  # (H, W) float32
            t = random.random()

            noise = np.random.normal(0.0, 0.1, imageSize).astype(np.float32)
            new_img = (1 - t) * img_arr + t * noise
            new_img = np.clip(new_img, 0, 1)

            flat_new = (new_img * 2 - 1).ravel().tolist()  # scale [0,1]→[-1,1]
            flat_orig = img_arr.ravel().tolist()

            flat_new.append(t)
            network.addInputs(flat_new)
            network.forwardPass()
            network.backpropagate(flat_orig, 0.008)  # slightly higher LR

            completed = iteration * len(images) + (i + 1)
            percent = int((completed / total) * 100)
            if percent != last_percent:
                print(f"Progress: {percent}%")
                last_percent = percent

        if (iteration + 1) % saveEvery == 0:
            with open(saveName + f"_iteration#{iteration}.pkl", "wb") as f:
                pickle.dump(network, f)

def useNetwork(path: str):
    with open(path + ".pkl", "rb") as f:
        net = pickle.load(f)
    
    startingImage = DealWithNoise.getNoise(imageSize[0], imageSize[1])
    flatImg = [item for sublist in startingImage for item in sublist]
    timesteps = 800
    for i in range(timesteps):
        flatImg.append((timesteps-i)/timesteps)
        net.addInputs(flatImg)
        net.forwardPass()
        result = net.neurons[-1]
        #noUse = flatImg.pop() # comment out when model is not predictiong noise
        #result = DealWithNoise.removeNoiseFlat(flatImg, (timesteps-i)/timesteps, result) # comment out when model is not predictiong noise
        result = np.clip(result, 0, 1)
        flatImg = DealWithNoise.addFlatNoise(result, 0.4*(i/timesteps), DealWithNoise.getFlatNoise(imageSize[0], imageSize[1]))
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
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

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