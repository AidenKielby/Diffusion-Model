import random


def addNoise(image: list[list[float]], t: float, noise: list[list[float]]):
    h = len(image)
    w = len(image[0])
    noisy = [[0.0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            noisy[y][x] = (1 - t) * image[y][x] + t * noise[y][x]

    return noisy

def addFlatNoise(image: list[float], t: float, noise: list[float]):
    h = len(image)
    noisy = [0.0 for _ in range(h)]

    for y in range(h):
        noisy[y] = (1 - t) * image[y] + t * noise[y]

    return noisy

def getNoise(x: int, y: int):
    noise = [[random.gauss(0, 0.1) for _ in range(x)] for _ in range(y)]
    return noise

def getFlatNoise(x: int, y: int):
    noise = [random.gauss(0, 0.1) for _ in range(x*y)]
    return noise

def removeNoise(image: list[list[float]], t: float, noise: list[list[float]]):
    h = len(image)
    w = len(image[0])
    noiseLess = [[0.0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            noiseLess[y][x] = image[y][x] - t * noise[y][x]

    return noiseLess

def removeNoiseFlat(image: list[float], t: float, noise: list[float]):
    h = len(image)
    noiseLess = [0.0 for _ in range(h)]

    for y in range(h):
        noiseLess[y] = image[y] - t * noise[y]

    return noiseLess