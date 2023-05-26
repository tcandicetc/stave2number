import cv2
import numpy as np
import matplotlib.pyplot as plt

def buildDataset():
    image = cv2.imread("stave.png")
    image = cv2.resize(image, (560, 800))

    datasetX = [] 

    x = 60
    y = 105
    line_space = 106
    w = 10
    h = 45
    for _ in range(4):
        for dy in range(-4, 5):
            for dx in range(450):
                crop_img = image[y+dy:y+dy+h, x+dx:x+dx+w][:, :, 0]
                datasetX.append(crop_img)
        y += line_space

    datasetSize = len(datasetX)
    print("size of dataset X =", datasetSize)
    np.save('datasetX_1', datasetX[:len(datasetX) // 3])
    np.save('datasetX_2', datasetX[len(datasetX) // 3 : 2 * len(datasetX) // 3])
    np.save('datasetX_3', datasetX[2 * len(datasetX) // 3:])
    np.save('datasetY_1', np.full(datasetSize // 3, -1, dtype=int))
    np.save('datasetY_2', np.full(datasetSize // 3, -1, dtype=int))
    np.save('datasetY_3', np.full(datasetSize // 3, -1, dtype=int))


def label_1():
    datasetX_1 = np.load("datasetX_1.npy")
    datasetY_1 = np.load("datasetY_1.npy")

    start = np.where(datasetY_1==-1)[0][0]
    
    for i in range(start, len(datasetX_1)):
        plt.imshow(datasetX_1[i], cmap='gray', vmin = 0, vmax = 255)
        plt.title(f"image {i}")
        plt.show()
        tmp = input(f"label: ")
        if tmp == 'q':
            break
        else:
            datasetY_1[i] = int(tmp)
    np.save('datasetY_1', datasetY_1)


def label_2():
    datasetX_2 = np.load("datasetX_2.npy")
    datasetY_2 = np.load("datasetY_2.npy")

    start = np.where(datasetY_2==-1)[0][0]
    
    for i in range(start, len(datasetX_2)):
        plt.imshow(datasetX_2[i], cmap='gray', vmin = 0, vmax = 255)
        plt.title(f"image {i}")
        plt.show()
        tmp = input(f"label: ")
        if tmp == 'q':
            break
        else:
            datasetY_2[i] = int(tmp)
    np.save('datasetY_2', datasetY_2)


def label_3():
    datasetX_3 = np.load("datasetX_3.npy")
    datasetY_3 = np.load("datasetY_3.npy")

    start = np.where(datasetY_3==-1)[0][0]
    
    for i in range(start, len(datasetX_3)):
        plt.imshow(datasetX_3[i], cmap='gray', vmin = 0, vmax = 255)
        plt.title(f"image {i}")
        plt.show()
        tmp = input(f"label: ")
        if tmp == 'q':
            break
        else:
            datasetY_3[i] = int(tmp)
    np.save('datasetY_3', datasetY_3)


def checkLabel_1():
    datasetX_1 = np.load("datasetX_1.npy")
    datasetY_1 = np.load("datasetY_1.npy")
    for i in range(0, len(datasetX_1), 10):
        plt.figure(figsize=(10, 5))
        for j in range(10):
            plt.subplot(1, 10, j+1)
            plt.imshow(datasetX_1[i+j], cmap='gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f"{i+j}: {datasetY_1[i+j]}")
        plt.show()


def checkLabel_2():
    datasetX_2 = np.load("datasetX_2.npy")
    datasetY_2 = np.load("datasetY_2.npy")
    for i in range(0, len(datasetX_2), 10):
        plt.figure(figsize=(10, 5))
        for j in range(10):
            plt.subplot(1, 10, j+1)
            plt.imshow(datasetX_2[i+j], cmap='gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f"{i+j}: {datasetY_2[i+j]}")
        plt.show()


def checkLabel_3():
    datasetX_3 = np.load("datasetX_3.npy")
    datasetY_3 = np.load("datasetY_3.npy")
    for i in range(0, len(datasetX_3), 10):
        plt.figure(figsize=(10, 5))
        for j in range(10):
            plt.subplot(1, 10, j+1)
            plt.imshow(datasetX_3[i+j], cmap='gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f"{i+j}: {datasetY_3[i+j]}")
        plt.show()


if __name__ == '__main__':
    # buildDataset() # WARNING: don't run this function or you will reset all the labels
    label_1()
    # checkLabel_1()