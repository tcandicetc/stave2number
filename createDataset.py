import cv2
import numpy as np

def main():
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
                crop_img = image[y+dy:y+dy+h, x+dx:x+dx+w]
                datasetX.append(crop_img)
        y += line_space

    print(len(datasetX))

    '''for i in range(-5, 5):
        cv2.imshow("Output Image", datasetX[16000+i])

        cv2.waitKey(0)
        cv2.destroyAllWindows()'''


if __name__ == '__main__':
    main()