import cv2
import numpy as np
import pickle
import gzip

def main(input_file, output_prefix, model):
    image = cv2.imread(input_file)
    image = cv2.resize(image, (560, 800))
    output_image = image.copy()
    
    x = 60
    y = 105
    line_space = 106
    w = 10
    h = 45
    for _ in range(2):
        dx = 0
        while dx < 450:
            crop_img = image[y:y+h, x+dx:x+dx+w][:, :, 0]
            pred = model.predict([np.ravel(crop_img)])[0]
            if pred == 0:
                dx += 3
            else:
                cv2.putText(output_image, str(pred % 7 if (pred % 7) != 0 else 7), \
                            (int(x+dx+w/2), int(y+h+4)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 0, 0))
                if pred >= 8:
                    cv2.putText(output_image, ".", \
                            (int(x+dx+w/2+2), int(y+h-4)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 0, 0))
                if pred == 15:
                    cv2.putText(output_image, ".", \
                            (int(x+dx+w/2+2), int(y+h-7)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 0, 0))
                dx += 9
        y += line_space
    
    print("finish")
    cv2.imwrite(output_prefix + input_file, output_image)

if __name__ == '__main__':
    input_file = "bee.png"
    with gzip.open('OneVsOneModel.pgz', 'r') as f:
        OneVsOneModel = pickle.load(f)
        main(input_file, "OneVsOne_", OneVsOneModel)

    with gzip.open('OneVsRestModel.pgz', 'r') as f:
        OneVsRestModel = pickle.load(f)
        main(input_file, "OneVsRest_", OneVsRestModel)
    
    with gzip.open('KNeighborsModel.pgz', 'r') as f:
        KNeighborsModel = pickle.load(f)
        main(input_file, "KNeighbors_", KNeighborsModel)