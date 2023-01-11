from keras.models import load_model, Sequential
from imutils.contours import sort_contours
from LeNet_5 import getNum
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

def main():
    image = cv2.imread('test_images/test_2.png')
    
    grayed = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    
    # edge detection
    edged: np.ndarray = cv2.Canny(blurred, 150, 150)
    blurred = cv2.GaussianBlur(edged, (5, 5), 0)
    # plt.imshow(blurred, cmap='gray')
    # plt.show()
    countours = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countours = imutils.grab_contours(countours)
    countours = sort_contours(countours)[0]
    
    nums: list[tuple[np.ndarray, tuple]] = []
    
    cnt: np.ndarray
    for cnt in countours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        if(w >= 15 and h >= 25):
            rect = grayed[y:y+h, x:x+w]
            
            thresh: np.ndarray = cv2.threshold(rect, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            
            # plt.imshow(rect, cmap='gray')
            # plt.show()
            
            # calculate padding
            if(tW > tH):
                thresh = imutils.resize(thresh, width=28)
            else:
                thresh = imutils.resize(thresh, height=28)
                
            # the model uses 28*28, but we pad it to 32*32 and then scale it down
            (tH, tW) = thresh.shape
            padX = int(max((32-tW)/2.0, 0))
            padY = int(max((32-tH)/2.0, 0))
            
            padded: cv2.Mat = cv2.copyMakeBorder(thresh, top=padY, bottom=32-tH-padY, left=padX, right=32-tW-padX, 
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))
            
            padded = padded.astype('float32')/255.0
            
            # plt.imshow(padded, cmap='gray')
            # plt.show()
            
            nums.append((padded, (x, y, w, h)))
            
    model: Sequential = load_model('LeNet_5_Model')
    
    for (image, info) in nums:
        imageResized = image.reshape((1, 28, 28, 1))
        print(getNum(model(imageResized)[0]))
        plt.imshow(image, cmap='gray')
        plt.show()
    

if(__name__ == '__main__'):
    main()