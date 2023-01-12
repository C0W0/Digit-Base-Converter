from keras.models import load_model, Sequential
from imutils.contours import sort_contours
from LeNet_5 import getNum
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

model: Sequential

def predict(image: cv2.Mat, numCount: int) -> list:
    grayed = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    
    # edge detection
    edged: np.ndarray = cv2.Canny(blurred, 150, 150)
    blurred = cv2.GaussianBlur(edged, (5, 5), 0)
    countours = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countours = imutils.grab_contours(countours)
    countours = sort_contours(countours)[0]
    
    nums: list[tuple[np.ndarray, list]] = []
    
    cnt: np.ndarray
    for cnt in countours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        if(w >= 5 and h >= 25):
            rect = grayed[y:y+h, x:x+w]
            
            thresh: np.ndarray = cv2.threshold(rect, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            
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
            
            nums.append((padded, [x+w/2, y+h/2]))
    
    clusteredData: list
    totalDistance = 0.0
    for i in range(max(10, numCount*2)):        
        (newData, newDistance) = k_means_cluster(numCount, nums)
        if(i == 0 or newDistance < totalDistance):
            clusteredData = newData
    
    detectionResult = [0]*numCount
    
    for i in range(numCount):
        clustered = clusteredData[i]
        image: np.ndarray
        for image in clustered:
            imageResized = image.reshape((1, 28, 28, 1))
            
            result = getNum(model(imageResized)[0])
            detectionResult[i] *= 10
            detectionResult[i] += result
    
    return detectionResult
    
    
def k_means_cluster(k: int, images: list[tuple[np.ndarray, list]]) -> tuple[np.ndarray, float]:
    points = []
    pointsData = []
    
    for (image, pt) in images:
        points.append(pt)
        pointsData.append(image)
    
    points = np.array(points)
    pointsData = np.array(pointsData)
    
    centres = points[np.random.choice(points.shape[0], k, replace=False)]
    
    shouldLoop = True
    while shouldLoop:
        distances = []
        
        for c in centres:
            distance = np.linalg.norm((points-c)**2, axis=1)
            distances.append(distance)
            
        distances = np.array(distances)
        labels: np.ndarray = np.argmin(distances, axis=0)
        
        newcentres = np.array([points[labels == i].mean(axis=0) for i in range(k)])
        
        shouldLoop = not np.allclose(centres, newcentres)
        centres = newcentres
    
    totalDistance = 0
    clusters = []
    for i in range(k):
        centre = centres[i]
        distance: np.ndarray = np.linalg.norm((points[labels == i]-centre)**2, axis=1)
        totalDistance += distance.sum()
        images = pointsData[labels == i]
        clusters.append(images)
        
    return (clusters, totalDistance)

def init_model():
    global model
    model = load_model('LeNet_5_Model')
    

if(__name__ == '__main__'):
    init_model()
    numCount = 3
    image = cv2.imread('test_images/test_2.png')
    print(predict(image, numCount))