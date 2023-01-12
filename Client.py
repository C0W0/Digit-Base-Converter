import requests
import json
import numpy as np
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/api/process'

img = cv2.imread('test_images/test_2.png')
# encode image as jpeg
img_encoded: np.ndarray
_, img_encoded = cv2.imencode('.png', img)

response = requests.post(url=test_url, data=img_encoded.tobytes(), params={'count': 3, 'conversion': 'HEX'})
print(response.text)