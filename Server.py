from flask import Flask, request, jsonify
import numpy as np
import cv2
import Digit_Localization
import json

app = Flask(__name__)
Digit_Localization.init_model()
convert = {'BIN': bin, 'HEX': hex}

@app.route('/api/process', methods=['POST'])
def process():
    r = request
    
    data = np.frombuffer(r.data, dtype=np.uint16)
    params = r.args.to_dict()
    
    count = int(params['count'])
    conversion = params['conversion']
    
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    result = Digit_Localization.predict(image, count)
    
    if(not conversion in convert):
        return json.dumps(result)
    
    for i in range(len(result)):
        result[i] = convert[conversion](result[i])
    
    return json.dumps(result)

app.run(host="0.0.0.0", port=5000)
