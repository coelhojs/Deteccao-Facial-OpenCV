#comando para iniciar o servidor:
#env FLASK_APP=server.py flask run
import io

import cv2
from flask import Flask, request, send_file
from PIL import Image

from DetectorFacial import DetectorFacial

app = Flask(__name__)


@app.route('/', methods=['POST'])
def vera_species_classify():
    try:
        
        detector = DetectorFacial()

        detection_result = detector.detect_faces(request.data)
        img = Image.fromarray(detection_result.astype('uint8'))

        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))

        # create file-object in memory
        file_object = io.BytesIO()
        
        # write PNG in file-object
        img.save(file_object, 'PNG')

        # move to beginning of file so `send_file()` it will read from start    
        file_object.seek(0)

        # return jsonpickle.encode(detector.detect_faces(request.data))
        return send_file(
                     file_object,
                     attachment_filename='face_detection.png',
                     mimetype='image/jpg'
               )

    except Exception as error:
        raise error
