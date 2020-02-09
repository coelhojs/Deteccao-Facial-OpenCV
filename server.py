#Comandos para iniciar o servidor:
#Linux:
#env FLASK_APP=server.py flask run
#Windows:
#set FLASK_APP=server.py python -m flask run

import io

import cv2
from flask import Flask, request, send_file
from PIL import Image

from FaceDetector import FaceDetector

app = Flask(__name__)


@app.route('/', methods=['POST'])
def face_detection_image():
    try:
        # Initializes the FaceDetector with its default values
        detector = FaceDetector()

        #Run the face detection
        detection_result = detector.detect_faces(request.data)
        
        #Reads the numpy array to an image and corrects the color scheme
        img = Image.fromarray(detection_result.astype('uint8'))
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))

        #Prepare the image to send back to the client
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        return send_file(
                     file_object,
                    #  attachment_filename='face_detection.png',
                     mimetype='image/jpg'
               )

    except Exception as error:
        raise error
