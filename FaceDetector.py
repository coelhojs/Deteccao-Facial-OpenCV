import cv2
import numpy as np


class FaceDetector:
    Confidence = ""
    Image = ""
    Model = ""

    # default constructor
    #The user can define the trained model and the prototxt, or use the default files
    def __init__(self, modelPath = "models/pre_trained_model/res10_300x300_ssd_iter_140000.caffemodel", prototxt = "models/pre_trained_model/deploy.prototxt.txt", confidence = 0.5):
        self.Confidence = confidence
        self.Model = cv2.dnn.readNetFromCaffe(prototxt, modelPath)
        
    # This method receives the image in base64 format, converts it to a numpy array
    # in the load_image method and executes the prediction, returning the image with 
    # the bounding box
    def detect_faces(self, imgData):
        image = self.load_image(imgData)

        (h, w) = image.shape[:2]
        
        imgBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        self.Model.setInput(imgBlob)
        detections = self.Model.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # The detections with a probability smaller than the defined value are discarded
            if confidence > self.Confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # Drawing the bounding box and the prediction score
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return image


    def load_image(self, imgData):
        nparr = np.fromstring(imgData, np.uint8)
        
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
