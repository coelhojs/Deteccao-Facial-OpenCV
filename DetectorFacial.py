# USAGE
# python detect_faces.py --image 200px-Carl_Sagan_Planetary_Society.jpeg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
import cv2
import numpy as np


class DetectorFacial:
    Confidence = ""
    Image = ""
    Model = ""

    # default constructor
    def __init__(self, modelPath = "models/pre_trained_model/res10_300x300_ssd_iter_140000.caffemodel", prototxt = "models/pre_trained_model/deploy.prototxt.txt", confidence = 0.5):
        self.Confidence = confidence
        self.Model = cv2.dnn.readNetFromCaffe(prototxt, modelPath)
        
    def detect_faces(self, imgData):
        image = self.loadImage(imgData)

        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        
        imgBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.Model.setInput(imgBlob)
        detections = self.Model.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.Confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # draw the bounding box of the face along with the associated
                # probability
                # text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                # cv2.putText(image, text, (startX, y),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output image
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
        return image


    def loadImage(self, imgData):
        # convert string of image data to uint8
        nparr = np.fromstring(imgData, np.uint8)
        # decode image
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
