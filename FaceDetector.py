import cv2
import numpy as np


def export_image_to_file(export_image_name, image_matrix):
    cv2.imwrite(f'{export_image_name}.jpg', image_matrix)


# current model path   : 'faceDetectionModel/deploy.prototxt.txt'
# current weights path : 'faceDetectionModel/opencv_face_detector.caffemodel'


class FaceDetector:

    def __init__(self, model_path, weights_path):
        self._net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    def detect_face_from_image(self, path_to_image, confidence_threshold=0.7):
        image = cv2.imread(path_to_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self._net.setInput(blob)
        detections = self._net.forward()

        num_detections = detections.shape[2]
        confidences = detections[0, 0, 0:num_detections, 2]
        detections = detections[0, 0, confidences > confidence_threshold, 0:7]

        if detections.size == 0:
            return

        # get the size of each box as the sum of two sides of the rectangle
        num_detections = detections.shape[0]
        sizes = ((detections[0:num_detections, 5] - detections[0:num_detections, 3])
                 + (detections[0:num_detections, 6] - detections[0:num_detections, 4]))

        box = detections[np.argmax(sizes), 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        maxY = image.shape[0]
        maxX = image.shape[1]

        
        image = image[max(startY - 50, 0):min(endY + 50, maxY), 
                      max(startX - 50, 0):min(endX + 50, maxX)]

        return image
