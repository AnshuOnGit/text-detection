import sys

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


# Load Image
def load_image(image_path) -> cv2.typing.MatLike | None:
    image = cv2.imread(image_path)
    if image is None:
        return None
    return image


def detect_text_using_EAST(image: cv2.typing.MatLike, inputSize: tuple[int, int]) -> [cv2.typing.MatLike]:
    # Set the input size for the EAST model
    # East model for text-detection
    textDetectorEAST = cv2.dnn_TextDetectionModel_EAST("./resources/frozen_east_text_detection.pb")
    # Set the Detection Confidence Threshold and NMS threshold
    conf_thresh = 0.8
    nms_thresh = 0.4
    textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
    textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)
    boxesEAST, confsEAST = textDetectorEAST.detect(image)
    return boxesEAST


def text_detected_image(image: cv2.typing.MatLike, points: [cv2.typing.MatLike]) -> cv2.typing.MatLike:
    imCopy = image.copy()
    cv2.polylines(imCopy, points, isClosed=True, color=(255, 0, 255), thickness=4)
    return imCopy


def show_image(image_title: str, image: cv2.typing.MatLike) -> None:
    cv2.imshow(image_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_matplotlib(image_title: str, image1: cv2.typing.MatLike, image2: cv2.typing.MatLike) -> None:
    img = cv2.hconcat([image1, image2])
    plt.figure(figsize=(20, 10))
    plt.title(image_title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    image_path = "./resources/visuals/dutch_signboard.jpg"
    image = load_image(image_path)
    #show_image_matplotlib("Original Image", image)
    if image is None:
        print("Image not found")
        sys.exit(0)
    points = detect_text_using_EAST(image, (320, 320))
    image_text_detected = text_detected_image(image, points)
    show_image_matplotlib("Text Detection", image, image_text_detected)
