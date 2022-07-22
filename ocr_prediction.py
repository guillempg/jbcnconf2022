import os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt
import Levenshtein

def levenshtein_of_single_result(single_ocr_result, real_plate):
    box = single_ocr_result[0]
    pred_text = single_ocr_result[1][0].replace(" ", "")
    confidence = single_ocr_result[1][1]
    levenshtein = Levenshtein.distance(real_plate, pred_text)
    return pred_text, confidence, box, levenshtein


def find_ocr_result_closest_to_real_license_plate(ocr_result, filename, img):
    fields = filename.split("_")
    suffix = ".png"
    real_plate = fields[1]
    readable_tmp = fields[2]
    real_is_readable = readable_tmp[:len(suffix)]

    best_levenshtein = 7
    best_pred = ""
    confidence_of_best_pred = 0
    box_of_best_pred = ""

    for line in ocr_result:
        pred_text, confidence, box, levenshtein = levenshtein_of_single_result(line, real_plate)
        if levenshtein < best_levenshtein:
            best_levenshtein = levenshtein
            best_pred = pred_text
            confidence_of_best_pred = confidence
            box_of_best_pred = box

    return Prediction(best_pred, confidence_of_best_pred, box_of_best_pred, best_levenshtein, real_plate,
                      real_is_readable, img)


class Prediction:
    def __init__(self, prediction, confidence, box, levenshtein_distance, real_value, real_is_readable, img):
        self.prediction = prediction
        self.confidence = confidence
        self.box = box
        self.levenshtein_distance = levenshtein_distance
        self.real_value = real_value
        self.real_is_readable = real_is_readable
        self.image = img


data_path = 'test/'
files = os.listdir(data_path)

img_arr = []
boxes = []
pred_texts = []
confidences = []
real_texts = []

# Number of images to display
num = 10000

# Appending the array of images to a list.
predictions = []

for fimg in files:
    if fimg.endswith('.png'):
        demo = np.asarray(Image.open(data_path + fimg))
        ocr = PaddleOCR(lang='en', show_log=False)
        result = ocr.ocr(demo)
        prediction = find_ocr_result_closest_to_real_license_plate(result, fimg, demo)

        if prediction.real_is_readable and prediction.box:
            print("Real plate:", prediction.real_value,
                  " bestPred:", prediction.prediction,
                  " confidence:", prediction.confidence,
                  " box:", prediction.box,
                  " levenshtein:", prediction.levenshtein_distance)
            predictions.append(prediction)
            print("("+str(len(predictions))+"/"+str(len(files))+")")
        if len(predictions) == num:
            break

## Plotting the histogram of levenshtein distances of predictions for readable images
distances = []
max_leven_dist = 0
for prediction in predictions:
    distances.append(prediction.levenshtein_distance)
    if prediction.levenshtein_distance > max_leven_dist:
        max_leven_dist = prediction.levenshtein_distance

print("Max leven dist:", max_leven_dist)

fig1 = plt.figure()
plt.title("Histogram of Levenshtein distance of predictions")
plt.xlabel("Levenshtein distance")
plt.ylabel("Count")
plt.hist(distances, bins=20)
plt.show()
fig1.savefig("Levenshtein_histogram_final.jpg")

## Plotting the images and predictions
for prediction, count in zip(predictions, range(0, len(predictions))):
    box = [prediction.box[0], prediction.box[1], prediction.box[2], prediction.box[3], prediction.box[0]]
    x, y = zip(*box)
    #print("Box:", prediction.box,
    #      " coord1:", prediction.box[0],
    #      " coord2:", prediction.box[1],
    #      " coord3:", prediction.box[2],
    #      " coord4:", prediction.box[3])

    title = "Truth:"+prediction.real_value+" Pred:"+prediction.prediction
    fig = plt.figure()
    plt.title(title, fontsize=25)
    plt.imshow(prediction.image)
    plt.plot(x, y, color="red", linewidth=9)
    plt.show()
    filename = "preds/predictions" + str(count) + ".jpg"
    fig.savefig(filename, dpi='figure')
    count += 1
    print("Count:", count)

