import math
import os

import cv2
import imutils
import numpy as np

# from openalpr import Alpr
# import tensorflow.compat.v1 as tf


class BlurTool:

    def __init__(self):
        np.random.seed(42)
        self.prototxt_path = "prototxt.txt"
        self.model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.model = cv2.dnn.readNetFromCaffe(self.prototxt_path,
                                              self.model_path)
        self.net = cv2.dnn.readNetFromDarknet("darknet/cfg/yolov3.cfg",
                                              "darknet/yolov3.weights")

    def draw_face(self, frame, start_y, end_y, start_x, end_x, width, height,
                  confidence):
        try:
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.GaussianBlur(face, (width, height), 0)
            # cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 255), 4)
            # cv2.putText(frame, str(confidence), (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            frame[start_y:end_y, start_x:end_x] = face
        except:
            # print(start_y, end_y, start_x, end_x)
            pass
        return frame

    def nearby_face(self, midpoint, confidence, size):
        # [midpoint, confidence, size, flag]
        for i in self.faces:
            if i[1] < 0.25:
                self.faces.remove(i)
                continue

            if (math.sqrt(
                    math.pow(midpoint[0] - i[0][0], 2) +
                    math.pow(midpoint[1] - i[0][1], 2)) < 20
                    and abs(i[2] - size) < i[2] * 0.05):
                i[3] = 1
                i[1] = i[1] + 0.2 * confidence
                i[0] = (
                    midpoint[0] * 0.8 + i[0][0] * 0.2,
                    midpoint[1] * 0.8 + i[0][1] * 0.2,
                )
                i[2] = 0.5 * size + i[2] * 0.5
                if i[1] > 0.25:
                    return True

        if confidence > 0.35:
            new_i = [midpoint, confidence, size, 1]
            self.faces.append(new_i)
            return True

        return False

    """ 
    Checks if contour is near to license plate in previous frames and within size requirements.
    When new license plate is detected, coordinate is stored in self.plates with counter of 10.
    If contour is found near license plate but is not detected as a plate itself, return True
    and counter of plate will decrement by 1. If contour is found near license plate and 
    is detected as plate itself, return True and counter of plate will increment by 2. 
    Coordinates removed when counter reaches 0. 
    """

    def nearby_plate(self, coords, ar, angle, size, img_size):
        # [(actual_x, actual_y), counter]
        for i in self.plates:
            if i[1] <= 0:
                self.plates.remove(i)
                continue
            if (math.sqrt(
                    math.pow(coords[0] - i[0][0], 2) +
                    math.pow(coords[1] - i[0][1], 2)) <
                    10) and size < img_size * 0.08:
                i[0] = (coords[0], coords[1])
                i[1] = (i[1] - 1 if
                        (ar < 2.5 or ar > 7 or
                         (angle > 20 and angle < 70)) else i[1] + 2)
                return True
        if (ar > 2.5 and ar < 7 and (angle < 20 or angle > 70)
                and size < img_size * 0.08):
            new_i = [(coords[0], coords[1]), 10]
            self.plates.append(new_i)
            return True
        return False

    """ 
    Blurs out license plate given the coordinates.
    """

    def draw_plates(self, frame, x, y, w, h):
        try:
            start_x, end_x, start_y, end_y = (
                x * 0.96,
                (x + w) * 1.04,
                y * 0.99,
                (y + h) * 1.01,
            )
            licensePlate = frame[int(start_y):int(end_y),
                                 int(start_x):int(end_x)]
            kernel_width = int(w // 3) | 1
            kernel_height = int(h // 3) | 1
            licensePlate = cv2.blur(licensePlate,
                                    (kernel_width, kernel_height), 20)
            frame[int(start_y):int(end_y),
                  int(start_x):int(end_x)] = licensePlate
        except Exception as e:
            # print(x, y, w, h)
            pass
        return frame

    def debug(self, image):
        cv2.imshow("debug_img", image)
        cv2.waitKey(10000)
        cv2.destroyWindow("debug_img")

    def blur_face(self, frame):
        h, w = frame.shape[:2]
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1
        blob = cv2.dnn.blobFromImage(frame, 2, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        output = np.squeeze(self.model.forward())
        for i in range(0, output.shape[0]):  #
            confidence = output[i, 2]
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(int)
            midpoint = ((start_x + end_x) // 2, (start_y + end_y) // 2)
            size = (end_x - start_x) * (end_y - start_y)
            # if self.nearby_face(midpoint, confidence, size):
            if confidence > 0.2:
                frame = self.draw_face(
                    frame,
                    start_y,
                    end_y,
                    start_x,
                    end_x,
                    kernel_width,
                    kernel_height,
                    confidence,
                )

        return frame

    """ 
    Given the frame of a vehicle, finds the license plate using edge detection (4 edges) or morphological
    whitehat transformation. 
    Performs decently but there is some flickering and false positives.
    """

    def blur_plate(self, vehicle, vehicle_x, vehicle_y):

        img_size = vehicle.shape[0] * vehicle.shape[1]
        gray = cv2.cvtColor(vehicle, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.equalizeHist(gray)
        edged = cv2.Canny(gray, 30, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(c)
            vehicle = self.draw_plates(vehicle, x, y, w, h)
            return vehicle

        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)

        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        grady = cv2.Sobel(whitehat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-2)
        grady = cv2.morphologyEx(grady, cv2.MORPH_CLOSE, rectKern)
        gradX = np.absolute(grady)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        threshy = cv2.threshold(gradX, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(threshy, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        # cv2.drawContours(vehicle, cnts, -1, (0, 255, 0), 3)
        # cv2.drawContours(vehicle, cnts, 0, (0, 255, 0), 3)
        licensePlate = None

        for c in cnts:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            (x, y), (w, h), angle = rect

            ar = max(w / h, h / w)
            size = w * h
            actual_x, actual_y = x + vehicle_x, y + vehicle_y

            if (ar > 2.5 and ar < 7 and (angle < 20 or angle > 70)
                    and size < img_size * 0.08):
                (x, y, w, h) = cv2.boundingRect(c)

                if len(self.plates_frame):
                    temp = np.array([i[-1][:2] for i in self.plates_frame])
                    current = np.array([x, y])
                    res = np.sum((temp - current)**2, axis=1)
                    i = np.argmin(res)
                    if res[i] <= 1000:
                        self.plates_frame[i].append([x, y, w, h])
                        self.plates_num[i] = self.frame_num
                    else:
                        self.plates_frame.append([[x, y, w, h]])
                        self.plates_num.append(self.frame_num)
                else:
                    self.plates_frame.append([[x, y, w, h]])
                    self.plates_num.append(self.frame_num)

        a = []
        for i in range(len(self.plates_frame)):
            if (len(self.plates_frame[i]) >=
                    300) or (self.frame_num - self.plates_num[i] >= 300):
                a.append(self.plates_frame[i][1:])
            else:
                a.append(self.plates_frame[i])
        self.plates_frame = a
        print("list: ", self.plates_frame)

        # [[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
        for i in self.plates_frame:
            x, y, w, h = np.mean(i, axis=0)
            vehicle = self.draw_plates(vehicle, x, y, w, h)

        return vehicle

    """ 
    Detects vehicles using yolov3. 
    Good accuracy.
    """

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame,
                                     1 / 255, (320, 320), [0, 0, 0],
                                     1,
                                     crop=False)
        self.net.setInput(blob)
        layersNames = self.net.getLayerNames()
        outputNames = [(layersNames[i - 1])
                       for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(outputNames)
        vehicles_index = [2, 3, 5, 7]
        people_index = 0
        vehicle_boxes = []
        vehicle_confidence_scores = []
        people_boxes = []
        people_confidence_scores = []
        vehicles = []
        people = []
        height, width = frame.shape[:2]
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if classId in vehicles_index:
                    if confidence > 0.9:
                        w, h = int(det[2] * width), int(det[3] * height)
                        x, y = int((det[0] * width) -
                                   w / 2), int((det[1] * height) - h / 2)
                        vehicle_boxes.append([x, y, w, h])
                        vehicle_confidence_scores.append(float(confidence))
                        # cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 10)
                if classId == people_index:
                    if confidence > 0.5:
                        w, h = int(det[2] * width), int(det[3] * height)
                        x, y = int((det[0] * width) -
                                   w / 2), int((det[1] * height) - h / 2)
                        people_boxes.append([x, y, w, h])
                        people_confidence_scores.append(float(confidence))

        vehicle_indices = cv2.dnn.NMSBoxes(vehicle_boxes,
                                           vehicle_confidence_scores, 0.9, 0.2)
        people_indices = cv2.dnn.NMSBoxes(people_boxes,
                                          people_confidence_scores, 0.5, 0.2)

        for i in vehicle_indices:
            x, y, w, h = (
                int(vehicle_boxes[i][0]),
                int(vehicle_boxes[i][1]),
                int(vehicle_boxes[i][2]),
                int(vehicle_boxes[i][3]),
            )
            vehicle = frame[y:y + h, x:x + w]
            vehicles.append((vehicle, (x, y, w, h)))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 10)
        for i in people_indices:
            x, y, w, h = (
                int(people_boxes[i][0]),
                int(people_boxes[i][1]),
                int(people_boxes[i][2]),
                int(people_boxes[i][3]),
            )
            person = frame[y:y + h, x:x + w]
            people.append((person, (x, y, w, h)))
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 10)

        return vehicles, people

    """ 
    Runs detect_vehicles to find vehicle frames, and runs blur_plate on each vehicle frame
    """

    def blur_all(self, frame):
        vehicles, people = self.detect_objects(frame)
        frame_size = frame.shape[0] * frame.shape[1]
        for v in vehicles:
            try:
                x, y, w, h = v[1]
                if w * h < frame_size * 0.03:
                    continue
                vehicle = self.blur_plate(v[0], x, y)
                frame[y:y + h, x:x + w] = vehicle
            except Exception as e:
                # print(x, y, w, h)
                pass
            # x, y, w, h = v[1]
            # if w * h < frame_size * 0.03:
            #     continue
            # vehicle = self.blur_plate(v[0], x, y)
            # frame[y:y + h, x:x + w] = vehicle

        for p in people:
            try:
                x, y, w, h = p[1]
                person = self.blur_face(p[0])
                frame[y:y + h, x:x + w] = person
            except Exception as e:
                # print(x, y, w, h)
                pass

        return frame

    """ 
    Slices video into frames and outputs new video.
    """

    def process_video(self, video):
        self.faces = []
        self.plates = []
        cap = cv2.VideoCapture(video)
        print(cap.isOpened())
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        result = cv2.VideoWriter(
            "output4.avi",
            cv2.VideoWriter_fourcc(*"MP4V"),
            3,
            (frame_width, frame_height),
        )

        self.frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = self.blur_plate(frame)
            frame = self.blur_all(frame)
            self.frame_num += 1
            # frame = self.blur_face(frame)
            result.write(frame)
            # cv2.line(frame, (10, 10), (10, 50), (0, 255, 0), 9)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        result.release()


bt = BlurTool()
bt.plates = []
bt.faces = []
bt.plates_frame = []
bt.plates_num = []
bt.process_video("dashcam.mp4")
while True:
    cv2.imshow("frame", bt.blur_all(cv2.imread("test_person.png")))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ls = []
# for f in os.listdir("/Users/student/Desktop/Datasets/hdd_data/camera"):
#     if f == '.DS_Store':
#         continue
#     set = [f, len(os.listdir("/Users/student/Desktop/Datasets/hdd_data/camera/%s" % f))]
#     ls.append(set)

# for i in ls:
#     print("%s" % i[1])

# print(len(ls))
# alpr = Alpr('sg', 'openalpr.conf', 'openalpr/runtime_data')
# if not alpr.is_loaded():
#     print('Error loading OpenALPR')
#     sys.exit(1)

# img = cv2.resize(
#     frame, (int(frame.shape[1]) * 2, int(frame.shape[0]) * 2))

# analyzed_file = alpr.recognize_ndarray(img)

# if analyzed_file['results']:
#     for result in analyzed_file['results']:
#         x1 = result['coordinates'][0]['x'] // 2
#         y1 = result['coordinates'][0]['y'] // 2
#         x2 = result['coordinates'][2]['x'] // 2
#         y2 = result['coordinates'][2]['y'] // 2

#         plate = frame[y1:y2, x1:x2]
#         try:
#             blurred_plate = cv2.GaussianBlur(plate, (21, 21), 4, 4, 0)
#             frame[y1:y2, x1:x2] = blurred_plate
#             cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 255), 10)
#         except:
#             print(x1, x2, y1, y2)

# return frame
