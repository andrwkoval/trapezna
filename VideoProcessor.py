import cv2
import numpy as np
import imutils

from utils import parse_datetime
from config import *


class VideoProcessor:
    def __init__(self, video_path="assets/peopleCounter.avi"):
        """
        Subtracting background from the video frames and find contours on the original frame
        which could be a person in the queue.
        :string video_path: path to the video to process
        """
        self.stream = cv2.VideoCapture(video_path)
        self.video_time = parse_datetime(video_path)
        # self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
        #     history=1000)
        self.min_area = min_contour_area_to_be_a_person
        self.max_area = max_contour_area_to_be_a_person

    @staticmethod
    def crop_interesting_region(frame):
        return frame[interesting_region_y_1:interesting_region_y_2,
               interesting_region_x_1:interesting_region_x_2]

    @staticmethod
    def leave_only_persons(frame, boxes):
        """
        Makes all areas without possible people black
        :ndarray frame: original frame from the video
        :list of (x,y,w,h) boxes: boxes suspected to have a person in
        :ndarray: frame_with_persons: frame with only suspected boxes to have person in
        """
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for x, y, w, h in boxes:
            mask[y:y + h, x:x + w] = 1

        frame_with_persons = cv2.bitwise_and(frame, frame, mask=mask)
        return frame_with_persons

    def process_frame(self, frame, prev, preview=False, crop=False):
        """
        Subtract background from the video. Leaves only contours which can be a person
        :ndarray frame: frame of the video
        :boolean preview:
        :ndarray: processed_frame: frame with only possible persons in the queue
        """
        global accum_image
        original_frame = frame.copy()
        # frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # gray = cv2.equalizeHist(gray)

        if prev is None:
            prev = gray
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)

        delta = cv2.absdiff(prev, gray)
        a = delta.copy()
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        cpp = thresh.copy()
        thresh = cv2.dilate(thresh, None, iterations=2)
        accum_image = cv2.add(accum_image, thresh)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        good_boxes = []
        # loop over the contours
        for c in contours:
            if self.min_area < cv2.contourArea(c) < self.max_area:
                (x, y, w, h) = cv2.boundingRect(c)
                good_boxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        processed_frame = self.leave_only_persons(original_frame, good_boxes)
        if preview:
            cv2.imshow("Original", original_frame)
            cv2.imshow('Only persons', processed_frame)
            cv2.imshow("Dilated", cpp)
            cv2.imshow("a", a)
            cv2.waitKey()
        return prev, accum_image

    def get_next_frame(self):
        grabbed, frame = self.stream.read()
        return grabbed, frame

    def show_video(self):
        grabbed, frame = self.get_next_frame()
        first_frame = frame.copy()
        prev = None
        while grabbed:
            prev, accum = self.process_frame(frame, prev, preview=True)
            grabbed, frame = self.get_next_frame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        color_image = cv2.applyColorMap(accum, cv2.COLORMAP_HOT)
        result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)
        cv2.imwrite("overlay.jpg", result_overlay)

if __name__ == "__main__":
    vidya = VideoProcessor()
    vidya.show_video()
