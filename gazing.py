import cv2
import dlib
import math
import numpy as np

class Pupil(object):
    # This class detects the iris of an eye and estimates the position of the pupil
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.detect_iris(eye_frame)
    @staticmethod
    def image_processing(eye_frame, threshold):
       #Performs operations on the eye frame to isolate the iris
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        return new_frame
    def detect_iris(self, eye_frame):
        # Detects the iris and estimates the position of the iris by calculating the centroid.
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)
        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass
class Eye(object):
    # This class creates a new frame to isolate the eye and initiates the pupil detection.
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
       # Returns the middle point (x,y) between two points
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        # Isolate an eye, to have a frame without other part of the face.
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        # Calculates a ratio that can indicate whether an eye is closed or not.
        # It's the division of the width of the eye, by its height.

        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
    # Detects and isolates the eye in a new frame, sends data to the calibration and initializes Pupil object.
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

    def is_left(self):
        """Returns true if the eye is looking to the left"""
        if self.pupil is not None:
            return self.pupil.x < self.center[0]

    def is_right(self):
        """Returns true if the eye is looking to the right"""
        if self.pupil is not None:
            return self.pupil.x > self.center[0]

    def is_center(self):
        """Returns true if the eye is looking to the center"""
        if self.pupil is not None:
            return self.is_right() is not True and self.is_left() is not True


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """
    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        # Returns the threshold value for the given eye.
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        # Returns the percentage of space that the iris takes up on the surface of the eye.
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        #Calculates the optimal threshold to binarize the frame for the given eye.
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)
        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        # Improves calibration by taking into consideration the
        given image.
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.gaze_away_count = 0  # Counter for gazing away
        self.center_gaze_count = 0  # Counter for gazing into the center

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        model_path = "/Users/vaishnavigogineni/Downloads/shape_predictor_68_face_landmarks.dat"
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        # Refreshes the frame and analyzes it.
        self.frame = frame
        self._analyze()

        # Check if the pupils are located and if they are gazing away or into the center
        if self.pupils_located:
            if self.eye_left.is_center() and self.eye_right.is_center():
                # Adjust this threshold as needed
                margin_threshold = 0.9
                if self.eye_left.pupil.x < margin_threshold and self.eye_right.pupil.x < margin_threshold:
                    self.center_gaze_count += 1
            else:
                # Check if either eye is gazing away
                if (self.eye_left.pupil.x >= 1.0 or self.eye_right.pupil.x >= 1.0) and (
                        self.eye_left.is_left() or self.eye_right.is_right()):
                    # Increment the gaze away count only if both horizontal and vertical gaze ratios indicate gazing away
                    if self.horizontal_ratio() >= 0.9 and self.vertical_ratio() >= 0.9:
                        self.gaze_away_count += 1
                # Check if either eye is gazing into the center
                elif self.eye_left.is_center() or self.eye_right.is_center():
                    self.center_gaze_count += 1

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

# Define the video URL
video_url = "https://credohirestorage1.blob.core.windows.net/devserver/661383568b1479873a2572d5_2052_av.webm"

def main():
    # Initialize GazeTracking
    gaze = GazeTracking()

    # Open the video capture
    cap = cv2.VideoCapture(video_url)
    max_faces_in_frame = 0  # Track maximum faces in a single frame

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Refresh the frame for gaze analysis
        gaze.refresh(frame)

        # Get the annotated frame with pupil positions
        annotated_frame = gaze.annotated_frame()

        # Count faces in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = gaze._face_detector(gray_frame)
        num_faces = len(faces)
        if num_faces > max_faces_in_frame:
            max_faces_in_frame = num_faces

        # Display the annotated frame
        cv2.imshow("Gaze Tracking", annotated_frame)
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    print("Number of times the person has gazed away:", gaze.gaze_away_count)
    print("Number of times the person has gazed into the center:", gaze.center_gaze_count)
    print("Maximum number of different faces detected in any frame throughout the video:", max_faces_in_frame)

if __name__ == "__main__":
    main()
