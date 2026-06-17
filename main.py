import cv2
import mediapipe as mp


# Path to the MediaPipe hand landmark model file
MODEL_PATH = "hand_landmarker.task"

# Colours
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
# Pairs of landmark indices that form the skeleton of the hand
LANDMARK_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index finger
    (5,9),(9,10),(10,11),(11,12),  # middle finger
    (9,13),(13,14),(14,15),(15,16),# ring finger
    (13,17),(17,18),(18,19),(19,20),# pinky
    (0,17),                        # palm base
]

def draw_landmarks(frame, hand_landmarks, w, h):
    # Convert normalized landmark coordinates to pixel positions
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Draw lines between connected landmarks
    for start, end in LANDMARK_CONNECTIONS:
        cv2.line(frame, points[start], points[end], GREEN, 1)

    # Draw a circle at each landmark position
    for x, y in points:
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

def draw_connecting_lines(frame, left_landmarks, right_landmarks, w, h):
    # Get the index finger tip (landmark 8) pixel position for each hand
    left_index_tip = (int(left_landmarks[8].x * w), int(left_landmarks[8].y * h))
    right_index_tip = (int(right_landmarks[8].x * w), int(right_landmarks[8].y * h))

    left_thumb_tip = (int(left_landmarks[4].x * w), int(left_landmarks[4].y * h))
    right_thumb_tip = (int(right_landmarks[4].x * w), int(right_landmarks[4].y * h))

    # Draw a blue line connecting the two index finger tips
    cv2.line(frame, left_index_tip, right_index_tip, BLUE, 1)
    
    # Draw a blue line connecting the two thumb tips
    cv2.line(frame, left_thumb_tip, right_thumb_tip, BLUE, 1)
    
    # Draw a line from right index tip to right thumb tip
    cv2.line(frame, right_index_tip, right_thumb_tip, BLUE, 1)

    # Draw a line from left index tip to left thumb tip
    cv2.line(frame, left_index_tip, left_thumb_tip, BLUE, 1)

'''
Essnetially I will be applying a mask to the region between the two hands
and then applying a distortion effect to that region.

The effect will be an anime style video effect - use hugging face models to generate the effect. 

The distortion will be applied to the region between the two hands, and the rest of the frame will remain unchanged. 
The effect will be applied in real-time as the hands move.
'''
def distort_region(frame):
    pass


# Camera Class to handle webcam input and hand landmark detection
class Cameras:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height

    @staticmethod
    def readStream():
        # Open the default webcam
        return cv2.VideoCapture(0)

    def runLoop(self, stream):
        # Configure webcam resolution
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Alias MediaPipe Tasks classes for brevity
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Set up the hand landmarker with the specified options
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = stream.read()
                if not success:
                    break

                # Flip horizontally for a natural mirrored view
                mirrored_frame = cv2.flip(frame, 1)
                h, w = mirrored_frame.shape[:2]

                # MediaPipe requires RGB input
                rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    # Build a dict of {handedness: landmarks} for easy lookup
                    # Mirrored frame flips left/right, so MediaPipe's labels are swapped
                    hands = {}
                    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                        label = handedness[0].category_name  # "Left" or "Right"
                        hands[label] = landmarks
                        draw_landmarks(mirrored_frame, landmarks, w, h)

                    # Draw blue line between both index tips when both hands are visible
                    if "Left" in hands and "Right" in hands:
                        draw_connecting_lines(mirrored_frame, hands["Left"], hands["Right"], w, h)

                cv2.imshow("Feed", mirrored_frame)

                # Exit on Esc key
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = Cameras()
    stream = cameras.readStream()
    cameras.runLoop(stream)
