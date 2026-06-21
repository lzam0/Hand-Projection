import cv2
import mediapipe as mp
import numpy as np
import threading

# Path to the MediaPipe hand landmark model file
MODEL_PATH = "hand_landmarker.task"

# Colours
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

# Pairs of landmark indices that form the skeleton of the hand
LANDMARK_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index finger
    (5,9),(9,10),(10,11),(11,12),   # middle finger
    (9,13),(13,14),(14,15),(15,16), # ring finger
    (13,17),(17,18),(18,19),(19,20),# pinky
    (0,17),                         # palm base
]

def draw_landmarks(frame, hand_landmarks, w, h):
    # Convert normalized landmark coordinates to pixel positions
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Draw lines between connected landmarks
    for start, end in LANDMARK_CONNECTIONS:
        cv2.line(frame, points[start], points[end], WHITE, 1)

    # Draw a circle at each landmark position
    for x, y in points:
        cv2.circle(frame, (x, y), 2, WHITE, -1)

def draw_index_connections(frame, left_landmarks, right_landmarks, w, h, buffers):
    # Get the index finger tip (landmark 8) pixel position for each hand
    left_index_tip  = (int(left_landmarks[8].x * w),  int(left_landmarks[8].y * h))
    right_index_tip = (int(right_landmarks[8].x * w), int(right_landmarks[8].y * h))
    left_thumb_tip  = (int(left_landmarks[4].x * w),  int(left_landmarks[4].y * h))
    right_thumb_tip = (int(right_landmarks[4].x * w), int(right_landmarks[4].y * h))

    # Draw a white line connecting the two index finger tips
    cv2.line(frame, left_index_tip, right_index_tip, WHITE, 1)
    # Draw a white line connecting the two thumb tips
    cv2.line(frame, left_thumb_tip, right_thumb_tip, WHITE, 1)
    # Draw a line from right index tip to right thumb tip
    cv2.line(frame, right_index_tip, right_thumb_tip, WHITE, 1)
    # Draw a line from left index tip to left thumb tip
    cv2.line(frame, left_index_tip, left_thumb_tip, WHITE, 1)

    risograph_distortion(frame, left_index_tip, right_index_tip, right_thumb_tip, left_thumb_tip, buffers)

# middle finger to index finger connection
def draw_middle_connections(frame, left_landmarks, right_landmarks, w, h, buffers):
    # Get the middle finger tip (landmark 12) pixel position for each hand
    left_middle_tip  = (int(left_landmarks[12].x * w),  int(left_landmarks[12].y * h))
    right_middle_tip = (int(right_landmarks[12].x * w), int(right_landmarks[12].y * h))
    left_index_tip  = (int(left_landmarks[8].x * w),  int(left_landmarks[8].y * h))
    right_index_tip = (int(right_landmarks[8].x * w), int(right_landmarks[8].y * h))

    # Draw a white line connecting the two middle finger tips
    cv2.line(frame, left_middle_tip, right_middle_tip, WHITE, 1)
    # Draw a white line connecting the two index finger tips
    cv2.line(frame, left_index_tip, right_index_tip, WHITE, 1)
    # Draw a line from right middle tip to right index tip
    cv2.line(frame, right_middle_tip, right_index_tip, WHITE, 1)
    # Draw a line from left middle tip to left index tip
    cv2.line(frame, left_middle_tip, left_index_tip, WHITE, 1)

    distort_region(frame, left_middle_tip, right_middle_tip, right_index_tip, left_index_tip, buffers)

def _roi_bounds(pt_a, pt_b, pt_c, pt_d, frame_w, frame_h):
    xs = [pt_a[0], pt_b[0], pt_c[0], pt_d[0]]
    ys = [pt_a[1], pt_b[1], pt_c[1], pt_d[1]]
    x0, y0 = max(min(xs), 0), max(min(ys), 0)
    x1, y1 = min(max(xs), frame_w), min(max(ys), frame_h)
    return x0, y0, x1, y1

# XRAY
def distort_region(frame, pt_a, pt_b, pt_c, pt_d, buffers):
    mask, gray, xray = buffers
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(pt_a, pt_b, pt_c, pt_d, fw, fh)

    # Build mask only within the bounding box of the four points
    roi_mask = mask[y0:y1, x0:x1]
    roi_mask[:] = 0
    pts = np.array([pt_a, pt_b, pt_c, pt_d], dtype=np.int32) - np.array([x0, y0])
    cv2.fillPoly(roi_mask, [pts], 255)

    # X-ray effect on the ROI slice only
    roi = frame[y0:y1, x0:x1]
    roi_gray = gray[y0:y1, x0:x1]
    roi_xray = xray[y0:y1, x0:x1]
    cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY, dst=roi_gray)
    cv2.bitwise_not(roi_gray, dst=roi_gray)
    roi_xray[:, :, 0] = roi_gray
    roi_xray[:, :, 1] = (roi_gray.astype(np.uint16) * 6 // 10).astype(np.uint8)
    roi_xray[:, :, 2] = 0
    cv2.copyTo(roi_xray, roi_mask, roi)

# Risograph Distortion Effect
def risograph_distortion(frame, pt_a, pt_b, pt_c, pt_d, buffers):
    mask, _, temp = buffers
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(pt_a, pt_b, pt_c, pt_d, fw, fh)

    # Build mask only within the bounding box of the four points
    roi_mask = mask[y0:y1, x0:x1]
    roi_mask[:] = 0
    pts = np.array([pt_a, pt_b, pt_c, pt_d], dtype=np.int32) - np.array([x0, y0])
    cv2.fillPoly(roi_mask, [pts], 255)

    # All operations on the ROI slice only
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    roi_temp = temp[y0:y1, x0:x1]

    # Channel misregistration within the ROI
    b, g, r = cv2.split(roi)
    shift = 5
    M_r = np.float32([[1, 0,  shift], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
    r = cv2.warpAffine(r, M_r, (rw, rh))
    b = cv2.warpAffine(b, M_b, (rw, rh))
    cv2.merge([b, g, r], dst=roi_temp)

    # Grain noise sized to the ROI
    noise = np.random.randint(0, 25, roi.shape, dtype=np.uint8)
    cv2.add(roi_temp, noise, dst=roi_temp)

    # Colour tint
    b_ch, g_ch, r_ch = cv2.split(roi_temp)
    r_ch = cv2.convertScaleAbs(r_ch, alpha=1.3, beta=15)
    b_ch = cv2.convertScaleAbs(b_ch, alpha=1.4, beta=10)
    g_ch = cv2.convertScaleAbs(g_ch, alpha=0.8, beta=-10)
    cv2.merge([b_ch, g_ch, r_ch], dst=roi_temp)

    cv2.copyTo(roi_temp, roi_mask, roi)

# Camera Class to handle webcam input and hand landmark detection
class Cameras:
    def __init__(self, frame_width=1920, frame_height=1080):
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
        stream.set(cv2.CAP_PROP_FPS, 60)

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

        h, w = self.frame_height, self.frame_width

        # Pre-allocate reusable buffers to avoid per-frame heap allocation
        buffers = (
            np.zeros((h, w), dtype=np.uint8),    # mask
            np.zeros((h, w), dtype=np.uint8),    # gray
            np.zeros((h, w, 3), dtype=np.uint8), # xray
        )

        # Shared state between main thread (render) and detection thread
        latest_frame = [None]
        last_result  = [None]
        frame_lock   = threading.Lock()
        result_lock  = threading.Lock()
        new_frame    = threading.Event()
        stop_event   = threading.Event()

        def detection_loop(landmarker):
            while not stop_event.is_set():
                # Wait until the main loop deposits a new frame (timeout avoids deadlock on exit)
                new_frame.wait(timeout=0.1)
                new_frame.clear()
                with frame_lock:
                    f = latest_frame[0]
                if f is None:
                    continue
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                with result_lock:
                    last_result[0] = result

        prev_tick = cv2.getTickCount()

        with HandLandmarker.create_from_options(options) as landmarker:
            det_thread = threading.Thread(target=detection_loop, args=(landmarker,), daemon=True)
            det_thread.start()

            while True:
                success, frame = stream.read()
                if not success:
                    break

                # Flip horizontally for a natural mirrored view
                mirrored_frame = cv2.flip(frame, 1)

                # Hand off a copy to the detection thread and signal it
                with frame_lock:
                    latest_frame[0] = mirrored_frame.copy()
                new_frame.set()

                # Render using whatever the detection thread last produced
                with result_lock:
                    result = last_result[0]

                if result and result.hand_landmarks:
                    # Build a dict of {handedness: landmarks} for easy lookup
                    # Mirrored frame flips left/right, so MediaPipe's labels are swapped
                    hands = {}
                    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                        label = handedness[0].category_name  # "Left" or "Right"
                        hands[label] = landmarks
                        # draw_landmarks(mirrored_frame, landmarks, w, h)

                    # Draw connecting lines and x-ray effect when both hands are visible
                    if "Left" in hands and "Right" in hands:
                        draw_index_connections(mirrored_frame, hands["Left"], hands["Right"], w, h, buffers)
                        draw_middle_connections(mirrored_frame, hands["Left"], hands["Right"], w, h, buffers)

                # Calculate and draw FPS in top-left corner
                now = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (now - prev_tick)
                prev_tick = now
                cv2.putText(mirrored_frame, f"FPS: {fps:.0f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

                cv2.imshow("Feed", mirrored_frame)

                # Exit on Esc key
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            stop_event.set()
            det_thread.join()

        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = Cameras()
    stream = cameras.readStream()
    cameras.runLoop(stream)
