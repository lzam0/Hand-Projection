import mediapipe
import cv2

class Cameras:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height

    @staticmethod
    def readStream():
        stream = cv2.VideoCapture(0)
        return stream

    def runLoop(self, stream):
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        while True:
            _, frame = stream.read()
            # Mirror the frame horizontally
            mirrored_frame = cv2.flip(frame, 1)
            
            # Display Feed
            cv2.imshow("Feed", mirrored_frame)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                stream.release()
                cv2.destroyAllWindows()
                break

# Main Loop
if __name__ == "__main__":
    cameras = Cameras()
    stream = cameras.readStream()
    cameras.runLoop(stream)