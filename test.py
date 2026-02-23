import cv2
import threading
import time
from collections import deque

# --- CONFIG ---
# Try 1280x720, but if it's still choppy, the USB bus might be struggling.
CAP_W, CAP_H = 1280, 720      
LIVE_W, LIVE_H = 640, 360     
TARGET_FPS = 30
BUFFER_SEC = 20
MAX_FRAMES = TARGET_FPS * BUFFER_SEC

class POCSystem:
    def __init__(self, index):
        # CHANGE: Try CAP_MSMF instead of CAP_DSHOW for modern Windows performance
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        self.buffer = deque(maxlen=MAX_FRAMES)
        self.live_proxy = None
        self.new_frame_available = False # Flag to avoid redundant drawing
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        # Increase thread priority by making it a daemon
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
        return self

    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.buffer.append(frame) 
                # Optimization: Resize once here
                small = cv2.resize(frame, (LIVE_W, LIVE_H))
                with self.lock:
                    self.live_proxy = small
                    self.new_frame_available = True
            else:
                # If camera fails, don't spin the CPU
                time.sleep(0.01)

    def get_latest(self):
        with self.lock:
            if not self.new_frame_available:
                return None
            self.new_frame_available = False
            return self.live_proxy

    def stop(self):
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    system = POCSystem(1).start()
    
    print("Commands: [p] Playback | [q] Quit")

    while True:
        # 1. Grab the latest frame ONLY if a new one has arrived
        live_img = system.get_latest()
        
        if live_img is not None:
            # OPTIMIZATION: Stop using .copy(). 
            # Only draw text if you absolutely have to for the POC.
            cv2.imshow("Live Feed", live_img)

        # 2. Hardcoded wait (yields CPU to the capture thread)
        # We use 30ms to be slightly faster than the camera (ensures no lag build-up)
        key = cv2.waitKey(30) & 0xFF 
        
        if key == ord('q'):
            break
            
        elif key == ord('p'):
            if len(system.buffer) > 0:
                print("Playing Back...")
                # Avoid long list conversions; just iterate the deque directly
                for i in range(max(0, len(system.buffer)-150), len(system.buffer)):
                    cv2.imshow("Playback", system.buffer[i])
                    if cv2.waitKey(33) & 0xFF == ord('c'): break
                cv2.destroyWindow("Playback")

    system.stop()
    cv2.destroyAllWindows()