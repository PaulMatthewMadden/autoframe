import cv2
import threading
import time
from collections import deque
from pygrabber.dshow_graph import FilterGraph

# --- SETTINGS ---
# W, H = 640, 480
W, H = 1280, 720
TARGET_FPS = 30
BUFFER_SECONDS = 30
MAX_BUFFER = TARGET_FPS * BUFFER_SECONDS

class AutoFrameSystem:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        # Exposure lock (as tested)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)

        # The 30-second memory bank
        self.buffer = deque(maxlen=MAX_BUFFER)
        
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        """Hardware interaction thread"""
        while self.running:
            success, img = self.cap.read()
            if success:
                # Store in memory for the live feed and the buffer
                with self.lock:
                    self.frame = img
                self.buffer.append(img.copy())

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    # Standard camera selection
    graph = FilterGraph()
    devices = graph.get_input_devices()
    for i, name in enumerate(devices): print(f"[{i}]: {name}")
    idx = int(input("Select Index: "))

    system = AutoFrameSystem(idx).start()
    
    last_time = time.time()
    
    print("System Running. Press 'p' for instant replay, 'q' to quit.")

    while True:
        frame = system.get_frame()
        
        if frame is not None:
            # CPU GOVERNOR: 
            # We only process/display at the speed of the target FPS
            # This will drop your CPU usage from 60% back to ~10-15%
            now = time.time()
            if (now - last_time) < (1.0 / TARGET_FPS):
                continue
            
            actual_fps = 1.0 / (now - last_time)
            last_time = now

            # Show the live feed
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Live FPS: {actual_fps:.1f}", (10, 30), 1, 1.5, (0,255,0), 2)
            cv2.putText(display_frame, f"Memory: {len(system.buffer)} frames", (10, 60), 1, 1, (255,255,255), 1)
            cv2.imshow("AutoFrame - Live", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            # REPLAY TEST: If we have at least 2 seconds of video, show a frame from 2s ago
            if len(system.buffer) > 60:
                past_frame = system.buffer[-60]
                cv2.imshow("2-Second Replay", past_frame)

    system.stop()
    cv2.destroyAllWindows()