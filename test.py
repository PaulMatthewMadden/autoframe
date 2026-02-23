import cv2
import threading
import time
from collections import deque

# --- CONFIG ---
CAP_W, CAP_H = 1280, 720      
LIVE_W, LIVE_H = 640, 360     
TARGET_FPS = 30
BUFFER_SEC = 20
MAX_FRAMES = TARGET_FPS * BUFFER_SEC

class POCSystem:
    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        # Get actual hardware specs after initialization
        self.real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.real_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.buffer = deque(maxlen=MAX_FRAMES)
        self.live_proxy = None
        self.new_frame_available = False
        self.running = False
        self.lock = threading.Lock()

    def print_debug_info(self):
        print("\n" + "="*30)
        print("   CAMERA DEBUG SUMMARY")
        print("="*30)
        print(f"1. Camera Index:    {self.index}")
        print(f"2. Live Feed:       {LIVE_W}x{LIVE_H} @ {TARGET_FPS} FPS")
        print(f"   (Source: {self.real_w}x{self.real_h} @ {self.real_fps} FPS)")
        print(f"3. Playback Window: {self.real_w}x{self.real_h} @ 30 FPS")
        print("="*30 + "\n")

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.buffer.append(frame) 
                small = cv2.resize(frame, (LIVE_W, LIVE_H))
                with self.lock:
                    self.live_proxy = small
                    self.new_frame_available = True
            else:
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
    system = POCSystem(1) # Assuming index 1
    system.print_debug_info()
    system.start()
    
    print("Commands: [p] Playback | [q] Quit")

    while True:
        live_img = system.get_latest()
        
        if live_img is not None:
            cv2.imshow("Live Feed", live_img)

        key = cv2.waitKey(30) & 0xFF 
        
        if key == ord('q'):
            break
            
        elif key == ord('p'):
            if len(system.buffer) > 0:
                print(f"Starting Playback: {system.real_w}x{system.real_h} @ 30fps")
                # Iterate the deque directly for speed
                for i in range(max(0, len(system.buffer)-150), len(system.buffer)):
                    cv2.imshow("Playback", system.buffer[i])
                    # 33ms ensures a smooth 30fps playback
                    if cv2.waitKey(33) & 0xFF == ord('c'): 
                        print("Playback cancelled.")
                        break
                cv2.destroyWindow("Playback")
                print("Playback finished.")

    system.stop()
    cv2.destroyAllWindows()