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
        
        self.real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.buffer = deque(maxlen=MAX_FRAMES)
        self.live_proxy = None
        self.new_frame_available = False
        self.running = False
        self.lock = threading.Lock()
        
        self.measured_fps = 0.0
        self.frame_intervals = deque(maxlen=30)

    def print_debug_info(self):
        print("\n" + "="*40)
        print("        SYSTEM DEBUG REPORT")
        print("="*40)
        print(f"1. Device:             Index {self.index}")
        print(f"2. Capture Resolution: {self.real_w}x{self.real_h}")
        print(f"3. Buffer Capacity:    {BUFFER_SEC}s / {MAX_FRAMES} frames")
        print(f"4. RAM Usage (Est):    ~{(self.real_w*self.real_h*3*MAX_FRAMES)/1e9:.2f} GB")
        print("="*40 + "\n")

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        last_t = time.perf_counter()
        while self.running:
            success, frame = self.cap.read()
            if success:
                now = time.perf_counter()
                self.buffer.append((now, frame))
                
                self.frame_intervals.append(now - last_t)
                last_t = now
                if len(self.frame_intervals) > 0:
                    self.measured_fps = 1.0 / (sum(self.frame_intervals)/len(self.frame_intervals))

                small = cv2.resize(frame, (LIVE_W, LIVE_H))
                with self.lock:
                    self.live_proxy = small
                    self.new_frame_available = True
            else:
                time.sleep(0.001)

    def get_latest(self):
        with self.lock:
            if not self.new_frame_available: return None
            self.new_frame_available = False
            return self.live_proxy

    def stop(self):
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    system = POCSystem(1).start()
    time.sleep(1) 
    system.print_debug_info()
    
    print("Commands:")
    print(" [p] Playback (1x) | [s] Slow-Mo (0.5x)")
    print(" [q] Quit | [c] Cancel playback")

    while True:
        img = system.get_latest()
        if img is not None:
            cv2.imshow("Live Feed", img)

        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key in [ord('p'), ord('s')]:
            speed_mult = 1.0 if key == ord('p') else 0.5
            
            if len(system.buffer) > 10:
                clip = list(system.buffer)
                
                # --- KEY CHANGE: CREATE RESIZABLE WINDOW ---
                cv2.namedWindow("Playback Window", cv2.WINDOW_NORMAL)
                # You can also set a default size if you want it to start large
                # cv2.resizeWindow("Playback Window", 1280, 720) 

                start_wall_time = time.perf_counter()
                start_frame_time = clip[0][0]
                
                print(f"Playing at {speed_mult}x. You can now maximize this window!")

                for ts, frame in clip:
                    target_elapsed = (ts - start_frame_time) / speed_mult
                    
                    while (time.perf_counter() - start_wall_time) < target_elapsed:
                        # Polling waitKey(1) keeps the window UI responsive while waiting
                        if cv2.waitKey(1) & 0xFF == ord('c'): break
                    
                    cv2.imshow("Playback Window", frame)
                    
                    # Check for cancel key during frame display
                    if cv2.waitKey(1) & 0xFF == ord('c'): 
                        break
                
                cv2.destroyWindow("Playback Window")
                print(">>> Playback Complete.")

    system.stop()
    cv2.destroyAllWindows()