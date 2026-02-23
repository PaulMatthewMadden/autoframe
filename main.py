import cv2
import threading
import time
import psutil
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
        self.device_name = self._get_device_name(index) #
        
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

    def _get_device_name(self, index):
        # Quick probe to find the hardware name
        temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        name = temp_cap.getBackendName() if temp_cap.isOpened() else "Unknown Device"
        temp_cap.release()
        return name

    def print_debug_info(self):
        print("\n" + "="*50)
        print("            SYSTEM DEBUG REPORT")
        print("="*50)
        print(f"1. DEVICE INFO")
        print(f"   - Index:            {self.index}")
        print(f"   - Name:             {self.device_name}") #
        print(f"\n2. BUFFER METRICS")
        print(f"   - Resolution:       {self.real_w}x{self.real_h}") #
        print(f"   - Target FPS:       {TARGET_FPS}") #
        print(f"   - Capacity:         {BUFFER_SEC} seconds") #
        print(f"   - Max Frames:       {MAX_FRAMES}") #
        print(f"\n3. LIVE FEED METRICS")
        print(f"   - Proxy Res:        {LIVE_W}x{LIVE_H}") #
        print(f"   - Measured FPS:     {self.measured_fps:.2f}") #
        print("="*50 + "\n")

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
    
    # Allow stabilization before printing the report
    time.sleep(1.5)
    system.print_debug_info()
    
    ram_text = "RAM: Initializing..."
    last_sys_update = 0

    while True:
        if time.time() - last_sys_update > 1.0:
            ram = psutil.virtual_memory()
            used_gb = ram.used / (1024 ** 3)
            ram_text = f"RAM: {used_gb:.1f}GB ({ram.percent}%)" #
            last_sys_update = time.time()

        img = system.get_latest()
        if img is not None:
            # Re-adding the overlay for the live feed window
            cv2.putText(img, ram_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Live Feed", img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
            
        elif key in [ord('p'), ord('s')]:
            speed_mult = 1.0 if key == ord('p') else 0.5
            if len(system.buffer) > 10:
                clip = list(system.buffer)
                cv2.namedWindow("Playback Window", cv2.WINDOW_NORMAL)
                
                start_wall_time = time.perf_counter()
                start_frame_time = clip[0][0]
                
                for ts, frame in clip:
                    target_elapsed = (ts - start_frame_time) / speed_mult
                    while (time.perf_counter() - start_wall_time) < target_elapsed:
                        if cv2.waitKey(1) & 0xFF == ord('c'): break
                    
                    cv2.imshow("Playback Window", frame)
                    if cv2.waitKey(1) & 0xFF == ord('c'): break
                
                cv2.destroyWindow("Playback Window")

    system.stop()
    cv2.destroyAllWindows()