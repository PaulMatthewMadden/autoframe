import cv2
import threading
import time
import psutil
from collections import deque
import numpy as np

# --- CONFIG ---
CAP_W, CAP_H = 1280, 720      
LIVE_W, LIVE_H = 640, 360     
TARGET_FPS = 30
BUFFER_SEC = 20  
MAX_FRAMES = TARGET_FPS * BUFFER_SEC

# Motion Sensitivity
START_SENSITIVITY = 500  
END_SENSITIVITY = 700    
MOTION_INTERVAL = 0.2    
POST_CAPTURE_WAIT = 10.0 

# Playback Padding
PRE_ACTION_PAD = 2.0 
POST_ACTION_PAD = 2.0 

class ROI:
    def __init__(self, x, y, w, h, color, name, sensitivity):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.color = color
        self.name = name
        self.sensitivity = sensitivity
        self.dragging = False
        self.resizing = False
        self.last_roi_frame = None

    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
        cv2.putText(img, f"{self.name}", (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        cv2.rectangle(img, (self.x + self.w - 10, self.y + self.h - 10), (self.x + self.w, self.y + self.h), self.color, -1)

    def detect_motion(self, current_roi_gray):
        if self.last_roi_frame is None:
            self.last_roi_frame = current_roi_gray
            return False
        diff = cv2.absdiff(self.last_roi_frame, current_roi_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = cv2.countNonZero(thresh)
        self.last_roi_frame = current_roi_gray
        return motion_score > self.sensitivity

class POCSystem:
    def __init__(self, index, r_start, r_end):
        self.index = index
        self.roi_start = r_start
        self.roi_end = r_end
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        self.buffer = deque(maxlen=MAX_FRAMES)
        self.live_proxy = None
        self.new_frame_available = False
        self.running = False
        self.lock = threading.Lock()
        
        self.status_message = "Waiting..."
        self.cooldown_message = ""
        self.is_triggered = False 
        self.cooldown_until = 0 
        
        self.last_start_time = None
        self.playback_clip = None # Stores the current looping clip

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._playback_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        last_motion_check = 0
        while self.running:
            success, frame = self.cap.read()
            if success:
                now = time.perf_counter()
                small = cv2.resize(frame, (LIVE_W, LIVE_H))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                start_flag, end_flag = False, False
                
                if now > self.cooldown_until:
                    self.cooldown_message = ""
                    if now - last_motion_check >= MOTION_INTERVAL:
                        sx, sy, sw, sh = self.roi_start.x, self.roi_start.y, self.roi_start.w, self.roi_start.h
                        if self.roi_start.detect_motion(gray[max(0,sy):min(LIVE_H,sy+sh), max(0,sx):min(LIVE_W,sx+sw)]):
                            start_flag = True
                            self.is_triggered = True
                            self.last_start_time = now # Track for padding
                            self.status_message = "Action Triggered"

                        if self.is_triggered:
                            ex, ey, ew, eh = self.roi_end.x, self.roi_end.y, self.roi_end.w, self.roi_end.h
                            if self.roi_end.detect_motion(gray[max(0,ey):min(LIVE_H,ey+eh), max(0,ex):min(LIVE_W,ex+ew)]):
                                end_flag = True
                                self.is_triggered = False
                                self.status_message = "Playback Started"
                                self.cooldown_until = now + POST_CAPTURE_WAIT
                                
                                # Extract clip with padding
                                self._create_playback_clip(self.last_start_time - PRE_ACTION_PAD, now + POST_ACTION_PAD)
                                
                                # Clear all flags in buffer
                                with self.lock:
                                    temp_list = list(self.buffer)
                                    self.buffer.clear()
                                    for t, f, _, _ in temp_list:
                                        self.buffer.append((t, f, False, False))
                        last_motion_check = now
                else:
                    self.cooldown_message = f"Cooldown: {int(self.cooldown_until - now)}s"

                self.buffer.append((now, frame, start_flag, end_flag))
                with self.lock:
                    self.live_proxy = small
                    self.new_frame_available = True
            else:
                time.sleep(0.001)

    def _create_playback_clip(self, start_t, end_t):
        """Copies relevant frames from the rolling buffer to a separate list"""
        with self.lock:
            current_buffer = list(self.buffer)
            self.playback_clip = [f for f in current_buffer if start_t <= f[0] <= end_t]

    def _playback_loop(self):
        """Dedicated thread to loop the last captured clip"""
        while self.running:
            if self.playback_clip and len(self.playback_clip) > 0:
                cv2.namedWindow("Auto Playback", cv2.WINDOW_NORMAL)
                clip = list(self.playback_clip)
                start_wall, start_frame = time.perf_counter(), clip[0][0]
                
                for ts, frame, _, _ in clip:
                    # Sync playback speed to real-time
                    while (time.perf_counter() - start_wall) < (ts - start_frame):
                        time.sleep(0.005)
                    cv2.imshow("Auto Playback", frame)
                    if cv2.waitKey(1) & 0xFF == ord('c'): break
            else:
                time.sleep(0.1)

    def get_latest(self):
        with self.lock:
            if not self.new_frame_available: return None
            self.new_frame_available = False
            return self.live_proxy

    def stop(self):
        self.running = False
        self.cap.release()

# Global UI Logic
roi_start = ROI(50, 50, 100, 100, (0, 255, 0), "START", START_SENSITIVITY)
roi_end = ROI(200, 50, 100, 100, (0, 0, 255), "END", END_SENSITIVITY)
active_roi = None

def mouse_event(event, x, y, flags, param):
    global active_roi
    rois = [roi_start, roi_end]
    if event == cv2.EVENT_LBUTTONDOWN:
        for r in rois:
            if r.x + r.w - 10 <= x <= r.x + r.w and r.y + r.h - 10 <= y <= r.y + r.h:
                r.resizing = True; active_roi = r; break
            elif r.x <= x <= r.x + r.w and r.y <= y <= r.y + r.h:
                r.dragging = True; r.ox, r.oy = x - r.x, y - r.y; active_roi = r; break
    elif event == cv2.EVENT_MOUSEMOVE and active_roi:
        if active_roi.dragging: active_roi.x, active_roi.y = x - active_roi.ox, y - active_roi.oy
        elif active_roi.resizing: active_roi.w, active_roi.h = max(20, x-active_roi.x), max(20, y-active_roi.y)
    elif event == cv2.EVENT_LBUTTONUP:
        if active_roi: active_roi.dragging = active_roi.resizing = False; active_roi = None

if __name__ == "__main__":
    system = POCSystem(1, roi_start, roi_end).start()
    cv2.namedWindow("Live Feed")
    cv2.setMouseCallback("Live Feed", mouse_event)
    
    ram_text, last_sys_update = "RAM: Init...", 0

    while True:
        if time.time() - last_sys_update > 1.0:
            ram = psutil.virtual_memory()
            ram_text = f"RAM: {ram.used / (1024**3):.1f}GB"
            last_sys_update = time.time()

        img = system.get_latest()
        if img is not None:
            roi_start.draw(img); roi_end.draw(img)
            # Display Status and Cooldown separately
            cv2.putText(img, system.status_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if system.cooldown_message:
                cv2.putText(img, system.cooldown_message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(img, ram_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Live Feed", img)

        if cv2.waitKey(20) & 0xFF == ord('q'): break

    system.stop()
    cv2.destroyAllWindows()