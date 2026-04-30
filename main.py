import cv2
import threading
import time
import psutil
import numpy as np
import sys
import os
from collections import deque
from datetime import datetime
from pygrabber.dshow_graph import FilterGraph 

# --- CONFIG ---
CAP_W, CAP_H = 1280, 720
LIVE_W, LIVE_H = 640, 360
TARGET_FPS = 30
BUFFER_SEC = 15
MAX_FRAMES = TARGET_FPS * BUFFER_SEC

# Action Trigger Behavior
ACTION_TRIGGER = "start_roi_first_motion"

# Motion Sensitivity
START_SENSITIVITY = 20 # Based on a perentage of the ROI area that must change to trigger (e.g., 10% of pixels)
END_SENSITIVITY = 10 # Based on a perentage of the ROI area that must change to trigger (e.g., 10% of pixels)
MOTION_INTERVAL = 0.2 # Hom many seconds between motion checks (e.g., 0.1 for 10 checks per second)
POST_CAPTURE_WAIT = 12.0 

# Playback Padding
# -1.0 PRE and -4.0 POST are good starting points when using ACTION_TRIGGER = "start_roi_first_motion"
# 2.0 PRE and 4.0 POST are good starting points when using ACTION_TRIGGER = "start_roi_last_motion"
PRE_ACTION_PAD = -1.0 # Seconds to include before the detected start of action (to capture lead-in motion)
POST_ACTION_PAD = -4.0 # Seconds to continue recording after the detected end of action (to capture follow-through motion)

# Speed multipliers for the playback loop
PLAYBACK_LOOP = [1.0, 0.5]
PLAYBACK_SPEED_CHG = 0.1
PLAYBACK_SPEED_MIN = 0.1
PLAYBACK_SPEED_MAX = 2.0

SAVED_CLIPS_DIR = r"C:\Users\paul\Documents\autoframe\clips"

class ROI:
    def __init__(self, x, y, w, h, color, name, sensitivity):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.color = color
        self.name = name
        self.sensitivity = sensitivity
        self.dragging = False
        self.resizing = False
        self.last_roi_frame = None
        self.current_percent = 0.0  # Added for the live meter

    def draw(self, img):
        # Draw the main bounding box
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
        
        # Display name and the live motion meter (e.g., "START: 1.2% / 5.0%")
        label = f"{self.name}: {self.current_percent:.1f}% / {self.sensitivity}%"
        cv2.putText(img, label, (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        
        # Draw the resize handle (bottom-right corner)
        cv2.rectangle(img, (self.x + self.w - 10, self.y + self.h - 10), (self.x + self.w, self.y + self.h), self.color, -1)

    def detect_motion(self, current_roi_gray):
        if self.last_roi_frame is None:
            self.last_roi_frame = current_roi_gray
            return False
            
        diff = cv2.absdiff(self.last_roi_frame, current_roi_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = cv2.countNonZero(thresh)
        self.last_roi_frame = current_roi_gray
        
        # Calculate percentage of ROI area that changed
        roi_area = self.w * self.h
        if roi_area == 0: 
            return False
            
        self.current_percent = (motion_score / roi_area) * 100
        
        # Trigger if current motion exceeds sensitivity percentage
        return self.current_percent > self.sensitivity

class POCSystem:
    def __init__(self, index, name, r_start, r_end):
        self.index = index
        self.device_name = name
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
        self.is_active = False 
        self.lock = threading.Lock()
        
        self.status_message = "Setup - Press (b) to begin action capture" 
        self.cooldown_message = ""
        self.is_triggered = False 
        self.cooldown_until = 0 
        self.last_start_time = None
        self.event_end_time = None 
        
        self.playback_clip = None 
        self.playback_idx = 0
        self.playback_start_wall = 0
        self.playback_speed_idx = 0 
        self.playback_current_speed = 1.0
        self.manual_speed_override = False
        self.playback_paused = False
        
        self.frame_intervals = deque(maxlen=30)
        self.measured_fps = 0.0

    def activate_capture(self):
        if not self.is_active:
            self.is_active = True
            self.status_message = "Waiting..."

    def reset_to_setup(self):
        """Resets the system back to the initial Setup state."""
        self.is_active = False
        self.is_triggered = False
        self.cooldown_until = 0
        self.event_end_time = None
        self.playback_clip = None
        self.cooldown_message = ""
        self.status_message = "Setup - Press (b) to begin action capture"
        # Clear ROI history to prevent immediate false triggers on next start
        self.roi_start.last_roi_frame = None
        self.roi_end.last_roi_frame = None

    def print_characteristics(self):
        print("\n" + "="*50)
        print("           SYSTEM CHARACTERISTICS")
        print("="*50)
        print("1. DEVICE INFO")
        print(f"   - Selected: [{self.index}] {self.device_name}")
        print("\n2. BUFFER METRICS")
        print(f"   - Res:      {CAP_W}x{CAP_H}")
        print(f"   - Target:   {TARGET_FPS} FPS")
        print(f"   - Capacity: {BUFFER_SEC}s ({MAX_FRAMES} f)")
        print("\n3. LIVE FEED METRICS")
        print(f"   - Res:      {LIVE_W}x{LIVE_H}")
        print(f"   - Actual:   {self.measured_fps:.2f} FPS")
        print("="*50 + "\n")

    def print_controls(self):
        print("\n" + "="*50)
        print("           SYSTEM CONTROLS")
        print("="*50)
        print("GENERAL")
        print(f"   - Quit Autoframe:     [Q]")
        print(f"   - Begin Capture:      [B]")
        print(f"   - End Capture:        [E]")
        print(f"   - Save Current Clip:  [S]")
        print("PLAYBACK")
        print(f"   - Pause/Resume:       [P]")
        print(f"   - Speed Up:           [>]")
        print(f"   - Slow Down:          [<]")
        print(f"   - Frame Backward:     [Left Arrow]")
        print(f"   - Frame Forward:      [Right Arrow]")
        print("="*50 + "\n")

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        last_motion_check = 0
        last_t = time.perf_counter()
        while self.running:
            success, frame = self.cap.read()
            if success:
                now = time.perf_counter()
                self.frame_intervals.append(now - last_t)
                last_t = now
                if len(self.frame_intervals) > 0:
                    self.measured_fps = 1.0 / (sum(self.frame_intervals)/len(self.frame_intervals))

                small = cv2.resize(frame, (LIVE_W, LIVE_H))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                
                if self.is_active and now > self.cooldown_until:
                    if self.cooldown_message != "":
                        self.cooldown_message = ""
                        self.status_message = "Waiting..."
                        self.is_triggered = False 
                        self.last_start_time = None
                        self.event_end_time = None
                        self.roi_start.last_roi_frame = None
                        self.roi_end.last_roi_frame = None

                    if now - last_motion_check >= MOTION_INTERVAL:
                        if self.event_end_time:
                            if now >= self.event_end_time:
                                self.status_message = "Playback Started"
                                self.cooldown_until = now + POST_CAPTURE_WAIT
                                self._create_playback_clip(self.last_start_time - PRE_ACTION_PAD, self.event_end_time)
                                with self.lock:
                                    temp_list = list(self.buffer)
                                    self.buffer.clear()
                                    for t, f, _, _ in temp_list: self.buffer.append((t, f, False, False))
                                self.event_end_time = None
                                self.is_triggered = False

                        else:
                            sx, sy, sw, sh = self.roi_start.x, self.roi_start.y, self.roi_start.w, self.roi_start.h
                            if ACTION_TRIGGER == "start_roi_first_motion":
                                if not self.last_start_time and self.roi_start.detect_motion(gray[max(0,sy):min(LIVE_H,sy+sh), max(0,sx):min(LIVE_W,sx+sw)]):
                                    self.is_triggered = True
                                    self.last_start_time = now
                                    self.status_message = "Action Triggered"
                            elif ACTION_TRIGGER == "start_roi_last_motion":
                                if self.roi_start.detect_motion(gray[max(0,sy):min(LIVE_H,sy+sh), max(0,sx):min(LIVE_W,sx+sw)]):
                                    self.is_triggered = True
                                    self.last_start_time = now
                                    self.status_message = "Action Triggered"

                            if self.is_triggered:
                                ex, ey, ew, eh = self.roi_end.x, self.roi_end.y, self.roi_end.w, self.roi_end.h
                                if self.roi_end.detect_motion(gray[max(0,ey):min(LIVE_H,ey+eh), max(0,ex):min(LIVE_W,ex+ew)]):
                                    self.event_end_time = now + POST_ACTION_PAD
                                    self.status_message = "Finishing Capture..."
                        last_motion_check = now
                elif self.is_active:
                    self.cooldown_message = f"Cooldown: {int(self.cooldown_until - now)}s"

                self.buffer.append((now, frame, False, False))
                with self.lock:
                    self.live_proxy = small
                    self.new_frame_available = True
            else:
                time.sleep(0.001)

    def _create_playback_clip(self, start_t, end_t):
        with self.lock:
            current_buffer = list(self.buffer)
            self.playback_clip = [f for f in current_buffer if start_t <= f[0] <= end_t]
            self.playback_idx = 0
            self.playback_speed_idx = 0 
            self.manual_speed_override = False
            self.playback_current_speed = PLAYBACK_LOOP[0]
            self.playback_start_wall = time.perf_counter()
            self.playback_paused = False

    def get_latest(self):
        with self.lock:
            if not self.new_frame_available: return None
            self.new_frame_available = False
            return self.live_proxy

    def stop(self):
        self.running = False
        self.cap.release()

def get_friendly_camera_list():
    return FilterGraph().get_input_devices()
def save_playback_clip(clip_data):
    """Saves the current playback buffer to an MP4 file in a background thread."""
    if not clip_data:
        print("Save failed: No clip data available.")
        return

    if not os.path.exists(SAVED_CLIPS_DIR):
        os.makedirs(SAVED_CLIPS_DIR)

    # Filename format: autoframe_yyyymmdd_hhmmss.mp4
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"autoframe_{timestamp}.mp4"
    filepath = os.path.join(SAVED_CLIPS_DIR, filename)

    # Get metadata from the first frame of the clip
    first_frame = clip_data[0][1]
    height, width = first_frame.shape[:2]
    
    # Define codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, TARGET_FPS, (width, height))

    print(f"Saving clip to {filepath}...")
    for _, frame, _, _ in clip_data:
        out.write(frame)
    
    out.release()
    print("Save Complete.")

# Global UI Setup
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
    device_names = get_friendly_camera_list()
    if not device_names:
        print("\nERROR: No cameras found.")
        sys.exit()

    if len(device_names) == 1:
        # Auto-select single camera
        selected_index = 0
        selected_name = device_names[0]
        print(f"\n[INFO] Auto-selected camera: [{selected_index}] {selected_name}")
    else:
        print("\n" + "="*40)
        print("        CAMERA SELECTION MENU")
        print("="*40)
        for i, name in enumerate(device_names):
            print(f" [{i}] {name}")
        print("="*40)
        
        try:
            choice = int(input("\nSelect camera number: "))
            selected_index = choice 
            selected_name = device_names[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Defaulting to [0].")
            selected_index = 0
            selected_name = device_names[0]

    system = POCSystem(selected_index, selected_name, roi_start, roi_end).start()
    time.sleep(1.5)
    system.print_characteristics()
    system.print_controls()

    cv2.namedWindow("Live Feed")
    cv2.setMouseCallback("Live Feed", mouse_event)
    
    ram_text, last_sys_update = "RAM: Init...", 0
    try:
        while True:
            if time.time() - last_sys_update > 1.0:
                ram = psutil.virtual_memory()
                ram_text = f"RAM: {ram.used / (1024**3):.1f}GB"
                last_sys_update = time.time()

            img = system.get_latest()
            if img is not None:
                roi_start.draw(img); roi_end.draw(img)
                color = (0, 255, 255) if not system.is_active else (255, 255, 0)
                cv2.putText(img, system.status_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if system.cooldown_message:
                    cv2.putText(img, system.cooldown_message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.putText(img, ram_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Live Feed", img)

            if system.playback_clip:
                cv2.namedWindow("Auto Playback", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Auto Playback", CAP_W, CAP_H) 
                clip = system.playback_clip
                
                if not system.playback_paused:
                    elapsed_needed = (clip[system.playback_idx][0] - clip[0][0]) / system.playback_current_speed
                    if (time.perf_counter() - system.playback_start_wall) >= elapsed_needed:
                        system.playback_idx += 1
                        if system.playback_idx >= len(clip):
                            system.playback_idx = 0
                            # Only cycle if manual override hasn't been used
                            if not system.manual_speed_override:
                                system.playback_speed_idx = (system.playback_speed_idx + 1) % len(PLAYBACK_LOOP)
                                system.playback_current_speed = PLAYBACK_LOOP[system.playback_speed_idx]
                            system.playback_start_wall = time.perf_counter()

                idx = min(system.playback_idx, len(clip) - 1)
                display_frame = clip[idx][1].copy()
                status_txt = f"Speed: {system.playback_current_speed:.1f}x"
                if system.manual_speed_override:
                    status_txt += " (Manual)"
                else:
                    status_txt += " (Auto Loop)"
                if system.playback_paused: status_txt += " [PAUSED]"
                cv2.putText(display_frame, status_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Auto Playback", display_frame)
            else:
                # If playback was reset, ensure window is closed
                if cv2.getWindowProperty("Auto Playback", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Auto Playback")

            key = cv2.waitKeyEx(1)
            if key != -1:
                key_8 = key & 0xFF
                if key_8 == ord('q'): break
                elif key_8 == ord('b'):
                    system.activate_capture()
                elif key_8 == ord('e'): # End capture and reset to Setup
                    system.reset_to_setup()
                elif key_8 == ord('s'):
                    if system.playback_clip:
                        # Run saving in a background thread so it doesn't freeze the UI
                        threading.Thread(target=save_playback_clip, args=(list(system.playback_clip),), daemon=True).start()
                elif key_8 == ord('p'):
                    if system.playback_clip:
                        system.playback_paused = not system.playback_paused
                        if not system.playback_paused:
                            offset = (system.playback_clip[system.playback_idx][0] - system.playback_clip[0][0]) / system.playback_current_speed
                            system.playback_start_wall = time.perf_counter() - offset
                
                # Speed Controls
                elif key_8 == ord(','): # (<) Key
                    if system.playback_clip:
                        system.manual_speed_override = True
                        system.playback_current_speed = max(PLAYBACK_SPEED_MIN, system.playback_current_speed - PLAYBACK_SPEED_CHG)
                        offset = (system.playback_clip[system.playback_idx][0] - system.playback_clip[0][0]) / system.playback_current_speed
                        system.playback_start_wall = time.perf_counter() - offset
                elif key_8 == ord('.'): # (>) Key
                    if system.playback_clip:
                        system.manual_speed_override = True
                        system.playback_current_speed = min(PLAYBACK_SPEED_MAX, system.playback_current_speed + PLAYBACK_SPEED_CHG)
                        offset = (system.playback_clip[system.playback_idx][0] - system.playback_clip[0][0]) / system.playback_current_speed
                        system.playback_start_wall = time.perf_counter() - offset

                # Frame Stepping
                elif key == 2424832: # Left Arrow
                    if system.playback_clip:
                        system.playback_paused = True
                        system.playback_idx = (system.playback_idx - 1) % len(system.playback_clip)
                elif key == 2555904: # Right Arrow
                    if system.playback_clip:
                        system.playback_paused = True
                        system.playback_idx = (system.playback_idx + 1) % len(system.playback_clip)
    finally:
        print("\n[INFO] Cleaning up resources...")
+       system.stop()
+       cv2.destroyAllWindows()
+       print("[INFO] Shutdown complete.")