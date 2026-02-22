import cv2
import numpy as np
import collections
import os
import threading
import time
from pygrabber.dshow_graph import FilterGraph

# --- [AUTOFRAME CONFIGURATION] ---
SAVE_PATH = r"C:\Users\paul\Documents\Temp\test_vid_01.mkv"
MIN_MOTION_AREA = 1000      
VAR_THRESHOLD = 45          
BLUR_SIZE = (7, 7)          

# --- [TIME BUFFERS (Seconds)] ---
PRE_APPROACH_BUFFER = 2.0   # Exactly how much to prepend to the start
POST_RELEASE_BUFFER = 2.0   

# --- [PLAYBACK VELOCITY] ---
SPEED_1 = 1.0   
SPEED_2 = 0.5   
SPEED_3 = 0.25   

# Global variables
roi1_coords = [50, 250, 50, 250]
roi2_coords = [50, 250, 500, 700]
dragging_roi1 = dragging_roi2 = False
new_shot_ready = threading.Event()
app_running = True 
is_recording = False 
actual_fps = 30.0 

def mouse_callback(event, x, y, flags, param):
    global roi1_coords, roi2_coords, dragging_roi1, dragging_roi2
    if event == cv2.EVENT_LBUTTONDOWN: dragging_roi1 = True
    elif event == cv2.EVENT_LBUTTONUP: dragging_roi1 = False
    elif event == cv2.EVENT_RBUTTONDOWN: dragging_roi2 = True
    elif event == cv2.EVENT_RBUTTONUP: dragging_roi2 = False
    if dragging_roi1:
        h, w = roi1_coords[1]-roi1_coords[0], roi1_coords[3]-roi1_coords[2]
        roi1_coords[0:4] = [y, y+h, x, x+w]
    if dragging_roi2:
        h, w = roi2_coords[1]-roi2_coords[0], roi2_coords[3]-roi2_coords[2]
        roi2_coords[0:4] = [y, y+h, x, x+w]

def list_system_cameras():
    graph = FilterGraph()
    device_names = graph.get_input_devices()
    available = []
    print("\n" + "="*30 + "\n AUTOFRAME HARDWARE SCOUT \n" + "="*30)
    for i, name in enumerate(device_names):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Index [{i}]: {name}"); available.append(i)
            cap.release()
    return available

def playback_thread():
    global app_running, is_recording
    while app_running:
        if is_recording:
            try: cv2.destroyWindow('AutoFrame - Instant Review')
            except: pass
            while is_recording and app_running:
                time.sleep(0.1)
            continue

        if os.path.exists(SAVE_PATH):
            cap_pb = cv2.VideoCapture(SAVE_PATH)
            vid_fps = cap_pb.get(cv2.CAP_PROP_FPS)
            if vid_fps <= 0: vid_fps = 30.0
            frames = []
            while True:
                ret, frame = cap_pb.read()
                if not ret: break
                frames.append(frame)
            cap_pb.release()

            if not frames: 
                time.sleep(0.5)
                continue
            
            speeds = [SPEED_1, SPEED_2, SPEED_3]
            while not new_shot_ready.is_set() and app_running and not is_recording:
                for speed in speeds:
                    if new_shot_ready.is_set() or not app_running or is_recording: break
                    target_duration = (1.0 / vid_fps) / speed
                    for f in frames:
                        t_start = time.perf_counter()
                        if new_shot_ready.is_set() or not app_running or is_recording: break
                        review_frame = f.copy()
                        cv2.putText(review_frame, f"Review Speed: {int(speed*100)}%", (10, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow('AutoFrame - Instant Review', review_frame)
                        elapsed = time.perf_counter() - t_start
                        sleep_time = max(1, int((target_duration - elapsed) * 1000))
                        if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
                            app_running = False
                            return
            new_shot_ready.clear() 
        else:
            time.sleep(0.5)

def run_autoframe(camera_index):
    global roi1_coords, roi2_coords, actual_fps, app_running, is_recording
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nCalibrating timing...")
    t_start = time.time()
    for _ in range(30): cap.read()
    actual_fps = 30 / (time.time() - t_start)
    print(f"Calibrated FPS: {actual_fps:.2f}")

    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    cv2.namedWindow('AutoFrame - Active Feed')
    cv2.setMouseCallback('AutoFrame - Active Feed', mouse_callback)
    threading.Thread(target=playback_thread, daemon=True).start()
    
    frame_buffer = collections.deque(maxlen=int(actual_fps * 20))
    state = 0 
    locked_start_idx = 0
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=VAR_THRESHOLD, detectShadows=True)

    print("\n[READY] - Monitoring for activity.")

    while app_running:
        ret, frame = cap.read()
        if not ret: break
        frame_buffer.append(frame.copy())
        
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BLUR_SIZE, 0)
        fgmask = fgbg.apply(gray)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

        y1, y2, x1, x2 = [max(0, int(c)) for c in roi1_coords]
        ry1, ry2, rx1, rx2 = [max(0, int(c)) for c in roi2_coords]
        
        roi1_m = any(cv2.contourArea(c) > MIN_MOTION_AREA for c in cv2.findContours(fgmask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        roi2_m = any(cv2.contourArea(c) > MIN_MOTION_AREA for c in cv2.findContours(fgmask[ry1:ry2, rx1:rx2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

        # --- STATE 0: IDLE ---
        if state == 0 and roi1_m:
            # LOCK THE START: 2 seconds before this exact frame
            locked_start_idx = max(0, len(frame_buffer) - int(actual_fps * PRE_APPROACH_BUFFER))
            print(f">> [STATUS] ROI 1 Triggered. Start locked at -{PRE_APPROACH_BUFFER}s.")
            is_recording = True
            state = 1

        # --- STATE 1: TRACKING ---
        elif state == 1:
            if roi2_m:
                print(f">> [STATUS] ROI 2 Triggered. Finalizing buffer...")
                cooldown_frames = int(actual_fps * POST_RELEASE_BUFFER)
                state = 2

        # --- STATE 2: SAVING ---
        elif state == 2:
            cooldown_frames -= 1
            if cooldown_frames <= 0:
                print(f">> [STATUS] Saving clip...")
                out = cv2.VideoWriter(SAVE_PATH, fourcc, actual_fps, (width, height))
                buffer_snapshot = list(frame_buffer)
                for i in range(locked_start_idx, len(buffer_snapshot)):
                    out.write(buffer_snapshot[i])
                out.release()
                
                print(">> [STATUS] Shot saved. Restarting monitoring...\n")
                is_recording = False
                new_shot_ready.set() 
                state = 0
                frame_buffer.clear()

        # Visuals
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2) 
        cv2.imshow('AutoFrame - Active Feed', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            app_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cams = list_system_cameras()
    if cams:

        c = int(input(f"\nAvailable cameras: {cams}\nEnter the index of the camera to use: "))

        run_autoframe(cams[c])