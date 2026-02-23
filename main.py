import cv2
import numpy as np
import collections
import threading
import time
import queue
import sys
from pygrabber.dshow_graph import FilterGraph

# --- [USER CONSTANTS] ---
FPS_CONSTANT = 15.0     # [NEW] Static FPS for all calculations
SENSITIVITY = 500      
PRE_ROLL_S = 4.0        
POST_ROLL_S = 2.0       
BUFFER_MAX_S = 15.0     

# ROI Definitions: [y1, y2, x1, x2]
roi1 = [100, 400, 100, 400]
roi2 = [100, 400, 800, 1100]

selected_roi = None
selected_corner = None 

# --- [SYSTEM CONFIG] ---
SAVE_PATH = r"C:\Users\paul\Documents\Temp\test_vid_01.mkv"
VAR_THRESHOLD = 45
PLAYBACK_SPEEDS = [1.0, 0.5, 0.25]

frame_queue = queue.Queue(maxsize=10)
replay_queue = queue.Queue(maxsize=1) 
app_running = True

def mouse_callback(event, x, y, flags, param):
    global roi1, roi2, selected_roi, selected_corner
    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(x - roi1[2]) < 20 and abs(y - roi1[0]) < 20: selected_roi, selected_corner = 1, 0
        elif abs(x - roi1[3]) < 20 and abs(y - roi1[1]) < 20: selected_roi, selected_corner = 1, 1
        elif abs(x - roi2[2]) < 20 and abs(y - roi2[0]) < 20: selected_roi, selected_corner = 2, 0
        elif abs(x - roi2[3]) < 20 and abs(y - roi2[1]) < 20: selected_roi, selected_corner = 2, 1
    elif event == cv2.EVENT_MOUSEMOVE and selected_roi:
        target = roi1 if selected_roi == 1 else roi2
        if selected_corner == 0: target[2], target[0] = x, y
        else: target[3], target[1] = x, y
    elif event == cv2.EVENT_LBUTTONUP: selected_roi = None

def camera_producer(cap):
    """Feeds frames into the processing queue."""
    global app_running
    while app_running:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(frame)
        else:
            time.sleep(0.01)

def playback_worker():
    """Handles RAM-based replay in a separate window."""
    global app_running
    cv2.namedWindow('AutoFrame - Instant Review')
    active_clip = []
    while app_running:
        try: active_clip = replay_queue.get_nowait()
        except queue.Empty: pass
        if not active_clip:
            time.sleep(0.1); continue

        for speed in PLAYBACK_SPEEDS:
            if not app_running or not replay_queue.empty(): break
            # Delay based on the static FPS constant
            delay = int((1.0 / FPS_CONSTANT) / speed * 1000)
            for f in active_clip:
                if not app_running or not replay_queue.empty(): break
                disp = f.copy()
                cv2.putText(disp, f"REPLAY: {int(speed*100)}% | CFG: {FPS_CONSTANT} FPS", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('AutoFrame - Instant Review', disp)
                if cv2.waitKey(max(1, delay)) & 0xFF == ord('q'):
                    app_running = False; return

def list_system_cameras():
    graph = FilterGraph()
    device_names = graph.get_input_devices()
    available = []
    print("\n" + "="*50 + "\n        AUTOFRAME HARDWARE SCOUT \n" + "="*50)
    for i, name in enumerate(device_names):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Test high-res capability
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.read() 
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Index [{i}]: {name}\n         -> Max Detected: {w}x{h}")
            available.append(i)
            cap.release()
    print("="*50 + "\n")
    return available

def run_autoframe(index):
    global app_running, roi1, roi2
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    # Force MJPG and Max Resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, FPS_CONSTANT)
    
    ret, first_frame = cap.read()
    if not ret: 
        print("Error: Could not read from camera."); return
    height, width = first_frame.shape[:2]
    print(f"\n[SYSTEM] Active: {width}x{height} | Target: {FPS_CONSTANT} FPS")

    # Fixed buffer size based on static FPS
    max_frames = int(FPS_CONSTANT * BUFFER_MAX_S)
    frame_buffer = collections.deque(maxlen=max_frames)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=VAR_THRESHOLD)

    cv2.namedWindow('AutoFrame - Active Feed')
    cv2.setMouseCallback('AutoFrame - Active Feed', mouse_callback)

    threading.Thread(target=camera_producer, args=(cap,), daemon=True).start()
    threading.Thread(target=playback_worker, daemon=True).start()

    state = 0 # 0:IDLE, 1:ROI1_HIT, 2:RECORDING
    start_time = 0
    cooldown_end = 0

    while app_running:
        try: frame = frame_queue.get(timeout=1)
        except queue.Empty: continue

        frame_buffer.append(frame)
        
        # UI Gauge
        pct = (len(frame_buffer) / max_frames * 100) if max_frames > 0 else 0
        sys.stdout.write(f"\r[BUF: {pct:3.0f}%] | RES: {width}x{height} | STATE: {state} ")
        sys.stdout.flush()

        fgmask = fgbg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        
        # ROI Bounds Protection
        r1y1, r1y2, r1x1, r1x2 = max(0,roi1[0]), min(height,roi1[1]), max(0,roi1[2]), min(width,roi1[3])
        r2y1, r2y2, r2x1, r2x2 = max(0,roi2[0]), min(height,roi2[1]), max(0,roi2[2]), min(width,roi2[3])
        
        m1 = np.sum(fgmask[r1y1:r1y2, r1x1:r1x2] > 250) > SENSITIVITY
        m2 = np.sum(fgmask[r2y1:r2y2, r2x1:r2x2] > 250) > SENSITIVITY

        if state == 0 and m1:
            start_time = time.time()
            state = 1
        elif state == 1 and m2:
            state = 2
            cooldown_end = time.time() + POST_ROLL_S
        elif state == 2:
            if m2: cooldown_end = time.time() + POST_ROLL_S
            if time.time() > cooldown_end:
                # Use static FPS for clip slicing
                frame_count = int(((time.time() - start_time) + PRE_ROLL_S) * FPS_CONSTANT)
                clip = list(frame_buffer)[-min(len(frame_buffer), frame_count):]
                replay_queue.put(clip)
                state = 0

        # UI
        ui_frame = frame.copy()
        cv2.rectangle(ui_frame, (roi1[2], roi1[0]), (roi1[3], roi1[1]), (0, 255, 0), 2)
        cv2.rectangle(ui_frame, (roi2[2], roi2[0]), (roi2[3], roi2[1]), (0, 0, 255), 2)
        cv2.imshow('AutoFrame - Active Feed', ui_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): app_running = False

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    cams = list_system_cameras()
    if cams:
        idx = input(f"Select Camera Index: ")
        run_autoframe(int(idx))