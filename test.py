import cv2
import time

def list_cameras():
    index = 0
    available_cameras = []
    while True:
        # Try to open the camera
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  # Release the camera if opened
        else:
            break  # No more cameras found
        index += 1
    
    return available_cameras

# Get the list of cameras
cameras = list_cameras()

if not cameras:
    print("No cameras found.")
else:
    print("Available cameras and their indices:")
    for cam in cameras:
        print(f"Camera Index: {cam}")

    # Choose a camera index to use
    selected_camera = cameras[0]  # You can modify this to choose a different camera
    print(f"\nUsing Camera Index: {selected_camera}")
    
    # Initialize video capture from the selected camera
    cap = cv2.VideoCapture(selected_camera)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec
    out = cv2.VideoWriter(r'C:\Users\paul\Documents\Temp\test_vid_01.mkv', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    # Record for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            out.write(frame)  # Write the frame to the video file
            cv2.imshow('Recording', frame)  # (Optional) Show the frame being recorded
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error reading from camera.")
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording complete.")

