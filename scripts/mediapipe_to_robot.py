import cv2
import mediapipe as mp
import threading
import time
import numpy as np
import argparse

# --- Your existing imports (Keep these) ---
# from general_motion_retargeting import GeneralMotionRetargeting as GMR
# from general_motion_retargeting import RobotMotionViewer

# --- Mocking them for this example to run standalone ---
# (Remove these mocks when using your actual library)
class GMR:
    def __init__(self, src_human, tgt_robot, actual_human_height): pass
    def retarget(self, frame): 
        # Mock output: root_pos(3), root_rot(4), dof(12 for example)
        return np.zeros(19) 

class RobotMotionViewer:
    def __init__(self, robot_type): pass
    def step(self, root_pos, root_rot, dof_pos, rate_limit): pass
# -------------------------------------------------------


class MediaPipeClient:
    """
    A drop-in replacement for NatNetClient that uses a Webcam + MediaPipe
    to estimate 3D pose in real-time.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.is_running = False
        self.lock = threading.Lock()
        
        # MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        # Data storage
        self.latest_frame_data = None
        self.frame_count = 0
        self.start_time = time.time()

    def run(self):
        """Starts the capture loop in the background thread."""
        cap = cv2.VideoCapture(self.camera_index)
        self.is_running = True
        
        print(f"MediaPipe Client: Camera {self.camera_index} started.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_world_landmarks:
                # Format data into a clean dictionary
                # pose_world_landmarks are in meters, centered at hip
                formatted_data = self._process_landmarks(results.pose_world_landmarks)
                
                with self.lock:
                    self.latest_frame_data = formatted_data
                    self.frame_count += 1
            
            # Optional: Sleep slightly to match camera rate vs CPU usage
            time.sleep(0.001)

        cap.release()
        print("MediaPipe Client: Stopped.")

    def _process_landmarks(self, world_landmarks):
        """
        Maps MediaPipe indices to named joints. 
        MediaPipe is Y-down by default in pixel space, but world_landmarks 
        are roughly metric (-1 to 1 range usually).
        """
        lm = world_landmarks.landmark
        
        # Helper to get np array
        def get_vec(idx):
            return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

        # Mapping MediaPipe landmarks to common skeletal names
        # Adjust these keys to match what GMR expects
        joints = {
            "Hips": (get_vec(23) + get_vec(24)) / 2.0,
            "LeftUpLeg": get_vec(23),
            "LeftLeg": get_vec(25),   # Knee
            "LeftFoot": get_vec(27),  # Ankle
            "RightUpLeg": get_vec(24),
            "RightLeg": get_vec(26),
            "RightFoot": get_vec(28),
            "Spine": (get_vec(11) + get_vec(12)) / 2.0, # Mid-shoulder
            "LeftArm": get_vec(11),   # Shoulder
            "LeftForeArm": get_vec(13), # Elbow
            "LeftHand": get_vec(15),    # Wrist
            "RightArm": get_vec(12),
            "RightForeArm": get_vec(14),
            "RightHand": get_vec(16),
            "Head": get_vec(0),       # Nose
        }
        return joints

    def connected(self):
        return self.is_running

    def get_frame(self):
        """Returns the latest available frame data."""
        with self.lock:
            return self.latest_frame_data
            
    def get_frame_number(self):
        return self.frame_count


def main(args):
    print("Starting MediaPipe Retargeting...")
    
    # 1. Setup the Camera Client
    client = MediaPipeClient(camera_index=0)

    # 2. Start the capture thread
    thread = threading.Thread(target=client.run, daemon=True)
    thread.start()
    
    # Allow camera to warm up
    time.sleep(1.5)

    if not client.connected():
        print("Failed to start Camera")
        exit(1)

    # 3. Setup Retargeter
    # We change src_human to indicate we are passing a dict/live data, 
    # though this depends on your specific GMR implementation.
    retarget = GMR(
        src_human="mediapipe", 
        tgt_robot=args.robot,
        actual_human_height=1.7, # MediaPipe needs height scaling often
    )
    
    viewer = RobotMotionViewer(robot_type=args.robot)

    print("Loop starting...")
    
    while True:
        try:
            # 4. Get Data
            frame_data = client.get_frame()
            frame_number = client.get_frame_number()

            if frame_data is None:
                continue

            # 5. Retarget
            # Ensure your GMR.retarget method can handle the dictionary 
            # returned by client.get_frame()
            qpos = retarget.retarget(frame_data)

            # 6. Visualize
            # Assuming qpos structure: [root_pos(3), root_rot(4), joints(N)]
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                rate_limit=False,
            )
            
        except KeyboardInterrupt:
            print("Stopping...")
            client.is_running = False
            thread.join()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We don't need server_ip/client_ip for USB camera
    parser.add_argument("--robot", type=str, default="unitree_g1")
    args = parser.parse_args()
    
    main(args)