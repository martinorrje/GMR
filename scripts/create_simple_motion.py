import pickle
import numpy as np
import csv

joint_names=[
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_hip_pitch_joint",
      "left_knee_joint",
      "left_ankle_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_hip_pitch_joint",
      "right_knee_joint",
      "right_ankle_joint",
      "torso_pitch_joint",
      "torso_roll_joint",
      "torso_yaw_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "neck_yaw_joint",
      "neck_pitch_joint",
      "neck_roll_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
    ]

goal_pos = np.zeros(24)
goal_pos[joint_names.index("left_elbow_joint")] = 0.75035
goal_pos[joint_names.index("left_shoulder_roll_joint")] = 1.0
goal_pos[joint_names.index("right_shoulder_roll_joint")] = -1.0
goal_pos[joint_names.index("right_elbow_joint")] = 0.75035

output = {
    "root_pos": np.array([[0.0, 0.0, 0.301027]]*1000),
    "root_rot": np.array([[0, 0, 0, 1]]*1000),
    "dof_pos": np.zeros((1000, 24)),
    "local_body_pos": None,
    "link_body_list": None,
    "fps": 50.0,
}

for i in range(500):
    output["dof_pos"][i] = i / 500 * goal_pos

for i in range(500):
    output["dof_pos"][500+i] = (500-i)/500 * goal_pos

with open("simple_motion.pkl", "wb") as f:
    pickle.dump(output, f)

with open("simple_motion.csv", "w") as f:
    csv_writer = csv.writer(f)
    for row in range(len(output["dof_pos"])):
        csv_writer.writerow(list(output["root_pos"][i]) + list(output["root_rot"][i]) + list(output["dof_pos"][i]))
