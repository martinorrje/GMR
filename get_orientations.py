import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

def parse_quat(qstr):
    # MuJoCo uses w x y z
    return np.array([float(x) for x in qstr.split()])

def multiply_quat(q1, q2):
    # Both in [w, x, y, z]
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    r = r1 * r2
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])

def traverse_bodies(elem, parent_quat, results):
    # Get this body's name and local quat
    name = elem.attrib.get('name')
    local_quat = np.array([1, 0, 0, 0])
    if 'quat' in elem.attrib:
        local_quat = parse_quat(elem.attrib['quat'])
    global_quat = multiply_quat(parent_quat, local_quat)
    if name:
        results[name] = {"global_quat": global_quat}
    # Recurse into children
    for child in elem:
        if child.tag == 'body':
            traverse_bodies(child, global_quat, results)

def smplx_to_mujoco_quat():
    # Quaternion that rotates SMPL-X (X right, Y up, Z forward)
    # to MuJoCo (X forward, Y left, Z up)
    # This is a -90 deg about X, then -90 deg about Z
    r = R.from_euler('x', 90, degrees=True) * R.from_euler('y', 90, degrees=True)
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])

def get_smplx_to_body_quats():
    tree = ET.parse('assets/berkeley_humanoid_lite/berkeley_humanoid_lite.xml')
    root = tree.getroot()
    worldbody = root.find('worldbody')
    results = {}
    smplx2mj = smplx_to_mujoco_quat()
    for body in worldbody.findall('body'):
        traverse_bodies(body, np.array([1, 0, 0, 0]), results)
    output = {}
    # For each body, compute the quaternion from SMPL-X to this body's global orientation
    for name, data in results.items():
        q_body = data["global_quat"]
        # The quaternion to rotate from SMPL-X to this body is:
        # q = q_body * (smplx2mj)^-1
        r_body = R.from_quat([q_body[1], q_body[2], q_body[3], q_body[0]])
        r_smplx2mj = R.from_quat([smplx2mj[1], smplx2mj[2], smplx2mj[3], smplx2mj[0]])
        r = r_body * r_smplx2mj.inv()
        q = r.as_quat()  # [x, y, z, w]
        data["smplx_to_body_quat"] = [q[3], q[0], q[1], q[2]]
        output[name] = [q[3], q[0], q[1], q[2]]
    return output
        
orientations = get_smplx_to_body_quats()  # Returns {body_name: [w, x, y, z], ...}

with open('general_motion_retargeting/ik_configs/smplx_to_bhl.json', 'r') as f:
    config = json.load(f)

def update_table(table):
    for body_name, entry in table.items():
        if body_name in orientations:
            # Replace the orientation (last element) with the computed quaternion
            entry[-1] = orientations[body_name]
        else:
            print(f"Warning: {body_name} not found in orientations, skipping.")

update_table(config['ik_match_table1'])
update_table(config['ik_match_table2'])

with open('general_motion_retargeting/ik_configs/smplx_to_bhl.json', 'w') as f:
    json.dump(config, f, indent=4)