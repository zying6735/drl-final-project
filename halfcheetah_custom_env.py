import os
import mujoco
import imageio
import numpy as np
import xml.etree.ElementTree as ET

def string_to_vec(s):
    return np.array([float(x) for x in s.split()])

def vec_to_string(v, precision=6):
    return " ".join([f"{x:.{precision}f}" for x in v])

def rotated_z_vector(axisangle_str_or_list):
    axisangle = string_to_vec(axisangle_str_or_list) if isinstance(axisangle_str_or_list, str) else np.array(axisangle_str_or_list, dtype=np.float64)
    axis, angle = axisangle[:3], axisangle[3]
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-9 or abs(angle) < 1e-9:
        return np.array([0., 0., 1.])
    axis /= norm_axis
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R @ np.array([0., 0., 1.])

def _modify_single_segment_xml(xml_root, parent_body_name, geom_name, child_body_name, new_full_segment_length):
    parent = xml_root.find(f'.//body[@name="{parent_body_name}"]')
    if parent is None: return False
    geom = parent.find(f'./geom[@name="{geom_name}"]')
    if geom is None or 'axisangle' not in geom.attrib or 'size' not in geom.attrib:
        return False

    direction = rotated_z_vector(geom.get('axisangle'))
    if geom_name in ["fthigh", "fshin"] and direction[2] > 0.5:
        direction *= -1

    radius = string_to_vec(geom.get('size'))[0]
    half_len = new_full_segment_length / 2.0
    geom.set('size', vec_to_string([radius, half_len]))
    geom.set('pos', vec_to_string(half_len * direction))

    if child_body_name:
        child = parent.find(f'./body[@name="{child_body_name}"]')
        if child is not None:
            child.set('pos', vec_to_string(new_full_segment_length * direction))

    return True

def set_floor_friction(xml_root, new_friction_vec):
    ground = xml_root.find('.//geom[@type="plane"]')
    if ground is None:
        return False
    ground.set('friction', vec_to_string(new_friction_vec))
    return True

def modify_all_leg_lengths_and_friction(xml_str, b_thigh_len, b_shin_len, f_thigh_len, f_shin_len, floor_friction):
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    segments = [
        ("bthigh", "bthigh", "bshin", b_thigh_len),
        ("bshin", "bshin", "bfoot", b_shin_len),
        ("fthigh", "fthigh", "fshin", f_thigh_len),
        ("fshin", "fshin", "ffoot", f_shin_len)
    ]
    if not all(_modify_single_segment_xml(root, *seg) for seg in segments):
        return None

    if not set_floor_friction(root, floor_friction):
        return None

    return ET.tostring(root, encoding='unicode')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "half_cheetah.xml")

    if not os.path.exists(xml_path):
        exit(1)

    with open(xml_path, 'r') as f:
        original_xml = f.read()

    new_lengths = dict(
        b_thigh_len=0.0725,
        b_shin_len=0.075,
        f_thigh_len=0.0665,
        f_shin_len=0.053
    )

    floor_friction = [0.8, 0.1, 0.01]  # sliding, torsional, rolling

    modified_xml = modify_all_leg_lengths_and_friction(
        original_xml,
        **new_lengths,
        floor_friction=floor_friction
    )

    if modified_xml is None:
        exit(1)

    try:
        model = mujoco.MjModel.from_xml_string(modified_xml)
    except Exception:
        exit(1)

    data = mujoco.MjData(model)
    frames = []
    try:
        renderer = mujoco.Renderer(model, width=640, height=480)
    except Exception:
        renderer = None

    if renderer:
        mujoco.mj_resetData(model, data)
        for step in range(200):
            data.ctrl[:] = 0.0 if step < 20 else np.random.uniform(-1.0, 1.0, size=model.nu)
            mujoco.mj_step(model, data)
            try:
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render())
            except Exception:
                break
        renderer.close()

    if frames:
        gif_path = os.path.join(script_dir, "halfcheetah_custom_lengths.gif")
        imageio.mimsave(gif_path, frames, fps=30)