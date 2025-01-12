from typing import List, Tuple

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from task2_inverse_kinematics import MetaData

class Task2Pose:
    def __init__(self, index: int, name: str, position: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=float), orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)):
        self.index = index
        self.name = name

        self.local_position = deepcopy(position)
        self.local_orientation = deepcopy(orientation)

        self.global_position = deepcopy(position)
        self.global_orientation = deepcopy(orientation)
        self.global_position_dirty = False
        self.global_orientation_dirty = False

        self.parent: Task2Pose = None
        self.children: List[Task2Pose] = []

    def get_local_position(self) -> np.ndarray:
        return deepcopy(self.local_position)
    
    def get_local_orientation(self) -> np.ndarray:
        return deepcopy(self.local_orientation)
        
    def get_global_position(self) -> np.ndarray:
        if self.global_position_dirty:
            if self.parent == None:
                self.global_position = self.get_local_position()
                self.global_orientation = self.get_local_orientation()
            else:
                parent_global_translation = self.parent.get_global_position()
                parent_global_rotation = self.parent.get_global_orientation()
                self.global_position = parent_global_translation + R.from_quat(parent_global_rotation).as_matrix() @ self.get_local_position()
            self.global_position_dirty = False
        return deepcopy(self.global_position)
    
    def get_global_orientation(self) -> np.ndarray:
        if self.global_orientation_dirty:
            if self.parent == None:
                self.global_position = self.get_local_position()
                self.global_orientation = self.get_local_orientation()
            else:
                parent_global_rotation = self.parent.get_global_orientation()
                self.global_orientation = (R.from_quat(parent_global_rotation) * R.from_quat(self.get_local_orientation())).as_quat()
            self.global_orientation_dirty = False
        return deepcopy(self.global_orientation)
    
    def notify_global_position_changed(self) -> np.ndarray:
        self.global_position_dirty = True
        for child in self.children:
            child.notify_global_position_changed()

    def notify_global_orientation_changed(self) -> np.ndarray:
        self.global_orientation_dirty = True
        for child in self.children:
            child.notify_global_orientation_changed()
            child.notify_global_position_changed()
    
    def set_local_position(self, position: np.ndarray):
        self.local_position = deepcopy(position)
        self.notify_global_position_changed()

    def set_local_orientation(self, orientation: np.ndarray):
        self.local_orientation = deepcopy(orientation)
        self.notify_global_orientation_changed()

    def set_global_position(self, position: np.ndarray):
        if self.parent == None:
            self.set_local_position(position)
        else:
            parent_global_translation = self.parent.get_global_position()
            parent_global_rotation = self.parent.get_global_orientation()
            local_position = R.from_quat(parent_global_rotation).inv().as_matrix() @ (position - parent_global_translation)
            self.set_local_position(local_position)
    
    def set_global_orientation(self, orientation: np.ndarray):
        if self.parent == None:
            self.set_local_orientation(orientation)
        else:
            parent_global_rotation = self.parent.get_global_orientation()
            local_orientation = (R.from_quat(parent_global_rotation).inv() * R.from_quat(orientation)).as_quat()
            self.set_local_orientation(local_orientation)
    
    def set_parent(self, parent: "Task2Pose"):
        parent.children.append(self)
        self.parent = parent
        self.set_global_position(self.get_global_position())
        self.set_global_orientation(self.get_global_orientation())

def init_joint_poses(out_joint_poses: List[Task2Pose], joint_num: int, joint_name: List[str], joint_parent: List[int], joint_positions: np.ndarray, joint_orientations: np.ndarray):
    for joint_i in range(joint_num):
        joint_pose = Task2Pose(joint_i, joint_name[joint_i], joint_positions[joint_i], joint_orientations[joint_i])
        if joint_parent[joint_i] != -1:
            joint_pose.set_parent(out_joint_poses[joint_parent[joint_i]])
        out_joint_poses.append(joint_pose)

def init_ik_joint_poses(out_ik_joint_poses: List[Task2Pose], joint_name: List[str], path: List[int], joint_positions: np.ndarray, joint_orientations: np.ndarray):
    for i, joint_i in enumerate(path):
        joint_pose = Task2Pose(joint_i, joint_name[joint_i], joint_positions[joint_i], joint_orientations[joint_i])
        if i > 0:
            joint_pose.set_parent(out_ik_joint_poses[i - 1])
        out_ik_joint_poses.append(joint_pose)

def move_to_target_direction_rotation(current_direction: np.ndarray, target_direction: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    current_direction = current_direction / np.linalg.norm(current_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)
    
    dot = np.dot(current_direction, target_direction)
    
    if abs(1.0 - dot) < epsilon:
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    if abs(-1 - dot) < epsilon:
        axis = np.cross(current_direction, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis) < epsilon:
            axis = np.cross(current_direction, np.array([0.0, 0.0, 1.0]))
        axis = axis / np.linalg.norm(axis)
        theta = np.pi
        w = np.cos(theta / 2)
        x, y, z = axis * np.sin(theta / 2)
        return np.array([x, y, z, w])
    
    theta = np.arccos(dot)
    axis = np.cross(current_direction, target_direction)
    axis = axis / np.linalg.norm(axis)
    w = np.cos(theta / 2)
    x, y, z = axis * np.sin(theta / 2)
    return np.array([x, y, z, w])

def ccd_ik(ik_joint_poses: List[Task2Pose], target_position: np.ndarray, error_precision: float = 0.01, max_iterations: int = 16):
    def target_distance() -> float:
        return np.linalg.norm(target_position - ik_joint_poses[-1].get_global_position())

    iteration = 0
    while target_distance() >= error_precision and iteration < max_iterations:
        for joint_pose in reversed(ik_joint_poses[1:]):
            current_direction = ik_joint_poses[-1].get_global_position() - joint_pose.parent.get_global_position()
            target_direction = target_position - joint_pose.parent.get_global_position()
            rotation = move_to_target_direction_rotation(current_direction, target_direction)
            target_orientation = (R.from_quat(rotation) * R.from_quat(joint_pose.parent.get_global_orientation())).as_quat()
            joint_pose.parent.set_global_orientation(target_orientation)
        iteration += 1

def apply_ik_results(out_joint_positions: np.ndarray, out_joint_orientations: np.ndarray, joint_num: int, path: List[int], path1: List[int], path2: List[int], joint_poses: List[Task2Pose], ik_joint_poses: List[Task2Pose]):
    for joint_i in range(joint_num):
        if joint_i in path:
            joint = ik_joint_poses[path.index(joint_i)]
            if joint_i == 0:
                joint_poses[joint_i].set_global_position(joint.get_global_position())
                joint_poses[joint_i].set_global_orientation(joint.get_global_orientation())
            elif joint_i in path1:
                joint_poses[joint_i].set_global_position(joint.get_global_position())
                joint_poses[joint_i].set_global_orientation(joint.get_global_orientation())
            elif joint_i in path2:
                if joint_i == path2[0]:
                    joint_poses[joint_i].set_global_orientation(joint.get_global_orientation())
                else:
                    ik_joint_i_position = joint.get_global_position()
                    ik_joint_i_parent_position = joint.parent.get_global_position()
                    target_direction = ik_joint_i_parent_position - ik_joint_i_position
                    
                    ik_joint_i_original_position = out_joint_positions[joint_i]
                    ik_joint_i_parent_original_position = out_joint_positions[joint.parent.index]
                    original_direction = ik_joint_i_parent_original_position - ik_joint_i_original_position

                    rotation = move_to_target_direction_rotation(original_direction, target_direction)
                    joint_i_orientation = (R.from_quat(rotation) * R.from_quat(out_joint_orientations[joint_i])).as_quat()
                    joint_poses[joint_i].set_global_orientation(joint_i_orientation)
                    joint_poses[joint_i].set_global_position(joint.get_global_position())

    for joint_i in range(joint_num):
        out_joint_positions[joint_i] = joint_poses[joint_i].get_global_position()
        out_joint_orientations[joint_i] = joint_poses[joint_i].get_global_orientation()

def part1_inverse_kinematics(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations: np.ndarray, target_pose: np.ndarray):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    # ik path:
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    
    # joint info:
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_num = len(joint_name)

    joint_poses: List[Task2Pose] = []
    init_joint_poses(joint_poses, joint_num, joint_name, joint_parent, joint_positions, joint_orientations)

    ik_joint_poses: List[Task2Pose] = []
    init_ik_joint_poses(ik_joint_poses, joint_name, path, joint_positions, joint_orientations)

    # ik:
    ccd_ik(ik_joint_poses, target_pose)
    
    # apply ik results:
    apply_ik_results(joint_positions, joint_orientations, joint_num, path, path1, path2, joint_poses, ik_joint_poses)
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations: np.ndarray, relative_x: float, relative_z: float, target_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    # ik path:
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # joint info:
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_num = len(joint_name)

    joint_poses: List[Task2Pose] = []
    init_joint_poses(joint_poses, joint_num, joint_name, joint_parent, joint_positions, joint_orientations)

    ik_joint_poses: List[Task2Pose] = []
    init_ik_joint_poses(ik_joint_poses, joint_name, path, joint_positions, joint_orientations)

    # ik target:
    root_joint_position = joint_poses[0].get_global_position()
    ik_target = root_joint_position + np.array([relative_x, 0.0, relative_z], dtype=float)
    ik_target[1] = target_height

    # ik:
    ccd_ik(ik_joint_poses, ik_target)

    # apply ik results:
    apply_ik_results(joint_positions, joint_orientations, joint_num, path, path1, path2, joint_poses, ik_joint_poses)
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations