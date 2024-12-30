from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path: str) -> Tuple[List[str], List[int], np.ndarray]:
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    bvh_file = open(bvh_file_path, "r")
    lines = bvh_file.readlines()
    bvh_file.close()

    # results:
    joint_name: List[str] = []
    joint_parent: List[str] = []
    joint_offset: List[Tuple[float, float, float]] = []

    # helpers:
    joint_stack: List[str] = []
    joint_index: Dict[str, int] = {}

    def parse_line(line: str) -> List[str]:
        return line.split()

    def parse_joint(lines: List[str], line_idx: int) -> int:
        name_items = parse_line(lines[line_idx])
        offset_items = parse_line(lines[line_idx + 2])

        name = name_items[1] if name_items[0] != "End" else joint_stack[-1] + "_end"
        parent = joint_index[joint_stack[-1]] if len(joint_stack) > 0 else -1
        offset = (float(offset_items[1]), float(offset_items[2]), float(offset_items[3]))

        joint_name.append(name)
        joint_parent.append(parent)
        joint_offset.append(offset)

        joint_stack.append(name)
        joint_index[name] = len(joint_name) - 1

        return line_idx + 4 if name_items[0] != "End" else line_idx + 3

    # parse:
    i: int = 0
    while i < len(lines):
        items = parse_line(lines[i])

        if items[0] == "ROOT" or items[0] == "JOINT" or items[0] == "End":
            i = parse_joint(lines, i)
        elif items[0] == "}":
            joint_stack.pop()
            i += 1
        else:
            i += 1
    
    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
