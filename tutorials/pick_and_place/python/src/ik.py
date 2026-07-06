"""Inverse kinematics for a fixed-base serial arm, matching the joint chain
that RemoteJointController.cs drives in Unity.

ikpy solves kinematics in the URDF's native frame: ROS/FLU convention
(X-forward, Y-left, Z-up). Unity uses RUF (X-right, Y-up, Z-forward). The
conversions below match Unity.Robotics.ROSTCPConnector.ROSGeometry.FLU exactly,
so a pose computed here lines up with what RemoteJointController reports back
(expressed relative to base_link, in Unity's RUF frame). That part generalizes
to any robot imported via URDF-Importer with the standard ROS/Unity convention.

Chain structure and its assumption
-----------------------------------
Parsing a URDF from "base_link" with ikpy's auto-traversal always yields an
auto-inserted, inactive "Origin" link at index 0. This module additionally
assumes the path from base_link to the end effector is exactly NUM_JOINTS
active joints with nothing fixed interspersed between them, optionally
followed by one trailing fixed joint into the actual end-effector frame (as
niryo_one has via hand_tool_joint -> tool_link). That gives active_links_mask
= [False] + [True] * NUM_JOINTS + [False], which is why every `[1:NUM_JOINTS +
1]` slice elsewhere in this file pulls out exactly the active entries.

This holds for many simple serial arms, but isn't universal: a robot with a
fixed mounting-bracket link *between* two of its revolute joints would need
extra inactive entries in the middle of the mask, not just at the ends, and
this module doesn't handle that case. Branches off the main chain (e.g. a
gripper) aren't a problem in themselves -- see trim_urdf_subtree -- as long as
the arm's own path from base_link to the end effector stays a plain serial
run of NUM_JOINTS active joints.

load_arm_chain's num_active_joints lets a caller mark a trailing subset of
those NUM_JOINTS joints inactive too (e.g. to lock a wrist) -- see its
docstring. Everywhere else in this file still treats the chain as having
NUM_JOINTS joint values (`[1:NUM_JOINTS + 1]`); locked joints just always come
back unchanged from whatever was passed into solve_ik's initial_angles,
because ikpy holds inactive links fixed at their given initial value.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ikpy.chain import Chain

NUM_JOINTS = 6


def flu_to_ruf(v: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a position from URDF/ROS convention (FLU) to Unity (RUF)."""
    x, y, z = v
    return np.array([-y, z, x])


def ruf_to_flu(v: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a position from Unity (RUF) to URDF/ROS convention (FLU)."""
    x, y, z = v
    return np.array([z, -x, y])


def trim_urdf_subtree(urdf_path: str | Path, root_link_name: str) -> str:
    """Write a copy of the URDF with root_link_name and everything descending from
    it removed, and return its path.

    Useful for cutting off a branch (e.g. a gripper or sensor mount) that isn't
    part of the arm's own kinematic chain and that ikpy either doesn't need or
    can't parse (e.g. a malformed joint axis).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_by_parent = {}
    for joint in root.findall("joint"):
        joints_by_parent.setdefault(joint.find("parent").attrib["link"], []).append(joint)

    drop_joints = set()
    drop_links = set()
    frontier = [root_link_name]
    while frontier:
        link_name = frontier.pop()
        drop_links.add(link_name)
        for joint in joints_by_parent.get(link_name, []):
            child = joint.find("child").attrib["link"]
            drop_joints.add((joint.attrib["name"], link_name, child))
            frontier.append(child)

    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if child == root_link_name:
            drop_joints.add((joint.attrib["name"], parent, child))

    for joint in root.findall("joint"):
        key = (joint.attrib["name"], joint.find("parent").attrib["link"], joint.find("child").attrib["link"])
        if key in drop_joints:
            root.remove(joint)
    for link in root.findall("link"):
        if link.attrib["name"] in drop_links:
            root.remove(link)

    trimmed = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tree.write(trimmed.name)
    return trimmed.name


def load_arm_chain(urdf_path: str | Path, num_active_joints: int = NUM_JOINTS) -> Chain:
    """Parse urdf_path into an ikpy Chain, per this module's chain-structure assumption
    (see module docstring). Callers are responsible for pre-trimming any branches
    (e.g. via trim_urdf_subtree) that would otherwise confuse or derail the parse.

    num_active_joints lets the IK solver only move the first N of the NUM_JOINTS
    joints, holding the rest fixed at whatever's passed as initial_angles in
    solve_ik -- e.g. num_active_joints=3 locks the last 3 (a "fixed wrist": the
    first 3 joints position the arm, the wrist joints never move, so the end
    effector's orientation is just whatever falls out of that, not solved for).
    """
    active_links_mask = (
        [False] + [True] * num_active_joints + [False] * (NUM_JOINTS - num_active_joints) + [False]
    )
    return Chain.from_urdf_file(
        urdf_path,
        base_elements=["base_link"],
        active_links_mask=active_links_mask,
    )


def joint_bounds(chain: Chain) -> list[tuple[float, float]]:
    """Lower/upper bound (radians) for each of the 6 active joints, in order."""
    return [chain.links[i].bounds for i in range(1, NUM_JOINTS + 1)]


def angles_to_normalized(
    angles: npt.ArrayLike, bounds: list[tuple[float, float]]
) -> npt.NDArray[np.float32]:
    """Convert joint angles (radians) to the [-1, 1] range RemoteJointController expects."""
    normalized = [2.0 * (angle - lower) / (upper - lower) - 1.0 for angle, (lower, upper) in zip(angles, bounds)]
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def solve_ik(
    chain: Chain,
    target_position_flu: npt.ArrayLike,
    target_orientation_flu: npt.NDArray[np.float64] | None = None,
    initial_angles: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Returns the 6 active joint angles (radians) reaching target_position_flu (FLU frame),
    optionally also matching target_orientation_flu (a 3x3 rotation matrix, FLU frame)."""
    full_initial = np.zeros(len(chain.links))
    if initial_angles is not None:
        full_initial[1:NUM_JOINTS + 1] = initial_angles

    kwargs = {}
    if target_orientation_flu is not None:
        kwargs["target_orientation"] = target_orientation_flu
        kwargs["orientation_mode"] = "all"

    full_solution = chain.inverse_kinematics(
        target_position=target_position_flu,
        initial_position=full_initial,
        **kwargs,
    )
    return full_solution[1:NUM_JOINTS + 1]


def forward_kinematics_flu(chain: Chain, angles: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Position (FLU frame) of the end effector for the given 6 active joint angles."""
    full_angles = np.zeros(len(chain.links))
    full_angles[1:NUM_JOINTS + 1] = angles
    return chain.forward_kinematics(full_angles)[:3, 3]
