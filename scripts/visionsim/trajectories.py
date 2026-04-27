"""Camera trajectory generators for VisionSIM scenes (pure Python, no bpy).

Each generator returns a list of (frame, location, rotation_euler) tuples in
world coordinates. World frame: +Y forward, +Z up.

Author: Ganesh Arivoli <arivoli@wisc.edu>
"""

import math


def head_on_approach(
    *,
    distance_to_edge: float = 1.5,
    speed: float = 0.3,
    fps: int = 25,
    camera_height: float = 0.15,
    pitch_deg: float = -15.0,
    start_frame: int = 1,
    x_offset: float = 0.0,
    start_y: float = 0.0,
):
    """Straight forward approach toward a cliff edge along +Y.

    Camera translates +Y by `distance_to_edge` meters, starting at
    (x_offset, start_y, camera_height). Pitch is constant; no roll, no yaw.
    Setting `x_offset` near a side cliff makes the path run parallel to that
    edge (e.g. start at the back-right corner of the table).

    Args:
        distance_to_edge: meters traveled forward (start_y -> start_y + this).
        speed: forward velocity in m/s.
        fps: scene framerate; with `speed` sets the per-frame translation.
        camera_height: meters above the surface the camera is mounted on.
        pitch_deg: downward pitch from horizontal (negative = looking down).
        start_frame: frame index of the first keyframe.
        x_offset: lateral offset of the path (use to hug a side edge).
        start_y: starting y; lets the path begin at the back of the table
            instead of mid-table.

    Returns:
        list of (frame, (x, y, z), (rx, ry, rz)) tuples; rotations in radians.
    """
    step_m = speed / fps
    n_frames = int(round(distance_to_edge / step_m)) + 1

    # Blender's default camera looks along -Z. Rotating +90deg around X aligns
    # its view direction with +Y. Adding `pitch_deg` (negative for downward)
    # then tilts it the requested amount below horizontal.
    rx = math.radians(90.0 + pitch_deg)
    rotation = (rx, 0.0, 0.0)

    keyframes = []
    for i in range(n_frames):
        y = start_y + i * step_m
        location = (x_offset, y, camera_height)
        keyframes.append((start_frame + i, location, rotation))
    return keyframes
