"""Build a back-left-corner tabletop cliff scene for VisionSIM + DA3 eval.

The table is pushed into the back-left corner of the room so its left edge
meets the left wall (no cliff there) and only the front (+Y) and right (+X
relative to the table) edges are real depth discontinuities. Camera hugs the
right cliff and drives toward the front cliff.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    blender --background --python scripts/visionsim/scenes/tabletop_corner.py -- \\
        data/sim_dataset/tabletop_corner/scene/scene.blend
"""

import json
import math
import os
import sys

# Make scripts/visionsim/ importable so we can pull in trajectories.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy

from trajectories import head_on_approach


# Scene constants (meters), in the robot/table coordinate frame
TABLE_Z = 0.0
FLOOR_Z = -0.75
CEILING_Z = 2.0
ROOM_HALF_W = 3.0
ROOM_Y_BACK = -0.5
ROOM_Y_FRONT = 6.0

# Table is in the back-left corner: left edge against the left wall, back edge
# against the back wall. Cliffs are on the front and right edges only.
TABLE_X_LEFT = -ROOM_HALF_W   # flush with left wall, no left cliff
TABLE_X_RIGHT = 0.0           # right cliff at x = 0
TABLE_Y_BACK = ROOM_Y_BACK    # flush with back wall
CLIFF_Y = 1.5                 # front cliff
CLIFF_X = TABLE_X_RIGHT       # right cliff

# (shape, name, x, y, z, sx, sy, sz, r, g, b) — referenced by add_distractors() and the meta sidecar
DISTRACTORS = [
    # On the table, off to the left so it doesn't collide with the camera path
    ("cylinder", "BlueCyl",        -1.05, 0.90,  0.10, 0.08, 0.08, 0.20, 0.20, 0.30, 0.85),
    # On the floor in the front-right corner, past both the front cliff and
    # (well past) the right cliff
    ("cube",     "WarehouseCrate",  2.85, 5.85, -0.60, 0.30, 0.30, 0.30, 0.55, 0.40, 0.20),
]

# Camera: start at the back-right corner of the table (just inside the right
# cliff) and drive forward along +Y parallel to the right cliff.
CAMERA_X_OFFSET = -0.10   # 10 cm inside the right cliff at x = 0
CAMERA_START_Y = -0.35    # 0.15 m inside the back edge of the 2 m-deep table
CAMERA_DISTANCE = 1.75    # travel along Y, ends at (-0.10, +1.40)
CAMERA_HEIGHT = 0.15
CAMERA_PITCH_DEG = -25.0
CAMERA_SPEED_MPS = 0.3
SCENE_FPS = 25


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def make_material(name, color, roughness=0.7):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_plane(name, location, scale_x, scale_y, rotation_euler, material):
    bpy.ops.mesh.primitive_plane_add(size=1, location=location)
    plane = bpy.context.active_object
    plane.name = name
    plane.scale = (scale_x, scale_y, 1.0)
    plane.rotation_euler = rotation_euler
    plane.data.materials.append(material)
    return plane


def build_room():
    table_mat = make_material("TableMat", (0.45, 0.25, 0.10, 1.0), roughness=0.6)
    floor_mat = make_material("FloorMat", (0.18, 0.18, 0.20, 1.0), roughness=0.9)
    wall_mat = make_material("WallMat", (0.60, 0.58, 0.55, 1.0), roughness=0.9)
    ceil_mat = make_material("CeilingMat", (0.80, 0.80, 0.80, 1.0), roughness=0.95)

    # Table: occupies the back-left corner, ends at the front and right cliffs
    table_x_size = TABLE_X_RIGHT - TABLE_X_LEFT
    table_x_center = (TABLE_X_RIGHT + TABLE_X_LEFT) / 2.0
    table_y_size = CLIFF_Y - TABLE_Y_BACK
    table_y_center = (CLIFF_Y + TABLE_Y_BACK) / 2.0
    make_plane(
        "Table", (table_x_center, table_y_center, TABLE_Z),
        scale_x=table_x_size, scale_y=table_y_size,
        rotation_euler=(0, 0, 0), material=table_mat,
    )

    # Floor: full room footprint, including the area past the cliffs
    room_y_size = ROOM_Y_FRONT - ROOM_Y_BACK
    room_y_center = (ROOM_Y_FRONT + ROOM_Y_BACK) / 2.0
    make_plane(
        "Floor", (0.0, room_y_center, FLOOR_Z),
        scale_x=2 * ROOM_HALF_W, scale_y=room_y_size,
        rotation_euler=(0, 0, 0), material=floor_mat,
    )

    wall_z_size = CEILING_Z - FLOOR_Z
    wall_z_center = (CEILING_Z + FLOOR_Z) / 2.0
    make_plane(
        "WallLeft", (-ROOM_HALF_W, room_y_center, wall_z_center),
        scale_x=wall_z_size, scale_y=room_y_size,
        rotation_euler=(0, math.radians(90), 0), material=wall_mat,
    )
    make_plane(
        "WallRight", (ROOM_HALF_W, room_y_center, wall_z_center),
        scale_x=wall_z_size, scale_y=room_y_size,
        rotation_euler=(0, math.radians(90), 0), material=wall_mat,
    )
    make_plane(
        "WallBack", (0.0, ROOM_Y_BACK, wall_z_center),
        scale_x=2 * ROOM_HALF_W, scale_y=wall_z_size,
        rotation_euler=(math.radians(90), 0, 0), material=wall_mat,
    )
    make_plane(
        "WallFront", (0.0, ROOM_Y_FRONT, wall_z_center),
        scale_x=2 * ROOM_HALF_W, scale_y=wall_z_size,
        rotation_euler=(math.radians(90), 0, 0), material=wall_mat,
    )

    make_plane(
        "Ceiling", (0.0, room_y_center, CEILING_Z),
        scale_x=2 * ROOM_HALF_W, scale_y=room_y_size,
        rotation_euler=(math.radians(180), 0, 0), material=ceil_mat,
    )


def add_distractors():
    for shape, name, x, y, z, sx, sy, sz, r, g, b in DISTRACTORS:
        loc = (x, y, z)
        if shape == "cube":
            bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        elif shape == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, location=loc)
        elif shape == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (sx, sy, sz)
        obj.data.materials.append(make_material(f"{name}Mat", (r, g, b, 1.0)))


def add_lighting():
    # Centered roughly over the table for even illumination of the corner setup
    bpy.ops.object.light_add(type="AREA", location=(-1.0, 1.0, CEILING_Z - 0.05))
    main = bpy.context.active_object
    main.name = "CeilingLight"
    main.data.energy = 250.0
    main.data.size = 3.0
    main.rotation_euler = (math.radians(180), 0, 0)


def setup_camera():
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.15))
    cam = bpy.context.active_object
    cam.name = "Camera"

    cam.data.sensor_width = 36.0
    cam.data.lens = 36.0 / (2.0 * math.tan(math.radians(70.0) / 2.0))
    cam.data.clip_start = 0.05
    cam.data.clip_end = 10.0

    bpy.context.scene.camera = cam
    return cam


def keyframe_camera(camera, keyframes):
    for frame, location, rotation in keyframes:
        camera.location = location
        camera.rotation_euler = rotation
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 25
    scene.view_settings.view_transform = "Filmic"


def build_scene():
    clear_scene()
    build_room()
    add_distractors()
    add_lighting()
    cam = setup_camera()
    keyframes = head_on_approach(
        x_offset=CAMERA_X_OFFSET,
        start_y=CAMERA_START_Y,
        distance_to_edge=CAMERA_DISTANCE,
        camera_height=CAMERA_HEIGHT,
        pitch_deg=CAMERA_PITCH_DEG,
        speed=CAMERA_SPEED_MPS,
        fps=SCENE_FPS,
    )
    keyframe_camera(cam, keyframes)
    scene = bpy.context.scene
    scene.frame_start = keyframes[0][0]
    scene.frame_end = keyframes[-1][0]
    configure_render()


def write_meta_sidecar(blend_path):
    meta_path = os.path.splitext(blend_path)[0] + ".meta.json"
    cam_y_end = CAMERA_START_Y + CAMERA_DISTANCE
    meta = {
        "scene_name": "tabletop_corner",
        "fps": SCENE_FPS,
        "units": "meters",
        "cliff_axis": "y",
        "cliff_position": CLIFF_Y,
        "cliff_drop": TABLE_Z - FLOOR_Z,
        "trajectory": "head_on_approach",
        "trajectory_params": {
            "distance_to_edge": CLIFF_Y - CAMERA_START_Y,
            "speed_mps": CAMERA_SPEED_MPS,
            "x_offset": CAMERA_X_OFFSET,
            "start_y": CAMERA_START_Y,
        },
        "layout": {
            "table": {"x": [TABLE_X_LEFT, TABLE_X_RIGHT],
                      "y": [TABLE_Y_BACK, CLIFF_Y], "z": TABLE_Z},
            "room":  {"x": [-ROOM_HALF_W, ROOM_HALF_W], "y": [ROOM_Y_BACK, ROOM_Y_FRONT],
                      "z": [FLOOR_Z, CEILING_Z]},
            "cliffs": [
                {"name": "front", "axis": "y", "value": CLIFF_Y,
                 "range": [TABLE_X_LEFT, TABLE_X_RIGHT]},
                {"name": "right", "axis": "x", "value": CLIFF_X,
                 "range": [TABLE_Y_BACK, CLIFF_Y]},
            ],
            "distractors": [
                {"name": n, "x": x, "y": y, "z": z, "color": [r, g, b]}
                for (_, n, x, y, z, _, _, _, r, g, b) in DISTRACTORS
            ],
            "camera_path": {
                "start": [CAMERA_X_OFFSET, CAMERA_START_Y],
                "end":   [CAMERA_X_OFFSET, cam_y_end],
            },
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta sidecar saved to {meta_path}")


def render_topdown(blend_path):
    import subprocess
    scene_dir = os.path.dirname(os.path.abspath(blend_path))
    helper = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scene_topdown.py")
    if not os.path.exists(helper):
        print(f"WARN: scene_topdown.py not found at {helper}; skipping diagram")
        return
    try:
        subprocess.run(["python", helper, scene_dir], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARN: top-down generation failed ({e}); scene.blend + meta still saved")


def main():
    argv = sys.argv
    try:
        idx = argv.index("--")
        output_path = argv[idx + 1]
    except (ValueError, IndexError):
        output_path = "data/sim_dataset/tabletop_corner/scene/scene.blend"

    build_scene()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Scene saved to {output_path}")
    write_meta_sidecar(output_path)
    render_topdown(output_path)


if __name__ == "__main__":
    main()
