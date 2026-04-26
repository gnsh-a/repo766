"""Create a simple Blender scene for VisionSIM + DA3 integration testing.

Generates a .blend file with a floor, walls, and several primitive objects
(spheres, cubes, cylinders, cones) placed at varying depths from a fixed
camera. The scene includes a 2-keyframe camera animation so VisionSIM's
render pipeline succeeds.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    blender --background --python scripts/visionsim/create_simple_scene.py -- data/simple_scene.blend
"""

import math
import sys

import bpy


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    return mat


def create_scene():
    clear_scene()

    # --- Room (box: floor, ceiling, 3 walls — open behind camera) ---
    room_mat = make_material("RoomMat", (0.55, 0.50, 0.45, 1.0))
    floor_mat = make_material("FloorMat", (0.4, 0.4, 0.4, 1.0))

    # Floor
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 8, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(floor_mat)

    # Ceiling
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 8, 8))
    ceiling = bpy.context.active_object
    ceiling.name = "Ceiling"
    ceiling.data.materials.append(room_mat)

    # Back wall
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 20, 4))
    back = bpy.context.active_object
    back.name = "BackWall"
    back.rotation_euler = (math.radians(90), 0, 0)
    back.data.materials.append(room_mat)

    # Left wall
    bpy.ops.mesh.primitive_plane_add(size=30, location=(-10, 8, 4))
    left = bpy.context.active_object
    left.name = "LeftWall"
    left.rotation_euler = (0, math.radians(90), 0)
    left.data.materials.append(room_mat)

    # Right wall
    bpy.ops.mesh.primitive_plane_add(size=30, location=(10, 8, 4))
    right = bpy.context.active_object
    right.name = "RightWall"
    right.rotation_euler = (0, math.radians(90), 0)
    right.data.materials.append(room_mat)

    # --- Objects at varying depths ---
    objects_spec = [
        # (type, name, location, scale, color)
        # z = half-height so objects sit on floor (z=0)
        ("cube",     "RedCube",     ( 0.0,  3.0, 0.50), (1.0, 1.0, 1.0), (0.8, 0.15, 0.15, 1)),
        ("sphere",   "GreenSphere", (-2.0,  5.5, 0.60), (0.6, 0.6, 0.6), (0.15, 0.7, 0.15, 1)),
        ("cylinder", "BlueCyl",     ( 2.5,  7.0, 0.50), (0.5, 0.5, 1.0), (0.15, 0.15, 0.8, 1)),
        ("cube",     "YellowCube",  (-1.5, 10.0, 0.75), (1.5, 1.5, 1.5), (0.85, 0.85, 0.1, 1)),
        ("sphere",   "PinkSphere",  ( 1.5,  4.0, 0.40), (0.4, 0.4, 0.4), (0.85, 0.2, 0.6, 1)),
        ("cone",     "OrangeCone",  ( 3.0, 12.0, 1.125),(0.8, 0.8, 1.5), (0.9, 0.5, 0.1, 1)),
        ("cube",     "TealCube",    (-3.0,  8.0, 0.40), (0.8, 0.8, 0.8), (0.1, 0.7, 0.7, 1)),
    ]

    for obj_type, name, loc, scale, color in objects_spec:
        if obj_type == "cube":
            bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        elif obj_type == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc)
        elif obj_type == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, location=loc)
        elif obj_type == "cone":
            bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.5, location=loc)

        obj = bpy.context.active_object
        obj.name = name
        obj.scale = scale
        obj.data.materials.append(make_material(f"{name}Mat", color))

    # --- Lighting (inside the room) ---
    # Ceiling area light (pointing down)
    bpy.ops.object.light_add(type="AREA", location=(0, 8, 7.5))
    main_light = bpy.context.active_object
    main_light.name = "CeilingLight"
    main_light.data.energy = 1000.0
    main_light.data.size = 8.0
    main_light.rotation_euler = (math.radians(180), 0, 0)

    # Fill light (side, inside room)
    bpy.ops.object.light_add(type="AREA", location=(-5, 4, 5))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 400.0
    fill.data.size = 4.0
    fill.rotation_euler = (math.radians(90), 0, math.radians(30))

    # --- Camera (2 frames needed for VisionSIM, we only use the first) ---
    bpy.ops.object.camera_add(location=(0, -2, 2.5))
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.data.lens = 35
    camera.rotation_euler = (math.radians(72), 0, 0)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 2

    # Keyframe 1
    camera.keyframe_insert(data_path="location", frame=1)
    camera.keyframe_insert(data_path="rotation_euler", frame=1)
    # Keyframe 2: nudge camera slightly so VisionSIM sees motion
    camera.location = (0.01, -2, 2.5)
    camera.keyframe_insert(data_path="location", frame=2)
    camera.keyframe_insert(data_path="rotation_euler", frame=2)

    # --- Render settings ---
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.image_settings.file_format = "PNG"

    return scene


def main():
    # Parse output path from args after "--"
    argv = sys.argv
    try:
        idx = argv.index("--")
        output_path = argv[idx + 1]
    except (ValueError, IndexError):
        output_path = "data/simple_scene.blend"

    create_scene()
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Scene saved to {output_path}")


if __name__ == "__main__":
    main()
