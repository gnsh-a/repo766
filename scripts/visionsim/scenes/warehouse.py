"""Realistic warehouse scene for VisionSIM + DA3 eval.

Builds on the tabletop_cliff geometry (same dock-edge cliff, same camera),
but swaps the empty room for a warehouse interior:
  - Procedural concrete floor (noise + bump for surface detail)
  - 2x3 grid of linear high-bay fluorescent fixtures replacing the bounce light
  - 4 stacked wooden pallets in the warehouse zone
  - 2 wall-mounted shelving units along the side walls (with crates on shelves)

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    blender --background --python scripts/visionsim/scenes/warehouse.py -- \\
        data/sim_dataset/warehouse/scene/scene.blend

    # Render a single preview frame (no animation):
    blender --background --python scripts/visionsim/scenes/warehouse.py -- \\
        data/sim_dataset/warehouse/scene/scene.blend --render-frame 30

    # Or just regenerate the topdown:
    python scripts/visionsim/scenes/warehouse.py --topdown-only
"""

import json
import math
import os
import sys

# Make scripts/visionsim/ importable so we can pull in trajectories.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bpy is only available inside Blender.
try:
    import bpy
except ImportError:
    bpy = None

from trajectories import head_on_approach


# Scene constants (meters), in the robot/dock coordinate frame
TABLE_Z = 0.0          # dock-floor height
FLOOR_Z = -0.75        # truck-bay / warehouse-floor height
CLIFF_Y = 1.5          # front edge of dock
ROOM_HALF_W = 3.0
ROOM_Y_BACK = 0.0      # back wall sits on world Y origin
ROOM_Y_FRONT = 8.5
CEILING_Z = 2.5        # taller for warehouse high-bay feel
TABLE_HALF_W = 1.5

# (shape, name, x, y, z, sx, sy, sz, r, g, b)
DISTRACTORS = [
    # Crate on the warehouse floor past the front cliff
    ("cube", "WarehouseCrate", 2.85, 6.35, -0.60, 0.30, 0.30, 0.30, 0.55, 0.40, 0.20),
]

# (x, y, top_z, height, color) — wooden pallets / pallet stacks in the warehouse
# zone. Each entry is a single cuboid object; multi-pallet stacks are modeled
# as one taller cuboid (e.g., a 0.30 m tall block representing two stacked
# 0.15 m pallets) rather than separate overlapping pieces.
PALLET_STACKS = [
    (-1.50, 5.50, FLOOR_Z + 0.15, 0.15, (0.55, 0.40, 0.22)),  # 1-pallet
    (1.50, 4.30, FLOOR_Z + 0.30, 0.30, (0.55, 0.40, 0.22)),   # 2-pallet stack (single object)
]

CAMERA_X_OFFSET = 1.40
CAMERA_START_Y = 0.15
CAMERA_DISTANCE = 1.25
CAMERA_HEIGHT = 0.15
CAMERA_PITCH_DEG = -25.0
CAMERA_SPEED_MPS = 0.2
SCENE_FPS = 25


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def make_simple_material(name, color, roughness=0.7):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_concrete_material(name, color_dark=(0.16, 0.16, 0.18),
                            color_light=(0.36, 0.36, 0.38),
                            roughness=0.85, bump_strength=0.25):
    """Procedural concrete: noise-driven color variation + bump.

    Color shifts between `color_dark` and `color_light` via a low-frequency
    Noise -> Color Ramp; surface detail comes from a high-frequency Noise -> Bump.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.6

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (*color_dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (*color_light, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump_noise = nodes.new("ShaderNodeTexNoise")
    bump_noise.inputs["Scale"].default_value = 80.0
    bump_noise.inputs["Detail"].default_value = 6.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    links.new(bump_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_wall_material(name):
    """Slightly noisy painted-block wall — beige base with low-amplitude variation."""
    return make_concrete_material(
        name,
        color_dark=(0.55, 0.53, 0.50),
        color_light=(0.66, 0.63, 0.58),
        roughness=0.9, bump_strength=0.10,
    )


def make_plane(name, location, scale_x, scale_y, rotation_euler, material):
    bpy.ops.mesh.primitive_plane_add(size=1, location=location)
    plane = bpy.context.active_object
    plane.name = name
    plane.scale = (scale_x, scale_y, 1.0)
    plane.rotation_euler = rotation_euler
    plane.data.materials.append(material)
    return plane


def make_cube(name, location, size, color, roughness=0.7):
    """Place a cuboid centered at `location` with full extents `size=(sx,sy,sz)`.

    primitive_cube_add(size=1) creates a unit cube with edge length 1 (vertices
    at ±0.5), so scaling by `size` directly produces a cuboid with extents
    matching `size`.
    """
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    obj.data.materials.append(make_simple_material(f"{name}Mat", (*color, 1.0), roughness))
    return obj


def build_room():
    # Sealed industrial concrete dock: warmer/lighter than the warehouse floor
    # (concrete that's been polished/sealed under foot traffic).
    dock_mat = make_concrete_material(
        "DockMat",
        color_dark=(0.30, 0.28, 0.26),
        color_light=(0.50, 0.46, 0.42),
        roughness=0.65, bump_strength=0.15,
    )
    floor_mat = make_concrete_material("FloorMat")
    wall_mat = make_wall_material("WallMat")
    ceil_mat = make_simple_material("CeilingMat", (0.78, 0.78, 0.80, 1.0), roughness=0.95)

    # Dock surface (formerly "table"): ends at the cliff
    dock_y_size = CLIFF_Y - ROOM_Y_BACK
    dock_y_center = (CLIFF_Y + ROOM_Y_BACK) / 2.0
    make_plane(
        "Dock", (0.0, dock_y_center, TABLE_Z),
        scale_x=2 * TABLE_HALF_W, scale_y=dock_y_size,
        rotation_euler=(0, 0, 0), material=dock_mat,
    )

    # Concrete warehouse floor
    room_y_size = ROOM_Y_FRONT - ROOM_Y_BACK
    room_y_center = (ROOM_Y_FRONT + ROOM_Y_BACK) / 2.0
    make_plane(
        "Floor", (0.0, room_y_center, FLOOR_Z),
        scale_x=2 * ROOM_HALF_W, scale_y=room_y_size,
        rotation_euler=(0, 0, 0), material=floor_mat,
    )

    wall_z_size = CEILING_Z - FLOOR_Z
    wall_z_center = (CEILING_Z + FLOOR_Z) / 2.0
    for name, loc, sx, sy, rot in [
        ("WallLeft",  (-ROOM_HALF_W, room_y_center, wall_z_center),
         wall_z_size, room_y_size, (0, math.radians(90), 0)),
        ("WallRight", (ROOM_HALF_W, room_y_center, wall_z_center),
         wall_z_size, room_y_size, (0, math.radians(90), 0)),
        ("WallBack",  (0.0, ROOM_Y_BACK, wall_z_center),
         2 * ROOM_HALF_W, wall_z_size, (math.radians(90), 0, 0)),
        ("WallFront", (0.0, ROOM_Y_FRONT, wall_z_center),
         2 * ROOM_HALF_W, wall_z_size, (math.radians(90), 0, 0)),
    ]:
        make_plane(name, loc, sx, sy, rot, wall_mat)

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
        obj.data.materials.append(make_simple_material(f"{name}Mat", (r, g, b, 1.0)))


def add_pallets():
    # Standard pallet footprint: 1.2 m x 1.0 m. Height varies (1-pallet vs stack).
    PALLET_FOOTPRINT = (1.2, 1.0)
    for i, (x, y, top_z, height, color) in enumerate(PALLET_STACKS):
        center_z = top_z - height / 2.0
        size = (PALLET_FOOTPRINT[0], PALLET_FOOTPRINT[1], height)
        make_cube(f"Pallet_{i}", (x, y, center_z), size, color, roughness=0.8)


def add_shelving(name, x_post, y_range, levels=(0.45, 1.0, 1.55),
                 shelf_depth=0.5, post_thickness=0.06):
    """Wall-mounted shelving: vertical posts + horizontal shelves + a few crates.

    `x_post` is the x-coord of the wall-side face of the posts (positive for
    the right wall, negative for the left wall). Shelves extend `shelf_depth`
    away from the wall (into the room).
    """
    y0, y1 = y_range
    sign = 1.0 if x_post < 0 else -1.0  # which way the shelves extend
    shelf_x_far = x_post + sign * shelf_depth
    shelf_x_center = (x_post + shelf_x_far) / 2.0
    metal_color = (0.45, 0.45, 0.50)

    # Vertical posts at both ends and one in the middle
    post_h = max(levels) + 0.02
    post_z = FLOOR_Z + post_h / 2.0
    for j, y_post in enumerate([y0, (y0 + y1) / 2.0, y1]):
        for x_off in (0.0, sign * (shelf_depth - post_thickness)):
            make_cube(f"{name}_Post_{j}_{int(x_off*100)}",
                      (x_post + x_off, y_post, post_z),
                      (post_thickness, post_thickness, post_h),
                      metal_color, roughness=0.5)

    # Horizontal shelf planks
    shelf_thickness = 0.04
    shelf_length = y1 - y0
    for k, z_level in enumerate(levels):
        make_cube(f"{name}_Shelf_{k}",
                  (shelf_x_center, (y0 + y1) / 2.0, FLOOR_Z + z_level + shelf_thickness / 2.0),
                  (shelf_depth, shelf_length, shelf_thickness),
                  metal_color, roughness=0.5)

    # Crates spread across two shelves so neither is empty.
    crate_color = (0.55, 0.40, 0.22)
    crate_size = (0.4, 0.4, 0.4)
    for level_idx, y_crates in [
        (0, [y0 + 1.0, (y0 + y1) / 2.0, y1 - 1.0]),  # bottom shelf, 3 crates
        (1, [y0 + 1.5, y1 - 1.5]),                    # middle shelf, 2 crates
    ]:
        crate_z = FLOOR_Z + levels[level_idx] + shelf_thickness + crate_size[2] / 2.0
        for k, y_crate in enumerate(y_crates):
            make_cube(f"{name}_Crate_L{level_idx}_{k}",
                      (shelf_x_center, y_crate, crate_z),
                      crate_size, crate_color, roughness=0.7)


SHELF_Y_RANGE = (ROOM_Y_FRONT - 5.0, ROOM_Y_FRONT - 0.05)
# Shelves anchored at the top corner of the warehouse (front wall) and extend
# back toward the camera 5 m along each side wall.


def add_warehouse_props():
    add_pallets()
    # Two wall-mounted shelving units along the side walls in the warehouse zone
    add_shelving("ShelfLeft",  x_post=-ROOM_HALF_W + 0.02, y_range=SHELF_Y_RANGE)
    add_shelving("ShelfRight", x_post=ROOM_HALF_W - 0.02,  y_range=SHELF_Y_RANGE)


# Rectangular high-bay fixtures (linear fluorescent), arranged in a 2 x 3 grid.
HIGH_BAY_FIXTURES = [
    {"x": x, "y": y} for y in (1.5, 4.5, 7.5) for x in (-1.5, 1.5)
]
FIXTURE_SIZE_X = 0.35  # short axis (lamp width)
FIXTURE_SIZE_Y = 1.50  # long axis (length of the tube housing)
FIXTURE_ENERGY = 70.0  # per fixture; total ~420 W
HOUSING_THICKNESS = 0.06


def add_lighting():
    """6 linear high-bay fixtures: each is an area light + a dark housing
    cuboid below the ceiling so the fixture has a physical body the camera
    can see (and that catches indirect light)."""
    housing_mat = make_simple_material(
        "FixtureHousing", (0.05, 0.05, 0.05, 1.0), roughness=0.4)
    for i, f in enumerate(HIGH_BAY_FIXTURES):
        # Housing cuboid hanging from the ceiling
        housing_z = CEILING_Z - HOUSING_THICKNESS / 2.0
        bpy.ops.mesh.primitive_cube_add(size=1, location=(f["x"], f["y"], housing_z))
        housing = bpy.context.active_object
        housing.name = f"HighBayHousing_{i}"
        housing.scale = (FIXTURE_SIZE_X / 2.0 + 0.04,
                         FIXTURE_SIZE_Y / 2.0 + 0.04,
                         HOUSING_THICKNESS / 2.0)
        housing.data.materials.append(housing_mat)

        # Light source just below the housing's bottom face
        bpy.ops.object.light_add(
            type="AREA",
            location=(f["x"], f["y"], CEILING_Z - HOUSING_THICKNESS - 0.005))
        light = bpy.context.active_object
        light.name = f"HighBay_{i}"
        light.data.shape = "RECTANGLE"
        light.data.size = FIXTURE_SIZE_X
        light.data.size_y = FIXTURE_SIZE_Y
        light.data.energy = FIXTURE_ENERGY
        # Default area-light emission is along -Z (downward) — leave rotation at 0.


def setup_camera():
    bpy.ops.object.camera_add(location=(0.0, 0.0, CAMERA_HEIGHT))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.data.sensor_width = 36.0
    cam.data.lens = 36.0 / (2.0 * math.tan(math.radians(70.0) / 2.0))
    cam.data.clip_start = 0.05
    cam.data.clip_end = 15.0
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
    scene.render.fps = SCENE_FPS
    scene.view_settings.view_transform = "Filmic"


def build_scene():
    clear_scene()
    build_room()
    add_distractors()
    add_warehouse_props()
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
        "scene_name": "warehouse",
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
            "table": {"x": [-TABLE_HALF_W, TABLE_HALF_W], "y": [ROOM_Y_BACK, CLIFF_Y], "z": TABLE_Z},
            "room":  {"x": [-ROOM_HALF_W, ROOM_HALF_W], "y": [ROOM_Y_BACK, ROOM_Y_FRONT],
                      "z": [FLOOR_Z, CEILING_Z]},
            "cliffs": [
                {"name": "front", "axis": "y", "value": CLIFF_Y,
                 "range": [-TABLE_HALF_W, TABLE_HALF_W]},
                {"name": "right", "axis": "x", "value": TABLE_HALF_W,
                 "range": [ROOM_Y_BACK, CLIFF_Y]},
            ],
            "distractors": [
                {"name": n, "x": x, "y": y, "z": z, "color": [r, g, b]}
                for (_, n, x, y, z, _, _, _, r, g, b) in DISTRACTORS
            ] + [
                # Pallets render as small wood-colored markers in the topdown
                {"name": f"Pallet_{i}", "x": x, "y": y, "z": top_z, "color": list(color)}
                for i, (x, y, top_z, _, color) in enumerate(PALLET_STACKS)
            ],
            "shelves": [
                {"name": "ShelfLeft", "x": [-ROOM_HALF_W, -ROOM_HALF_W + 0.5],
                 "y": list(SHELF_Y_RANGE)},
                {"name": "ShelfRight", "x": [ROOM_HALF_W - 0.5, ROOM_HALF_W],
                 "y": list(SHELF_Y_RANGE)},
            ],
            "lights": [
                {"name": f"HighBay_{i}", "type": "AREA",
                 "x": [f["x"] - FIXTURE_SIZE_X / 2, f["x"] + FIXTURE_SIZE_X / 2],
                 "y": [f["y"] - FIXTURE_SIZE_Y / 2, f["y"] + FIXTURE_SIZE_Y / 2],
                 "z": CEILING_Z, "energy_w": FIXTURE_ENERGY}
                for i, f in enumerate(HIGH_BAY_FIXTURES)
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
        return
    try:
        subprocess.run(["python", helper, scene_dir], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARN: top-down generation failed ({e})")


def render_single_frame(blend_path, frame_index, png_path):
    """Set the scene to `frame_index` and render a single PNG for preview."""
    scene = bpy.context.scene
    scene.frame_set(frame_index)
    scene.render.filepath = os.path.abspath(png_path)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"Preview frame {frame_index} saved to {png_path}")


def main():
    argv = sys.argv
    post = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]

    output_path = "data/sim_dataset/warehouse/scene/scene.blend"
    topdown_only = False
    render_frames = []
    it = iter(post)
    for a in it:
        if a == "--topdown-only":
            topdown_only = True
        elif a == "--render-frame":
            render_frames.extend(int(x) for x in next(it).split(","))
        else:
            output_path = a

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if topdown_only:
        write_meta_sidecar(output_path)
        render_topdown(output_path)
        return

    if bpy is None:
        sys.exit("Needs Blender (bpy). Run via `blender --background --python ...` "
                 "or pass --topdown-only.")
    build_scene()
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Scene saved to {output_path}")
    write_meta_sidecar(output_path)
    render_topdown(output_path)

    for f in render_frames:
        png_path = os.path.join(os.path.dirname(output_path), f"preview_frame_{f:03d}.png")
        render_single_frame(output_path, f, png_path)


if __name__ == "__main__":
    main()
