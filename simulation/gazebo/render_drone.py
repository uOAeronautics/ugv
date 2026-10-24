"""Render the same drone parts used by Gazebo, including a UGV scale reference.

blender -b --python render_drone.py -- /absolute/output/directory
"""
from pathlib import Path
import sys, json, math, argparse
import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drone_model import drone_assembly, DRONE_COLORS
from assembly import assembly, COLORS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', nargs='?', type=Path)
parser.add_argument('--room', action='store_true')
parser.add_argument('--save-only', action='store_true', help='Build and save the scene without rendering')
parser.add_argument('--engine', choices=('CYCLES','BLENDER_EEVEE'), default='CYCLES')
parser.add_argument('--match-video', action='store_true', help='Render four user-anchored portrait views from the reference video')
parser.add_argument('--match-view', choices=('all','teaching-04','south-08','glazing-11','back-17'), default='all')
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if args.match_video:
    args.room = True
out = args.output.resolve() if args.output else HERE / 'preview' / ('room' if args.room else 'drone')
out.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

materials = {}
def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*rgb, 1)
    bs.inputs['Roughness'].default_value = .42
    if name == 'room_room_light':
        bs.inputs['Emission Color'].default_value = (*rgb, 1)
        bs.inputs['Emission Strength'].default_value = 2
    if any(word in name for word in ('silver', 'metal', 'aluminium')):
        bs.inputs['Metallic'].default_value = .8
        bs.inputs['Roughness'].default_value = .27
    if 'carbon' in name:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        tex = nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 175
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .18
        bump.inputs['Distance'].default_value = .0004
        links.new(tex.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bs.inputs['Normal'])
    return mat

color_sets = [('drone_', DRONE_COLORS), ('ugv_', COLORS)]
if args.room:
    from room_model import room_assembly, ROOM_COLORS
    room_parts = room_assembly()
    room_config = json.loads((HERE/'room_dimensions.json').read_text())
    color_sets.append(('room_', ROOM_COLORS))
for prefix, colors in color_sets:
    for name, rgb in colors.items():
        materials[prefix+name] = material(prefix+name, rgb)

mesh_cache = {}
def primitive_mesh(part, prefix):
    key = (part['shape'], tuple(part['size']), prefix+part['color'])
    if key in mesh_cache:
        return mesh_cache[key]
    mesh = bpy.data.meshes.new(part['name'].replace('.', '_'))
    if part['shape'] == 'box':
        x,y,z = [v/2 for v in part['size']]
        vertices = [(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),
                    (-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)]
        faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    else:
        r,h = part['size']
        vertices = [(r*math.cos(i*math.tau/32),r*math.sin(i*math.tau/32),z) for z in (-h/2,h/2) for i in range(32)]
        faces = [(i,(i+1)%32,(i+1)%32+32,i+32) for i in range(32)]
        faces += [tuple(reversed(range(32))),tuple(range(32,64))]
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(materials[prefix+part['color']])
    if part['shape'] == 'cylinder':
        for poly in mesh.polygons:
            poly.use_smooth = len(poly.vertices) == 4
    mesh_cache[key] = mesh
    return mesh

def add(part, prefix, offset=(0, 0, 0)):
    pos = Vector(part['pos']) + Vector(offset)
    if part['shape'] == 'mesh':
        key = (part['mesh'], prefix+part['color'])
        if key not in mesh_cache:
            vertices, faces = [], []
            for line in (HERE / part['mesh']).read_text().splitlines():
                values = line.split()
                if values and values[0] == 'v':
                    vertices.append(tuple(float(x) for x in values[1:4]))
                elif values and values[0] == 'f':
                    faces.append(tuple(int(x.split('/')[0])-1 for x in values[1:]))
            mesh = bpy.data.meshes.new(part['name'].replace('.', '_'))
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            mesh.materials.append(materials[prefix+part['color']])
            mesh_cache[key] = mesh
        ob = bpy.data.objects.new(part['name'].replace('.', '_'), mesh_cache[key])
        bpy.context.collection.objects.link(ob)
        ob.location = pos
        ob.rotation_euler = part['rot']
        objects = [ob]
    else:
        ob = bpy.data.objects.new(part['name'].replace('.', '_'), primitive_mesh(part, prefix))
        bpy.context.collection.objects.link(ob)
        ob.location = pos
        ob.rotation_euler = part['rot']
        objects = [ob]
    for ob in objects:
        ob.name = prefix+part['name'].replace('.', '_')
        if part['shape'] != 'mesh':
            bevel = ob.modifiers.new('Manufactured edge', 'BEVEL')
            bevel.width = .0004 if prefix == 'drone_' else .00018
            bevel.segments = 2
            ob.modifiers.new('Weighted normals', 'WEIGHTED_NORMAL')
        if part['shape'] == 'cylinder':
            for poly in ob.data.polygons:
                poly.use_smooth = len(poly.vertices) == 4
    return objects

room_objects = []
if args.room:
    for part in room_parts:
        for ob in add(part, 'room_'):
            room_objects.append((ob, part))
for part in drone_assembly():
    add(part, 'drone_')
if args.room:
    pose = room_config['drone_pose']
    bpy.context.view_layer.update()
    for ob in list(bpy.data.objects):
        if ob.name.startswith('drone_'):
            from mathutils import Matrix
            ob.matrix_world = Matrix.Translation(Vector(pose[:3])) @ Matrix.Rotation(pose[5],4,'Z') @ ob.matrix_world
p = json.loads((HERE / 'dimensions.json').read_text())
car_groups = assembly(p)
for group in car_groups.values():
    for part in group['parts']:
        add(part, 'ugv_', Vector(group['origin']) + Vector((0, 0, 0) if args.room else (-.40, 0, 0)))

# Passive mecanum rollers are separate SDF links rather than assembly parts.
# Use the simulator's own generator so the visual preview follows that model.
if any(part['name'] == 'hub_core' for group in car_groups.values() for part in group['parts']):
    import xml.etree.ElementTree as ET
    from mecanum_geometry import add_passive_rollers
    def xml_node(parent, tag, text=None, **attrs):
        element = ET.SubElement(parent, tag, attrs)
        if text is not None:
            element.text = str(text)
        return element
    temporary_model = ET.Element('model')
    add_passive_rollers(temporary_model, xml_node, car_groups, radius=p['wheel_diameter_in']*.0254/2)
    for link in temporary_model.findall('link'):
        pose = [float(x) for x in link.findtext('pose').split()]
        shape = link.find('visual/geometry/cylinder')
        part = dict(name=link.get('name'), shape='cylinder', pos=pose[:3], rot=pose[3:],
                    size=(float(shape.findtext('radius')),float(shape.findtext('length'))), color='rubber')
        add(part, 'ugv_', (0,0,0) if args.room else (-.40,0,0))
if args.room:
    pose = room_config['ugv_pose']
    bpy.context.view_layer.update()
    for ob in list(bpy.data.objects):
        if ob.name.startswith('ugv_'):
            from mathutils import Matrix
            ob.matrix_world = Matrix.Translation(Vector(pose[:3])) @ Matrix.Rotation(pose[5],4,'Z') @ ob.matrix_world

floor = material('Concrete', (.23, .25, .27))
if not args.room:
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -.001))
    bpy.context.object.data.materials.append(floor)
lights = [((x, y, 3.3), 300, 4) for x in (-3,3) for y in (-8,0,8)] if args.room else [((1.2, -1.4, 2.3), 180, 2), ((-.8, 1.1, 1.5), 120, 1.5), ((.5, .9, 2), 100, 1)]
for loc, power, size in lights:
    bpy.ops.object.light_add(type='AREA', location=loc)
    light = bpy.context.object
    light.data.energy = power
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (Vector((0, 0, .25))-light.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.object.camera_add()
cam = bpy.context.object
scene = bpy.context.scene
scene.camera = cam
scene.render.engine = args.engine
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
scene.cycles.diffuse_bounces = 3
scene.cycles.glossy_bounces = 2
scene.render.resolution_x = 1440
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.55, .62, .72, 1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .18 if args.room else .45
scene.view_settings.view_transform = 'AgX'
scene.view_settings.exposure = -.4 if args.room else 0
scene.unit_settings.system = 'METRIC'
if args.save_only:
    if args.room:
        cam.location = room_config['panorama_camera_estimate']
        pitch = math.radians(room_config['panorama_pitches_deg'][4])
        cam.rotation_euler = Vector((math.cos(pitch),0,math.sin(pitch))).to_track_quat('-Z','Y').to_euler()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'lab-docking.blend'))
    raise SystemExit(0)

def shot(name, loc, target, scale):
    cam.location = loc
    cam.rotation_euler = (Vector(target)-cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scale
    scene.render.filepath = str(out / (name+'.png'))
    bpy.ops.render.render(write_still=True)

if args.match_video:
    scene.render.resolution_x, scene.render.resolution_y = 576, 1024
    scene.cycles.samples = 24
    cam.data.type = 'PERSP'
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 36
    cam.data.lens = 37
    cam.location = room_config['panorama_camera_estimate']
    for name, second in [('teaching-04',4),('south-08',8),('glazing-11',11),('back-17',17)]:
        if args.match_view != 'all' and name != args.match_view:
            continue
        yaw = math.radians(room_config['panorama_yaws_deg'][second])
        pitch = math.radians(room_config['panorama_pitches_deg'][second])
        direction=Vector((math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)))
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(out / (name+'.png'))
        bpy.ops.render.render(write_still=True)
elif args.room:
    for ob, part in room_objects:
        x,y,z = part['pos']
        ob.hide_render = part.get('category') == 'ceiling' or (part.get('category') == 'wall' and (x < -5 or y < -11))
    shot('room-overview', (-18, -20, 19), (0, 0, .3), 29)
    for ob, part in room_objects:
        ob.hide_render = False
    cam.data.type = 'PERSP'
    cam.data.lens = 20
    cam.location = (2, -2, 1.62)
    cam.rotation_euler = (Vector((6, -2, 1.5))-cam.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath = str(out / 'room-interior.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'lab-docking.blend'))
    bpy.ops.render.render(write_still=True)
else:
    shot('drone-overview', (1.2, -1.5, 1.1), (-.05, 0, .28), 1.35)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'drone-docking.blend'))
    shot('drone-approach', (-1.4, -.4, .24), (0, 0, .23), .83)
    shot('drone-top', (0, 0, 2), (0, 0, 0), 1.3)
