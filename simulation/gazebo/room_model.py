"""Photo-informed university robotics workshop. SI units, floor top Z=0.

Panorama suggests an open central demonstration floor surrounded by teaching,
computing and fabrication zones. Object placements are estimates, not a survey.
"""
import json
import math
import os
import tempfile
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _atomic_text(path, text):
    """Publish complete mesh bytes without racing concurrent scene readers."""
    if path.exists() and path.read_text() == text:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

ROOM_COLORS = {
    'room_concrete': (.22,.215,.20), 'room_ceiling': (.18,.19,.20), 'room_floor_light': (.25,.245,.23),
    'room_floor_dark': (.19,.185,.17), 'room_wall': (.63,.63,.59),
    'room_trim': (.15,.16,.16), 'room_white': (.91,.92,.89),
    'room_metal': (.43,.46,.46), 'room_dark': (.065,.075,.08),
    'room_glass': (.62,.72,.68), 'room_glass_light': (.69,.77,.70),
    'room_wood': (.70,.55,.33), 'room_red': (.65,.015,.035),
    'room_blue': (.025,.18,.37), 'room_exit_green': (.01,.7,.33),
    'room_light': (.98,.98,.88), 'room_ink': (.22,.30,.32),
    'room_floor_tape': (.58,.10,.08), 'room_silver': (.67,.70,.69),
}

def _authored_room_assembly():
    parts=[]
    def box(n,p,s,c='room_wall',r=(0,0,0),collision=False,category='prop'):
        parts.append(dict(name=n,shape='box',pos=tuple(p),size=tuple(s),color=c,
                          rot=tuple(r),collision=collision,category=category))
    def cyl(n,p,rad,h,c='room_metal',r=(0,0,0),collision=False,category='prop'):
        parts.append(dict(name=n,shape='cylinder',pos=tuple(p),size=(rad,h),color=c,
                          rot=tuple(r),collision=collision,category=category))
    def rod(n,a,b,rad,c='room_metal',collision=False,category='prop'):
        d=[b[i]-a[i] for i in range(3)];h=math.sqrt(sum(x*x for x in d))
        cyl(n,tuple((a[i]+b[i])/2 for i in range(3)),rad,h,c,
            (0,math.acos(d[2]/h),math.atan2(d[1],d[0])),collision,category)
    box('concrete_slab',(0,0,-.09),(24,12,.18),'room_concrete',collision=True,category='floor')
    # Fine concrete variations: flat organic triangles avoid checkerboard tiling.
    rng=random.Random(24012)
    folder=HERE/'assets'/'room';folder.mkdir(parents=True,exist_ok=True)
    for shade in ('light','dark'):
        verts=[];faces=[]
        for i in range(85):
            x=rng.uniform(-11.9,11.9);y=rng.uniform(-5.9,5.9)
            radius=rng.uniform(.025,.22);start=len(verts)
            verts.append((x,y,.00012))
            for j in range(7):
                a=j*math.tau/7;rr=radius*rng.uniform(.5,1.4)
                verts.append((max(-11.99,min(11.99,x+rr*math.cos(a))),max(-5.99,min(5.99,y+rr*math.sin(a))),.00012))
            for j in range(7):faces.append((start,start+j+1,start+(j+1)%7+1))
        mesh='assets/room/concrete_'+shade+'.obj'
        _atomic_text(HERE/mesh, ''.join('v %.5f %.5f %.5f\n'%v for v in verts)+''.join('f %d %d %d\n'%tuple(x+1 for x in f) for f in faces))
        parts.append(dict(name='concrete_wear_'+shade,shape='mesh',mesh=mesh,pos=(0,0,0),size=(1,1,1),rot=(0,0,0),color='room_floor_'+shade,collision=False,category='floor'))
    # North teaching wall, south workshop wall; east storage wall.
    for n,p,s in [('north',(0,6.09,1.9),(24,.18,3.8)),('south',(0,-6.09,1.9),(24,.18,3.8)),('east',(12.09,0,1.9),(.18,12,3.8))]:
        box('wall_'+n,p,s,collision=True,category='wall')
    # West glazing wall with two real doorway recesses at either end.
    box('window_wall_spandrel',(-12.09,0,.19),(.18,12,.38),collision=True,category='wall')
    box('window_wall_header',(-12.09,0,3.45),(.18,12,.7),collision=True,category='wall')
    for j in range(9):
        y=-4.4+j*1.1
        box('frosted_glazing_'+str(j),(-12.005,y,1.75),(.03,1.075,2.72),'room_glass_light' if j%2 else 'room_glass',collision=True,category='wall')
        box('window_mullion_'+str(j),(-11.975,y-.55,1.75),(.065,.025,2.75),'room_white',category='wall')
        rod('blind_pull_'+str(j),(-11.94,y+.48,2.91),(-11.94,y+.48,1.64),.003,'room_metal',category='wall')
        box('blind_weight_'+str(j),(-11.94,y+.48,1.63),(.018,.018,.085),'room_metal',category='wall')
    for j,y in enumerate((-5.48,5.48)):
        box('exit_door_'+str(j),(-11.96,y,1.12),(.085,.94,2.24),'room_metal',collision=True,category='wall')
        for sy in (-1,1):box(f'door_jamb_{j}_{sy}',(-11.91,y+sy*.49,1.16),(.10,.045,2.32),'room_trim',category='wall')
        box('door_lintel_'+str(j),(-11.91,y,2.3),(.10,1.02,.06),'room_trim',category='wall')
        rod('door_pushbar_'+str(j),(-11.84,y-.34,1.05),(-11.84,y+.34,1.05),.015,'room_silver',category='wall')
        box('exit_sign_housing_'+str(j),(-11.86,y,2.61),(.10,.46,.20),'room_white',category='wall')
        box('exit_sign_green_'+str(j),(-11.801,y,2.61),(.004,.41,.15),'room_exit_green',category='wall')
        # Bright white running-person geometry and arrow within green sign.
        box('exit_body_'+str(j),(-11.796,y,2.61),(.004,.035,.065),'room_white',category='wall')
        cyl('exit_head_'+str(j),(-11.79,y,2.663),.016,.004,'room_white',(0,math.pi/2,0),category='wall')
    for side in (-1,1):
        box('baseboard_'+str(side),(0,side*5.975,.065),(24,.045,.13),'room_trim',category='wall')
    # Structural piers dividing tall frosted window bays.
    for y in (-3.45,2.15):box('structural_pier_'+str(y),(-11.65,y,1.9),(.65,.62,3.8),'room_metal',collision=True,category='wall')
    # Teaching wall long whiteboards and low cupboards (north), marker trays.
    for i,x in enumerate((1.0,4.7)):
        box('whiteboard_frame_'+str(i),(x,5.91,1.95),(3.5,.07,1.24),'room_silver',category='wall')
        box('whiteboard_face_'+str(i),(x,5.865,1.95),(3.43,.018,1.17),'room_white',category='wall')
        box('whiteboard_tray_'+str(i),(x,5.80,1.31),(3.5,.15,.035),'room_metal',category='wall')
        for k in range(8):
            # Sparse uneven short strokes evoke formulas without fake readable claims.
            xx=x-1.35+k*.34; zz=1.7+.16*(k%3)
            box(f'whiteboard_note_{i}_{k}',(xx,5.852,zz),(.13+.03*(k%2),.002,.004),'room_ink',category='wall')
        box('low_cupboard_'+str(i),(x,5.64,.48),(3.45,.65,.88),'room_white',collision=True)
        box('cupboard_top_'+str(i),(x,5.64,.94),(3.5,.7,.04),'room_metal',collision=True)
        for j in range(4):
            xx=x-1.28+j*.85
            box(f'cabinet_seam_{i}_{j}',(xx+.42,5.306,.48),(.008,.005,.82),'room_metal')
            box(f'cabinet_pull_{i}_{j}',(xx+.30,5.295,.69),(.012,.025,.11),'room_silver')
    # Industrial open ceiling: steel deck, beams, duct mains, pendants and vents.
    box('ceiling_deck',(0,0,3.84),(24,12,.08),'room_ceiling',category='ceiling')
    for x in range(-10,12,4):
        box('ceiling_beam_'+str(x),(x,0,3.56),(.28,12,.40),'room_white',category='ceiling')
    for y in (-4.3,1.2):
        rod('main_duct_'+str(y),(-11.8,y,3.27),(11.8,y,3.27),.19,'room_silver',category='ceiling')
        for x in range(-10,12,4):
            cyl(f'duct_collar_{y}_{x}',(x,y,3.27),.198,.045,'room_metal',(0,math.pi/2,0),category='ceiling')
    for x in (-9,-3,3,9):
        for y in (-2.5,3.8):
            rod(f'light_wire_a_{x}_{y}',(x-.7,y,3.8),(x-.7,y,3.12),.004,'room_dark',category='ceiling')
            rod(f'light_wire_b_{x}_{y}',(x+.7,y,3.8),(x+.7,y,3.12),.004,'room_dark',category='ceiling')
            box(f'led_housing_{x}_{y}',(x,y,3.11),(1.8,.10,.055),'room_silver',category='ceiling')
            box(f'led_diffuser_{x}_{y}',(x,y,3.078),(1.75,.085,.009),'room_light',category='ceiling')
            cyl(f'vent_drop_{x}_{y}',(x,y+.60,3.40),.12,.37,'room_silver',category='ceiling')
            cyl(f'vent_disc_{x}_{y}',(x,y+.60,3.19),.28,.055,'room_white',category='ceiling')
            cyl(f'vent_inset_{x}_{y}',(x,y+.60,3.158),.22,.008,'room_metal',category='ceiling')
    # Exposed wall conduit and sockets.
    rod('north_conduit',(-11,5.83,.95),(11.8,5.83,.95),.014,'room_metal')
    for x in (-9,-5,0,5,10):box('socket_'+str(x),(x,5.81,1.01),(.075,.035,.12),'room_white')
    # Freestanding wheeled whiteboard in the middle distance.
    bx,by=-6.5,3.5
    box('mobile_whiteboard',(bx,by,1.28),(1.15,.06,1.68),'room_white',collision=True)
    for x in (bx-.62,bx+.62):
        rod('board_upright_'+str(x),(x,by,.15),(x,by,2.14),.018)
        rod('board_foot_'+str(x),(x,by-.28,.10),(x,by+.28,.10),.018)
        for sy in (-1,1):cyl(f'board_caster_{x}_{sy}',(x,by+sy*.26,.055),.045,.025,'room_dark',(math.pi/2,0,0),True)
    for j in range(14):box('mobile_board_note_'+str(j),(bx-.44+(j%3)*.28,by-.032,1.91-(j//3)*.19),(.15,.002,.006),'room_ink')
    # Lightweight wooden wing test stand with tripod from west-wall frame.
    cx,cy=-9,-1.8
    for x in (cx-.95,cx+.95):
        rod('wing_stand_post_'+str(x),(x,cy,.08),(x,cy,1.2),.018,'room_wood',True)
        box('wing_stand_foot_'+str(x),(x,cy,.035),(.05,.68,.07),'room_wood',collision=True)
    for z in (.87,1.12):box('wing_spar_'+str(z),(cx,cy,z),(2.4,.025,.025),'room_wood')
    for j in range(13):
        x=cx-1.15+j*.19
        box('wing_rib_'+str(j),(x,cy,1.00),(.014,.34,.024),'room_wood')
    rod('tripod_mast',(cx,cy+.7,.52),(cx,cy+.7,1.43),.025,'room_metal',True)
    for a in (0,math.tau/3,2*math.tau/3):rod('tripod_leg_'+str(a),(cx,cy+.7,.76),(cx+.43*math.cos(a),cy+.7+.43*math.sin(a),.04),.012,'room_metal',True)
    cyl('tripod_instrument',(cx,cy+.7,1.46),.06,.08,'room_dark')
    # Floor tape seen around demo area. Visual-only, no imaginary road grid.
    for i,(x,y,a) in enumerate(((1.9,1.9,.1),(2.3,1.9,.1),(-1.1,-1.7,.4))):box('floor_tape_'+str(i),(x,y,.0005),(.16,.025,.001),'room_floor_tape',(0,0,a))
    from room_furniture import furniture_assembly, FURNITURE_COLORS
    ROOM_COLORS.update(FURNITURE_COLORS)
    parts.extend(furniture_assembly())
    # Refine common azimuths from the continuous panorama: board -> computers
    # -> exits -> opposite glazing -> fabrication/flag -> teaching table.
    # Relocate whole named assemblies rigidly so their small details stay aligned.
    def relocate(part, old, new, yaw=0):
        x,y,z=part['pos'];dx=x-old[0];dy=y-old[1]
        c,ss=math.cos(yaw),math.sin(yaw)
        part['pos']=(new[0]+c*dx-ss*dy,new[1]+ss*dx+c*dy,z)
        rx,ry,rz=part['rot'];part['rot']=(rx,ry,rz+yaw)
    window_prefixes=('window_wall_','frosted_glazing_','window_mullion_',
        'blind_pull_','blind_weight_','exit_door_','door_jamb_','door_lintel_',
        'door_pushbar_','exit_sign_','exit_body_','exit_head_','structural_pier_')
    furniture_moves=[
        ('computer0_', (6.5,5.15),(11.1,3.0),-math.pi/2),
        ('computer1_', (8.2,5.15),(11.1,1.3),-math.pi/2),
        ('computer2_', (9.9,5.15),(11.1,-.4),-math.pi/2),
        ('locker0_',(-10.5,5.48),(8.4,5.48),0),
        ('locker1_',(-9.55,5.48),(9.35,5.48),0),
        ('locker2_',(-8.6,5.48),(10.3,5.48),0),
        ('shop_island_', (7,-3),(-7,-3),0),
        ('flag_partition_', (6.8,-4.7),(-3.8,2.4),-math.pi/2),
        ('tools0_',(-6,-3),(-9.4,-3),0),
        ('tools1_',(-7.15,-3),(-10.3,-3),0),
        ('tools2_', (5,-4.8),(-5,-4.8),0),
        ('supply_shelf_', (10.9,-4.9),(-10.7,-4.85),0),
        ('rear_shelf_', (-9.9,-4.9),(-11.3,3.5),math.pi/2),
        ('foam_aircraft_',(-3.8,-5.65),(-11.65,-2.0),math.pi/2),
        ('white_foam_stock_',(-5.4,-5.42),(-8.9,-3.6),0),
        ('teaching_table4_',(-8.3,.5),(5,-3.9),0),
        ('teaching_table5_',(-8.3,1.4),(5,-4.8),0),
        ('chair_a4_',(-8.7,-.25),(4.6,-3.15),0),
        ('chair_a5_',(-8.7,.65),(4.6,-4.05),0),
    ]
    for part in parts:
        n=part['name']
        if n=='wall_south':
            part['name']='wall_west';part['pos']=(-12.09,0,1.9)
            part['size']=(.18,12,3.8)
        elif n.startswith(window_prefixes):
            x,y,z=part['pos'];part['pos']=(-2*y,x/2,z)
            rx,ry,rz=part['rot'];part['rot']=(rx,ry,rz+math.pi/2)
            # Stretch glazing bays to the long wall; doors and hardware retain
            # real furniture scale while their positions span the wider facade.
            if n.startswith(('window_wall_','frosted_glazing_','window_mullion_')):
                size=list(part['size'])
                if part['shape']=='box':size[1]*=2
                part['size']=tuple(size)
        elif n.startswith(('wing_stand_','wing_spar_','wing_rib_','tripod_')):
            relocate(part,(-9,-1.8),(0,-4.7))
        elif n=='baseboard_-1':
            part['pos']=(-11.975,0,.065);part['size']=(.045,12,.13)
        for prefix,old,new,yaw in furniture_moves:
            if n.startswith(prefix):
                relocate(part,old,new,yaw)
                break
    return parts


def room_assembly():
    """Cardinal layout anchored by user: 8s south, 11s west, 17s north."""
    from room_details import room_details, DETAIL_COLORS
    details = room_details()
    ROOM_COLORS.update(DETAIL_COLORS)
    parts = []
    replaced = ('computer', 'locker', 'flag_partition_', 'mobile_whiteboard',
                'board_upright_', 'board_foot_', 'board_caster_', 'mobile_board_note_',
                'shop_island_', 'foam_aircraft_', 'white_foam_stock_',
                'wing_stand_', 'wing_spar_', 'wing_rib_', 'tools0_')
    teaching = ('teaching_table0_', 'teaching_table1_', 'teaching_table2_', 'teaching_table3_',
                'chair_a0_', 'chair_a1_', 'chair_b2_', 'chair_b3_', 'teaching_end_',
                'ref_laptop', 'ref_power_', 'ref_backpack_')
    for p in _authored_room_assembly():
        n = p['name']
        if n.startswith(replaced) or n == 'wall_east':
            continue
        x,y,z = p['pos']
        if n.startswith(('whiteboard_', 'low_cupboard_', 'cupboard_top_', 'cabinet_seam_', 'cabinet_pull_')):
            # Standard eight-foot boards and matching cabinet runs fit the
            # two visible panels much better than the first 3.5m estimate.
            center = 1.0 if x < 2.85 else 4.7
            x = (1.0 if center == 1.0 else 3.5) + (x-center)*.70
            if p['shape']=='box' and p['size'][0] > 1:
                p['size']=(p['size'][0]*.70,*p['size'][1:])
        if n.startswith(teaching):
            x -= 4
        if n.startswith(('teaching_table4_', 'teaching_table5_', 'chair_a4_', 'chair_a5_')):
            # Meeting tables are interior furnishings seen to the right at 8s.
            y += 3.4
            x += 2.8
        p['pos'] = (y,-x,z)
        rx,ry,rz = p['rot']; p['rot'] = (rx,ry,rz-math.pi/2)
        if n == 'wall_north': p['name'] = 'cardinal_east_wall'
        elif n == 'wall_west': p['name'] = 'cardinal_north_wall'
        parts.append(p)
    def box(n, pos, size, color, collision=False):
        parts.append(dict(name='south_facade_'+n,shape='box',pos=pos,size=size,
                          rot=(0,0,0),color=color,collision=collision,category='wall'))
    box('header',(0,-12.09,3.45),(12,.18,.70),'room_wall',True)
    box('spandrel',(0,-12.09,.19),(12,.18,.38),'room_wall',True)
    for j in range(9):
        x=-4.4+j*1.1
        box('glass_'+str(j),(x,-12,1.75),(1.075,.03,2.72),'room_glass',True)
        box('mullion_'+str(j),(x-.55,-11.975,1.75),(.025,.065,2.75),'room_white')
    for j,x in enumerate((-5.45,5.45)):
        box('door_'+str(j),(x,-11.96,1.12),(.94,.085,2.24),'room_metal',True)
        box('pushbar_'+str(j),(x,-11.90,1.05),(.72,.035,.035),'room_silver')
        box('exit_'+str(j),(x,-11.89,2.61),(.41,.02,.15),'room_exit_green')
    # A loose red chair is visible near the operator in both the 4s teaching
    # view and the return toward the divider; it is not part of the table row.
    for chair_part in list(parts):
        if chair_part['name'].startswith('chair_a0_'):
            copy = dict(chair_part)
            copy['name']='loose_reference_'+chair_part['name']
            x,y,z=copy['pos'];copy['pos']=(x+.75,y-4.2,z)
            parts.append(copy)
    for part in parts:
        n=part['name'];x,y,z=part['pos']
        if n.startswith(('teaching_table4_', 'teaching_table5_', 'chair_a4_', 'chair_a5_')):
            part['pos']=(-2.6+(x+.95),-5.7+(y+7.8)*1.5,z)
            if n.startswith('teaching_table') and part['shape']=='box':
                a,b,c=part['size'];part['size']=(a*1.5,b,c)
        elif n.startswith('tools0_'):
            part['pos']=(-2.1+(x+3),4.7+(y-9.4),z)
    # Chairs flank the joined meeting tops, with three on each outer edge.
    templates=[p for p in parts if p['name'].startswith(('chair_a4_', 'chair_a5_'))]
    parts=[p for p in parts if not p['name'].startswith(('chair_a4_', 'chair_a5_'))]
    for part in templates:
        west=part['name'].startswith('chair_a5_')
        ox=-2.3 if west else -1.4
        x,y,z=part['pos'];dx=x-ox;dy=y+5.1
        for j,yy in enumerate((-6.5,-5.7,-4.9)):
            q=dict(part);q['name']='meeting_row'+str(j)+'_'+part['name']
            q['pos']=((-3.8-dx if west else -1.4+dx),yy-dy if west else yy+dy,z)
            if west:
                rx,ry,rz=q['rot'];q['rot']=(rx,ry,rz+math.pi)
            parts.append(q)
    for part in parts:
        if part['name'].startswith(('teaching_table4_', 'teaching_table5_', 'meeting_row')):
            x,y,z=part['pos'];part['pos']=(x-1.0,y+1.8,z)
    # Loose and stacked chairs occupy the bay behind the meeting table.
    template=[p for p in parts if p['name'].startswith('meeting_row1_chair_a4_')]
    for j,(cx,cy,cz) in enumerate(((-5.3,-3.25,0),(-5.3,-3.25,.10),(-5.3,-2.7,0))):
        for p in template:
            q=dict(p);q['name']='window_chair_'+str(j)+'_'+p['name']
            x,y,z=p['pos'];q['pos']=(cx+(x+2.4),cy+(y+3.9),z+cz)
            parts.append(q)
    from room_walk_details import walk_parts, WALK_COLORS
    ROOM_COLORS.update(WALK_COLORS)
    parts.extend(walk_parts())
    parts.extend(details)
    return parts


def add_room(world,node):
    """Add finite static room with furniture collisions, preserving open floor."""
    model=node(world,'model',name='photo_robotics_lab');node(model,'static','true')
    link=node(model,'link',name='lab_structure')
    for p in room_assembly():
        for kind in (('visual','collision') if p['collision'] else ('visual',)):
            obj=node(link,kind,name=p['name']+'_'+kind)
            node(obj,'pose',' '.join(map(str,(*p['pos'],*p['rot']))))
            shape=node(node(obj,'geometry'),p['shape'])
            if p['shape']=='box':node(shape,'size',' '.join(map(str,p['size'])))
            elif p['shape']=='mesh':node(shape,'uri',p['mesh']);node(shape,'scale','1 1 1')
            else:node(shape,'radius',p['size'][0]);node(shape,'length',p['size'][1])
            if kind=='visual':
                mat=node(obj,'material');rgba=' '.join(map(str,(*ROOM_COLORS[p['color']],1)))
                node(mat,'ambient',rgba);node(mat,'diffuse',rgba)
                node(mat,'specular','.12 .12 .12 1' if p['color'] in ('room_silver','room_metal') else '.015 .015 .015 1')
                if p['color']=='room_light':node(mat,'emissive','.7 .7 .6 1')
    for i,(x,y) in enumerate(((-3,-8),(3,-8),(-3,0),(3,0),(-3,8),(3,8))):
        light=node(world,'light',name='lab_ceiling_light_'+str(i),type='point')
        node(light,'pose',f'{x} {y} 3.02 0 0 0')
        node(light,'diffuse','.50 .49 .44 1');node(light,'specular','.07 .07 .07 1')
        node(light,'cast_shadows','false')
        attenuation=node(light,'attenuation');node(attenuation,'range',10)
        node(attenuation,'constant',1);node(attenuation,'linear',.08);node(attenuation,'quadratic',.015)
    return model
