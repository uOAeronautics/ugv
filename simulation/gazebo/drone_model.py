"""Detailed photo-informed stationary quadcopter parking target.

X is along the two landing skids, Y left, Z up; origin at floor level.
The flat part schema matches assembly.py, adding shape=mesh and mesh=<path>.
Collision follows separate tubes/plates: the central drive-under space stays open.
"""
import json
import math
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRONE_COLORS = {
    'carbon': (.027,.032,.035), 'carbon_edge': (.012,.016,.019),
    'foam': (.018,.019,.022), 'red': (.69,.025,.055),
    'silver': (.58,.63,.66), 'steel': (.28,.32,.34),
    'white': (.93,.94,.90), 'black': (.006,.008,.009),
    'copper': (.55,.22,.075), 'green': (.045,.24,.10),
    'yellow': (.98,.62,.015), 'blue': (.015,.25,.65),
    'wire_white': (.75,.76,.67), 'gold': (.67,.40,.08),
}

def _prop_mesh():
    """Cambered tapered two-blade propeller, closed triangle surface, SI units."""
    folder=HERE/'assets'/'drone'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'propeller.obj'
    verts=[];faces=[]
    # Root, broad shoulder, curved swept tip. Each section has upper/lower lip.
    stations=[(.009,.008,0),(.027,.014,.003),(.061,.024,.005),
              (.105,.022,.008),(.148,.014,.012),(.181,.007,.016),(.190,.001,.017)]
    for direction in (1,-1):
        start=len(verts)
        for x,chord,sweep in stations:
            pitch=.21*(1-x/.24)
            for side,zlip in ((-1,-.001), (1,-.001), (1,.001),(-1,.001)):
                y=sweep+side*chord
                z=side*chord*pitch+zlip
                verts.append((direction*x,direction*y,z))
        for i in range(len(stations)-1):
            for k in range(4):
                a=start+i*4+k;b=start+i*4+(k+1)%4
                c=start+(i+1)*4+(k+1)%4;d=start+(i+1)*4+k
                faces.extend(((a,b,c),(a,c,d)))
        faces.extend(((start,start+2,start+1),(start,start+3,start+2)))
        j=start+(len(stations)-1)*4
        faces.extend(((j,j+1,j+2),(j,j+2,j+3)))
    text='# Photo estimated 380mm two blade propeller\n'
    text+=''.join('v %.7f %.7f %.7f\n'%v for v in verts)
    text+=''.join('f %d %d %d\n'%tuple(i+1 for i in f) for f in faces)
    # Gazebo generation and Blender previews may request this asset concurrently.
    # Preserve unchanged assets, and publish changed contents as one atomic rename.
    try:
        unchanged = path.read_text(encoding='utf-8') == text
    except FileNotFoundError:
        unchanged = False
    if not unchanged:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=folder, prefix='.propeller-', suffix='.obj', delete=False) as output:
                temporary = Path(output.name)
                output.write(text)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return 'assets/drone/propeller.obj'

def drone_assembly():
    """Return reusable visual/collision part dictionaries (all dimensions SI)."""
    p=json.loads((HERE/'drone_dimensions.json').read_text())
    parts=[]
    def add(name,shape,pos,size,color,rot=(0,0,0),collision=False,**extra):
        parts.append(dict(name=name,shape=shape,pos=tuple(pos),size=tuple(size),
                          color=color,rot=tuple(rot),collision=collision,**extra))
    def box(n,xyz,size,c,rot=(0,0,0),collision=False):add(n,'box',xyz,size,c,rot,collision)
    def cyl(n,xyz,r,h,c,rot=(0,0,0),collision=False):add(n,'cylinder',xyz,(r,h),c,rot,collision)
    def rod(n,a,b,r,c,collision=False):
        dx,dy,dz=[b[i]-a[i] for i in range(3)]
        length=math.sqrt(dx*dx+dy*dy+dz*dz)
        cyl(n,[(a[i]+b[i])/2 for i in range(3)],r,length,c,
            (0,math.acos(max(-1,min(1,dz/length))),math.atan2(dy,dx)),collision)
    def wire(n,points,c,r=.0015):
        for i,(a,b) in enumerate(zip(points,points[1:])):rod(n+'_'+str(i),a,b,r,c)

    # Continuous black skid cores with individual foam sleeves and red annular seams.
    for s in (-1,1):
        y=s*p['skid_center_spacing']/2
        rod(f'skid_core_{s}',(-.235,y,.016),(.235,y,.016),.015,'carbon',True)
        for j in range(23):
            x=-.22+j*.02
            cyl(f'foam_{s}_{j}',(x,y,.017),.015,.0175,'foam',(0,math.pi/2,0))
            cyl(f'red_ring_{s}_{j}',(x+.009,y,.017),.0151,.0018,'red',(0,math.pi/2,0))
        for end in (-1,1):
            cyl(f'endcap_{s}_{end}',(end*.237,y,.017),.0105,.006,'steel',(0,math.pi/2,0))
        # One splayed tube joins the CENTER of each skid to the central deck.
        # Photos 1/3/5 show T-shaped landing gear, rather than a four-leg cage.
        a=(0,s*.095,.407); b=(0,y,.029)
        rod(f'landing_leg_{s}',a,b,.011,'carbon',True)
        rod(f'leg_upper_collar_{s}',a,(0,s*.103,.373),.014,'green')
        rod(f'leg_lower_collar_{s}',b,(0,s*.164,.066),.013,'carbon_edge')
        box(f'leg_clamp_{s}',(0,y,.032),(.032,.035,.015),'carbon_edge')
        cyl(f'leg_clamp_bolt_{s}',(0,y,.042),.003,.004,'silver')

    # Stacked circular carbon sandwich, exposed silver spacers and perimeter bolts.
    for z in (.410,.438):
        cyl('center_plate_'+str(z),(0,0,z),p['deck_diameter']/2,.004,'carbon',collision=True)
        for j in range(12):
            a=j*math.tau/12; x=.105*math.cos(a);y=.105*math.sin(a)
            cyl(f'plate_bolt_{z}_{j}',(x,y,z+.003),.0032,.002,'silver')
            if z==.410:cyl(f'plate_spacer_{j}',(x,y,.424),.003,.024,'steel')
    propmesh=_prop_mesh()
    # Four radial carbon booms, bolted root saddles, vented motor cages.
    for i,a in enumerate((math.pi/4,3*math.pi/4,5*math.pi/4,7*math.pi/4)):
        c,s=math.cos(a),math.sin(a); x,y=.36*c,.36*s
        rod(f'boom_{i}',(.08*c,.08*s,.43),(x,y,.43),.013,'carbon',True)
        for rr in (.105,.145):
            box(f'boom_saddle_{i}_{rr}',(rr*c,rr*s,.43),(.032,.039,.044),'carbon_edge',(0,0,a))
            for q in (-1,1):
                xx=rr*c-q*.015*s;yy=rr*s+q*.015*c
                cyl(f'boom_bolt_{i}_{rr}_{q}',(xx,yy,.454),.0028,.003,'silver')
        box(f'motor_base_{i}',(x,y,.431),(.065,.057,.005),'carbon',(0,0,a),True)
        for k in (-1,1):
            box(f'motor_cage_{i}_{k}',(x-k*.028*s,y+k*.028*c,.451),(.063,.004,.030),'carbon_edge',(0,0,a))
            for j in (-1,0,1):
                # Silver heads give the elongated motor brackets a perforated appearance.
                cyl(f'cage_fastener_{i}_{k}_{j}',(x+j*.019*c-k*.031*s,y+j*.019*s+k*.031*c,.448),.003,.002,'steel',(math.pi/2,0,a))
        cyl(f'motor_bottom_{i}',(x,y,.461),.025,.008,'steel')
        cyl(f'motor_can_{i}',(x,y,.479),.024,.026,'black',collision=True)
        cyl(f'motor_top_{i}',(x,y,.494),.023,.004,'silver')
        for j in range(12):
            t=j*math.tau/12
            box(f'motor_winding_{i}_{j}',(x+.015*math.cos(t),y+.015*math.sin(t),.496),(.004,.006,.005),'copper',(0,0,t))
        cyl(f'prop_hub_{i}',(x,y,.505),.013,.012,'silver')
        add(f'propeller_{i}','mesh',(x,y,.508),(1,1,1),'white',(0,0,a+.25+i*.49),mesh=propmesh)
        # Rounded conical spinner built from rings avoids external mesh dependency.
        for j in range(9):
            cyl(f'spinner_{i}_{j}',(x,y,.516+j*.0022),.0085*(1-j/10)**.65,.0024,'silver')
        wire(f'esc_wire_{i}',[(.08*c,.08*s,.447),(.16*c,.16*s,.452),(.29*c,.29*s,.447),(x,y,.459)],'black',.002)
        box(f'esc_{i}',(.19*c,.19*s,.445),(.055,.019,.009),'black',(0,0,a))

    # Flight controller electronics, little connectors, ribbon sockets, tangled harness.
    box('autopilot',(0,.014,.457),(.075,.044,.019),'black',collision=True)
    box('controller_label',(-.008,.014,.467),(.035,.027,.0008),'white')
    box('receiver',(.061,-.017,.455),(.041,.028,.015),'green')
    box('receiver_shield',(.061,-.017,.464),(.025,.020,.003),'silver')
    for j in range(9):
        box(f'signal_socket_{j}',(-.038+j*.009,.042,.465),(.006,.008,.008),'black')
        cyl(f'socket_pin_{j}',(-.038+j*.009,.042,.470),.0006,.003,'gold')
    for i,c in enumerate(('red','blue','yellow','green','wire_white','black')*3):
        xx=-.075+(i%7)*.021; yy=-.065+(i%4)*.033
        wire(f'harness_{i}',[(xx,yy,.446),(xx+.009,yy-.014,.466+i%3*.006),
             (-xx*.6,.024+(i%4)*.005,.479+i%5*.003),(.025,-.025,.461)],c,.00075)
    rod('radio_antenna',(-.055,-.025,.47),(-.17,-.08,.471),.003,'black')
    # Tall GPS puck on a slender carbon mast, with visible loose red/black wiring.
    rod('gps_mast',(-.033,.025,.444),(-.033,.025,.637),.0035,'carbon',True)
    cyl('gps_stalk_foot',(-.033,.025,.449),.009,.010,'black')
    cyl('gps_puck',(-.033,.025,.665),.027,.055,'black',collision=True)
    cyl('gps_puck_rim',(-.033,.025,.640),.028,.005,'carbon_edge')
    cyl('gps_cap',(-.033,.025,.693),.026,.002,'carbon')
    for j,c in enumerate(('red','black','wire_white')):
        wire(f'gps_cable_{j}',[(-.033+j*.003,.025,.638),(-.041+j*.003,.029,.592),
             (-.032+j*.003,.024,.549),(-.025+j*.003,.021,.47)],c,.0008)

    # Open hanging payload cage. Its bottom is above the UGV, no floor-to-deck box.
    for x in (-.046,.046):
        for y in (-.035,.035):rod(f'payload_post_{x}_{y}',(x,y,.406),(x,y,.267),.0025,'carbon',True)
    box('payload_shelf',(0,0,.266),(.108,.085,.004),'carbon',collision=True)
    box('payload_silver',(0,0,.28),(.065,.055,.025),'silver',collision=True)
    box('gimbal_servo',(.008,-.052,.248),(.042,.022,.033),'black',collision=True)
    box('servo_label',(.008,-.064,.248),(.021,.001,.021),'white')
    cyl('servo_horn',(.008,-.066,.248),.011,.004,'gold',(math.pi/2,0,0))
    box('payload_camera',(.037,0,.239),(.022,.042,.028),'black',collision=True)
    cyl('payload_lens',(.051,0,.239),.009,.008,'black',(0,math.pi/2,0))
    # Dangling power connectors kept near one side of overhead payload.
    for i,c in enumerate(('red','black')):
        wire(f'power_dangle_{i}',[(.057,-.07+i*.004,.422),(.075,-.10+i*.004,.35),(.09,-.11+i*.004,.315)],c,.0022)
    box('yellow_xt60',(.092,-.11,.308),(.021,.015,.022),'yellow',(0,.3,.15))
    for j in (-1,1):cyl('xt60_socket_'+str(j),(.092+j*.004,-.11,.295),.0025,.003,'gold')
    wire('servo_cable',[(-.012,-.04,.245),(-.03,-.05,.21),(-.019,-.06,.196)],'black',.0011)

    # Real DICT_4X4_50 markers, 50mm black square plus 9mm white quiet zone.
    grids={0: ("1011","0101","0011","0010"),
           1: ("0000","1111","1001","1010"),
           9: ("1100","1111","0101","0110"),
           12:("0000","1110","1011","0111")}
    # Side cards face outward across Y; payload card faces -X toward approach.
    def card(name,pos,rot,size,marker_id):
        # Local XY plane, normal +Z; rotate whole card into vertical plane.
        rx,ry,rz=rot
        def transform(v):
            x,y,z=v; y,z=y*math.cos(rx)-z*math.sin(rx),y*math.sin(rx)+z*math.cos(rx)
            x,z=x*math.cos(ry)+z*math.sin(ry),-x*math.sin(ry)+z*math.cos(ry)
            x,y=x*math.cos(rz)-y*math.sin(rz),x*math.sin(rz)+y*math.cos(rz)
            return tuple(pos[i]+(x,y,z)[i] for i in range(3))
        box(name+'_paper',pos,(size,size,.0008),'white',rot)
        cell=.05/6
        # One printed face per physical card; four distinct cards in total.
        for face in (1,):
            box(name+'_ink_'+str(face),transform((0,0,face*.0006)),(.05,.05,.0003),'black',rot)
            for row,line in enumerate(grids[marker_id],1):
                for col,bit in enumerate(line,1):
                    if bit=='1':
                        xx=(col-2.5)*cell*face
                        box(f'{name}_pixel_{face}_{row}_{col}',transform((xx,(2.5-row)*cell,face*.0009)),
                            (cell,cell,.0003),'white',rot)
    card('payload_marker_front',(-.058,-.005,.202),(math.pi/2,0,-math.pi/2),.068,0)
    card('payload_marker_rear',(.058,-.005,.202),(math.pi/2,0,math.pi/2),.068,1)
    for s in (-1,1):
        card('leg_marker_'+str(s),(0,s*.168,.092),(-s*math.pi/2,0,0),.068,
             9 if s==-1 else 12)
    return parts

def add_drone(world,node,pose=(.8,0,0,0,0,0)):
    """Insert static photo model with only local solid collision components."""
    model=node(world,'model',name='photo_quadcopter')
    node(model,'static','true');node(model,'pose',' '.join(map(str,pose)))
    link=node(model,'link',name='drone_frame')
    for part in drone_assembly():
        for kind in (('visual','collision') if part['collision'] else ('visual',)):
            obj=node(link,kind,name=part['name']+'_'+kind)
            node(obj,'pose',' '.join(map(str,(*part['pos'],*part['rot']))))
            shape=node(node(obj,'geometry'),part['shape'])
            if part['shape']=='box':node(shape,'size',' '.join(map(str,part['size'])))
            elif part['shape']=='mesh':
                node(shape,'uri',part['mesh']);node(shape,'scale','1 1 1')
            else:
                node(shape,'radius',part['size'][0]);node(shape,'length',part['size'][1])
            if kind=='visual':
                mat=node(obj,'material');rgba=' '.join(map(str,(*DRONE_COLORS[part['color']],1)))
                node(mat,'ambient',rgba);node(mat,'diffuse',rgba)
                node(mat,'specular','.22 .22 .22 1' if part['color']=='silver' else '.025 .025 .025 1')
    return model
