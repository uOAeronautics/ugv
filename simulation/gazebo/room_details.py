"""Video-anchor detail pass in cardinal coordinates: E+X,N+Y, 12x24m."""
import math
import os
import tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
DETAIL_COLORS={
 'detail_black':(.018,.022,.026),'detail_metal':(.28,.31,.32),
 'detail_silver':(.58,.62,.63),'detail_wood':(.58,.43,.25),
 'detail_pale':(.72,.65,.47),'detail_red':(.58,.012,.025),
 'detail_white':(.81,.82,.78),'detail_blue':(.02,.12,.31),
 'detail_orange':(.76,.19,.018),'detail_yellow':(.81,.63,.04),
 'detail_screen':(.015,.028,.035),'detail_green':(.025,.16,.07),
 'detail_cork':(.42,.31,.20),'detail_paper':(.77,.76,.69),
}

def _mesh(name,vertices,faces):
    path=HERE/'assets'/'room'/name;path.parent.mkdir(parents=True,exist_ok=True)
    data=''.join('v %.6f %.6f %.6f\n'%v for v in vertices)+''.join('f %s\n'%' '.join(str(i+1) for i in f) for f in faces)
    if not path.exists() or path.read_text()!=data:
        tmp=None
        try:
            with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+name+'.',suffix='.tmp',delete=False) as f:
                tmp=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if tmp:tmp.unlink(missing_ok=True)
    return 'assets/room/'+name


def room_details():
    p=[]
    def box(n,pos,size,c='detail_black',r=(0,0,0),collision=False):
        p.append(dict(name='detail_'+n,shape='box',pos=pos,size=size,color=c,rot=r,collision=collision))
    def cyl(n,pos,rad,h,c='detail_black',r=(0,0,0),collision=False):
        p.append(dict(name='detail_'+n,shape='cylinder',pos=pos,size=(rad,h),color=c,rot=r,collision=collision))
    def rod(n,a,b,rad=.012,c='detail_metal',collision=False):
        d=[b[i]-a[i] for i in range(3)];h=math.sqrt(sum(v*v for v in d))
        cyl(n,tuple((a[i]+b[i])/2 for i in range(3)),rad,h,c,(0,math.acos(d[2]/h),math.atan2(d[1],d[0])),collision)
    def mesh(n,path,pos,c,r=(0,0,0)):
        p.append(dict(name='detail_'+n,shape='mesh',mesh=path,pos=pos,size=(1,1,1),color=c,rot=r,collision=False))
    def monitor(n,x,y,z,normal='south'):
        # South-facing normal -Y; east-wall screens face west -X.
        box(n+'_stand',(x,y,z+.08),(.035,.04,.16),'detail_black')
        box(n+'_foot',(x,y-.03,z+.005),(.20,.16,.014),'detail_black')
        if normal=='south':
            box(n+'_housing',(x,y,z+.30),(.48,.045,.29),'detail_black')
            box(n+'_screen',(x,y-.025,z+.30),(.45,.003,.255),'detail_screen')
        else:
            box(n+'_housing',(x,y,z+.30),(.045,.48,.29),'detail_black')
            box(n+'_screen',(x-.025,y,z+.30),(.003,.45,.255),'detail_screen')
    def bench(n,x,y,w=2.0,d=.8):
        box(n+'_top',(x,y,.86),(w,d,.055),'detail_pale',collision=True)
        for sx in (-1,1):
            for sy in (-1,1):box(f'{n}_leg_{sx}_{sy}',(x+sx*(w/2-.08),y+sy*(d/2-.07),.43),(.045,.045,.84),'detail_metal',collision=True)
        box(n+'_rail',(x,y+d/2-.07,.25),(w-.1,.035,.045),'detail_metal',collision=True)
    # East computing station and five narrow black lockers, immediately beyond
    # the whiteboards in the clockwise sweep. Avoid generic three-desk row.
    box('east_desk_top',(5.05,-3,.76),(.72,1.65,.035),'detail_pale',collision=True)
    for yy in (-3.65,-2.35):
        box('east_desk_leg_'+str(yy),(5.05,yy,.39),(.06,.06,.72),'detail_silver',collision=True)
        box('east_desk_foot_'+str(yy),(5.05,yy,.045),(.62,.06,.04),'detail_silver',collision=True)
    monitor('east_monitor_a',5.14,-3.43,.79,'west');monitor('east_monitor_b',5.14,-2.90,.79,'west')
    box('east_keyboard',(4.91,-2.95,.792),(.16,.40,.018),'detail_black')
    box('controller_base',(4.94,-3.45,.81),(.23,.24,.10),'detail_black')
    # Steering-wheel controller: hollow polygon rim with three spokes.
    center=(4.86,-3.45,.97);rad=.115
    for j in range(20):
        aa=j*math.tau/20;bb=(j+1)*math.tau/20
        rod('steer_rim_'+str(j),(center[0],center[1]+rad*math.cos(aa),center[2]+rad*math.sin(aa)),(center[0],center[1]+rad*math.cos(bb),center[2]+rad*math.sin(bb)),.012,'detail_black')
    for j in range(3):
        a=j*math.tau/3;rod('steer_spoke_'+str(j),center,(center[0],center[1]+rad*math.cos(a),center[2]+rad*math.sin(a)),.013,'detail_metal')
    cyl('steer_center',center,.032,.035,'detail_black',(0,math.pi/2,0))
    for j in range(5):
        y=-4.1-j*.36
        box('locker_body_'+str(j),(5.50,y,1.0),(.62,.35,2),'detail_black',collision=True)
        box('locker_seam_'+str(j),(5.178,y-.172,1.0),(.006,.008,1.90),'detail_metal')
        box('locker_latch_'+str(j),(5.17,y+.095,1.03),(.018,.025,.10),'detail_silver')
        for k in range(4):box(f'locker_vent_{j}_{k}',(5.175,y,1.76-k*.035),(.006,.21,.008),'detail_metal')
    # West-side small aircraft and instruments visible between tripod and shop.
    bench('west_folding',-5.0,2.0,1.55,.60)
    box('west_crate',(-5.0,2.05,1.05),(.48,.36,.32),'detail_black',collision=True)
    box('west_crate_open_lid',(-5.0,2.23,1.27),(.48,.035,.34),'detail_black',(.35,0,0))
    for j in range(3):box('west_instrument_'+str(j),(-5.5+j*.28,1.84,.925),(.18,.12,.08),'detail_white')
    # Red folding X stand, small silver aircraft supported at waist height.
    for yy in (.65,.95):
        rod('red_x_a_'+str(yy),(-5.2,yy,.03),(-4.35,yy,.86),.012,'detail_red',True)
        rod('red_x_b_'+str(yy),(-4.35,yy,.03),(-5.2,yy,.86),.012,'detail_red',True)
    rod('redstand_top',(-5.2,.65,.86),(-5.2,.95,.86),.012,'detail_red')
    box('small_plane_fuselage',(-4.76,.80,.91),(.90,.045,.045),'detail_silver')
    box('small_plane_wing',(-4.8,.80,.93),(.22,1.18,.018),'detail_silver',(0,.03,.08))
    box('small_plane_tail',(-4.42,.80,.96),(.11,.35,.015),'detail_red')
    # Northwest black fabrication table with aircraft and a rolling stool.
    bench('northwest_shop',-4.5,6.4,2.3,1.0)
    box('northwest_shop_lower',(-4.5,6.4,.22),(2.1,.8,.04),'detail_black',collision=True)
    box('large_airframe_body',(-4.5,6.4,.97),(1.55,.08,.09),'detail_silver')
    box('large_airframe_wing',(-4.5,6.4,1.00),(.38,1.45,.027),'detail_silver')
    box('large_airframe_tail',(-3.85,6.4,1.01),(.16,.53,.02),'detail_red')
    # North back wall workstations, power tools and open wooden shelves.
    for i,x in enumerate((-3.8,-1.35)):
        bench('north_bench'+str(i),x,10.8,2.15,.85)
        for k in range(2):monitor(f'north_monitor_{i}_{k}',x-.42+k*.56,11.01,.90)
        box('north_keyboard_'+str(i),(x,10.56,.91),(.43,.15,.018),'detail_black')
        box('north_pcb_'+str(i),(x+.6,10.55,.92),(.25,.18,.05),'detail_green')
        cyl('north_stool_'+str(i),(x,9.9,.49),.18,.07,'detail_black',collision=True)
        rod('north_stool_post_'+str(i),(x,9.9,.08),(x,9.9,.46),.025,'detail_silver',True)
        for a in range(5):
            q=a*math.tau/5;rod(f'north_stool_foot_{i}_{a}',(x,9.9,.10),(x+.24*math.cos(q),9.9+.24*math.sin(q),.055),.012,'detail_silver')
        for k in range(5):
            box(f'north_clutter_{i}_{k}',(x-.70+k*.27,10.7,.95+.015*(k%2)),(.14,.16,.10),'detail_white' if k%2 else 'detail_orange')
    # Open wood shelves behind flagpartition: asymmetric boxes and loose stock.
    for sx in (-1,1):box('north_shelf_side_'+str(sx),(2.4+sx*.7,11.47,1.02),(.05,.48,2.04),'detail_wood',collision=True)
    for j in range(6):
        z=.10+j*.37;box('north_shelf_'+str(j),(2.4,11.47,z),(1.45,.50,.04),'detail_wood',collision=True)
        for k in range(3):box(f'north_shelf_stock_{j}_{k}',(1.95+k*.45,11.44,z+.10),(.34,.36,.15),'detail_white' if (j+k)%2 else 'detail_pale')
    cyl('orange_shop_vac',(-.1,11.0,.30),.22,.43,'detail_orange',collision=True)
    cyl('shop_vac_lid',(-.1,11,.54),.24,.07,'detail_black')
    rod('shop_vac_hose_a',(-.1,10.78,.45),(.32,10.45,.21),.025,'detail_black')
    rod('shop_vac_hose_b',(.32,10.45,.21),(.55,10.54,.06),.025,'detail_black')
    for j in range(4):cyl('back_container_'+str(j),(.35+j*.30,11.5,.24),.11,.45,'detail_white' if j%2 else 'detail_black',collision=True)
    # Distinct blue delta wing on the north wall, not a conventional crossplane.
    delta=_mesh('north_delta_wing.obj',[(-.65,0,-.5),(.65,0,-.5),(0,0,.62),(-.13,-.035,-.51),(.13,-.035,-.51),(0,-.035,.62)],[(0,1,2),(3,5,4),(0,2,5),(0,5,3),(1,4,5),(1,5,2),(0,3,4),(0,4,1)])
    mesh('north_delta',delta,(-3.45,11.89,2.5),'detail_blue')
    box('delta_centerline',(-3.45,11.85,2.45),(.025,.015,.76),'detail_silver')
    for j in range(3):box('north_spare_wing_'+str(j),(-4.95,11.88,2.15+j*.18),(.76,.05,.075),'detail_white')
    # Two wheeled room-divider panels between camera and back wall, flag on left.
    fx,fy=3.0,3.0
    for i in range(2):
        xx=fx-.82+i*1.66
        box('partition_cork_'+str(i),(xx,fy,1.3),(1.58,.085,.86),'detail_cork',collision=True)
        box('partition_bottom_'+str(i),(xx,fy,.55),(1.58,.10,.56),'detail_metal',collision=True)
        for ss in (-1,1):
            px=xx+ss*.82;box(f'partition_post_{i}_{ss}',(px,fy,1.0),(.055,.12,1.96),'detail_metal',collision=True)
            box(f'partition_foot_{i}_{ss}',(px,fy,.07),(.065,.62,.055),'detail_metal',collision=True)
            for ssy in (-1,1):cyl(f'partition_castor_{i}_{ss}_{ssy}',(px,fy+ssy*.25,.035),.035,.025,'detail_black',(0,math.pi/2,0))
        box('partition_top_'+str(i),(xx,fy,1.8),(1.68,.12,.07),'detail_metal')
        for j in range(5):box(f'partition_yellow_clip_{i}_{j}',(xx-.60+j*.30,fy,1.88),(.07,.08,.12),'detail_yellow')
    xx=fx-.82
    box('flag_white',(xx,fy-.049,1.31),(1.13,.008,.65),'detail_white')
    for ss in (-1,1):box('flag_red_'+str(ss),(xx+ss*.425,fy-.055,1.31),(.28,.008,.65),'detail_red')
    outline=[(0,.30),(.05,.17),(.13,.22),(.11,.10),(.25,.15),(.21,.04),(.29,.0),(.13,-.08),(.16,-.14),(.025,-.11),(.025,-.25),(-.025,-.25),(-.025,-.11),(-.16,-.14),(-.13,-.08),(-.29,0),(-.21,.04),(-.25,.15),(-.11,.10),(-.13,.22),(-.05,.17)]
    maple=_mesh('canadian_maple_detail.obj',[(0,0,0)]+[(x,0,z) for x,z in outline],[(0,j+1,(j+1)%len(outline)+1) for j in range(len(outline))])
    mesh('flag_maple',maple,(xx,fy-.062,1.31),'detail_red')
    # Open trays, a bucket and small blue bag under the two panels.
    for j in range(4):box('partition_storage_'+str(j),(fx-1.25+j*.64,fy,.16),(.48,.38,.25),'detail_blue' if j==0 else 'detail_black')
    cyl('partition_bucket',(fx+1.20,fy-.14,.24),.14,.40,'detail_white')
    cyl('partition_bucket_lid',(fx+1.20,fy-.14,.45),.15,.035,'detail_orange')
    try:
        from room_south_details import south_details,SOUTH_COLORS
        DETAIL_COLORS.update(SOUTH_COLORS);p.extend(south_details())
    except ImportError:pass
    # Apply the final video-anchor locations while preserving local detail.
    for part in p:
        name=part['name'];x,y,z=part['pos']
        if name.startswith(('detail_east_','detail_controller_','detail_steer_')):
            part['pos']=(x+.05,y-1.0,z)
        elif name.startswith('detail_locker_'):
            part['pos']=(x-.20,y-.80,z)
        elif name.startswith('detail_northwest_shop_') or name.startswith('detail_large_airframe_'):
            part['pos']=(x+.5,y-1.4,z)
    p = [part for part in p if not part['name'].startswith(('detail_northwest_shop_', 'detail_large_airframe_'))]
    from room_west_bays import bay_parts, BAY_COLORS
    DETAIL_COLORS.update(BAY_COLORS)
    p.extend(bay_parts())
    return p
