"""Photo-informed visual parts shared by Gazebo and Blender. All values in metres.

X points toward the camera/front, Y left, Z up. Small components are approximate.
"""
import math
from tt_geometry import add_tt_motor
from camera_model import camera_geometry, load_camera

COLORS = {
    'pcb': (.018, .024, .03), 'green': (.025, .26, .10),
    'blue': (.025, .11, .32), 'rubber': (.016, .019, .022),
    'yellow_dark': (.62, .32, .006), 'yellow': (.95, .55, .008), 'silver': (.52, .56, .59),
    'gold': (.56, .34, .07), 'chip': (.012, .014, .018),
    'orange': (.78, .28, .025), 'white': (.82, .84, .79),
    'red': (.55, .018, .012), 'lens': (.02, .045, .065),
}

def assembly(p):
    inch = .0254
    L=p['chassis_length_in']*inch
    W=p['chassis_width_in']*inch
    R=p['wheel_diameter_in']*inch/2
    tire=p.get('wheel_width_mm_assumed',(p['overall_width_in']*inch-W)*500)/1000
    track=p.get('wheel_track_mm_assumed',(p['overall_width_in']*inch-tire)*1000)/1000
    wb=p['overall_length_in']*inch-2*R
    board=p['main_board_height_in']*inch
    thick=p['board_thickness_mm_assumed']/1000
    motorbottom=p['motor_bottom_clearance_in']*inch
    groups={'chassis': {'origin': (0,0,0), 'parts': []}}

    def part(group,name,shape,pos,size,color,rot=(0,0,0),collision=False):
        groups[group]['parts'].append(dict(name=name,shape=shape,pos=pos,size=size,
                                         color=color,rot=rot,collision=collision))
    def box(name,pos,size,color,rot=(0,0,0),collision=False):
        part('chassis',name,'box',pos,size,color,rot,collision)
    def cyl(name,pos,r,h,color,rot=(0,0,0),collision=False):
        part('chassis',name,'cylinder',pos,(r,h),color,rot,collision)

    box('main_board',(0,0,board-thick/2),(L,W,thick),'pcb',collision=True)
    # Four TT gearmotors: yellow gearbox and inboard silver can on each side.
    for end,x in [('front',wb/2),('rear',-wb/2)]:
        direction=-1 if end=='front' else 1
        for side,s in [('left',1),('right',-1)]:
            tag=end+'_'+side
            y=s*(W/2-.011)
            add_tt_motor(groups['chassis']['parts'], tag, x, y, direction, s, motorbottom, R)
            group=tag+'_wheel'
            groups[group]={'origin':(x,s*track/2,R),'parts':[]}
            from mecanum_geometry import wheel_parts
            groups[group]['parts'].extend(wheel_parts(R,tire))

    # Estimated low central LiPo envelope; mass and placement in mass_distribution.json.
    box('lipo_pack',(-.002,0,.030),(.085,.044,.020),'silver',collision=True)
    for x in (-.042,.038):
        box('lipo_wrap_'+str(x),(x,0,.030),(.005,.0445,.0205),'yellow')
    box('battery_retaining_strap',(-.002,0,.0195),(.012,.046,.001),'chip')
    for y in (-.0225,.0225):
        box('battery_strap_side_'+str(y),(-.002,y,.031),(.012,.001,.023),'chip')
    box('battery_power_lead',(.044,.008,.034),(.013,.002,.002),'red')
    box('battery_connector',(.052,.008,.034),(.006,.006,.005),'yellow')

    # Raised Raspberry Pi near the rear; positions estimated from top and side views.
    pi_z=board+.016
    for x in (-.082,-.020):
        for y in (-.025,.025):
            cyl('standoff_'+str((x,y)),(x,y,board+.007),.002,.014,'gold')
            cyl('pi_screw_'+str((x,y)),(x,y,pi_z+.0013),.0025,.0015,'silver')
    box('raspberry_pi',(-.048,0,pi_z),(.085,.056,.0016),'green',collision=True)
    box('pi_processor',(-.045,-.002,pi_z+.0025),(.015,.015,.003),'chip')
    for i,y in enumerate((-.018,.004)):
        box('usb_stack_'+str(i),(-.082,y,pi_z+.008),(.018,.016,.015),'silver')
        box('usb_opening_'+str(i),(-.0913,y,pi_z+.008),(.0008,.012,.010),'chip')
    box('ethernet',(-.08,.024,pi_z+.008),(.021,.016,.015),'silver')
    for x in (-.059,-.040):
        box('hdmi_'+str(x),(x,-.026,pi_z+.003),(.009,.007,.005),'silver')
    box('gpio_header',(-.041,.022,pi_z+.005),(.050,.005,.008),'chip')
    # Arduino Nano on tall pin headers over the Pi.
    nano_z=pi_z+.022
    box('nano_board',(-.054,.003,nano_z),(.043,.018,.0016),'blue',collision=True)
    box('nano_radio',(-.047,.002,nano_z+.0025),(.012,.010,.003),'silver')
    box('nano_usb',(-.075,.003,nano_z+.003),(.006,.008,.004),'silver')
    for x in range(12):
        for y in (-.005,.011):
            box('nano_pin_'+str((x,y)),(-.072+x*.003,y,nano_z-.009),(.0006,.0006,.018),'gold')
    # New upright white 40x40mm camera stand, 40mm behind board front.
    camera=load_camera();mount=camera['mount'];geometry=camera_geometry(p,camera)
    cx,cz=geometry['stand_x'],geometry['center_z']
    depth,rail=mount['depth_m_assumed'],mount['rail_width_m_assumed']
    for side in (-1,1):
        box('camera_stand_side_'+str(side),(cx,side*(mount['width_m']-rail)/2,cz),
            (depth,rail,mount['height_m']),'white',collision=True)
        box('camera_stand_edge_'+str(side),(cx,0,cz+side*(mount['height_m']-rail)/2),
            (depth,mount['width_m']-2*rail,rail),'white',collision=True)
    box('camera_stand_foot',(cx-.003,0,board+.001),( .012,.040,.002),'white',collision=True)
    box('camera_pcb',(cx+.003,0,cz),(.0016,.035,.035),'green',collision=True)
    box('camera_lens_base',(cx+.006,0,cz),(.004,.017,.017),'chip')
    cyl('camera_lens_barrel',(cx+.010,0,cz),.007,.0104,'chip',(0,math.pi/2,0),True)
    cyl('camera_glass',(geometry['optical_xyz'][0]-.0003,0,cz),.0055,.0006,'lens',(0,math.pi/2,0))
    for y in (-.0145,.0145):
        for z in (-.0145,.0145):
            cyl('camera_screw_'+str((y,z)),(cx+.0045,y,cz+z),.0018,.0014,'silver',(0,math.pi/2,0))
    # Retained servo geometry sits behind the fixed stand; position is estimated.
    servo_x=cx-.020
    box('front_servo',(servo_x,0,board+.009),(.023,.013,.017),'chip',collision=True)
    cyl('servo_pivot',(servo_x,0,board+.019),.003,.005,'white')
    box('servo_horn',(servo_x,0,board+.022),(.004,.022,.002),'white')
    # Photo-informed ribbon loop rises from the Pi to the stand's top connector.
    path=[(-.015,pi_z+.003),(-.003,board+.067),(.023,board+.070),(cx-.004,board+.039)]
    for i,((ax,az),(bx,bz)) in enumerate(zip(path,path[1:])):
        box('camera_ribbon_'+str(i),((ax+bx)/2,0,(az+bz)/2),
            (math.hypot(bx-ax,bz-az),.015,.0003),'orange',(0,-math.atan2(bz-az,bx-ax),0))
    cyl('buzzer',(.071,-.030,board+.0055),.006,.011,'chip')
    cyl('buzzer_label',(.071,-.030,board+.0112),.0053,.0003,'white')
    for x in (-.026,-.011):
        box('switch_base_'+str(x),(x,-.037,board+.003),(.009,.008,.005),'white')
        box('switch_blue_'+str(x),(x,-.037,board+.006),(.004,.004,.002),'blue')
    for x in (-.082,-.048,-.014,.025,.080):
        for y in (-.039,.039):
            box('led_'+str((x,y)),(x,y,board+.001),(.0035,.0035,.0015),'white')
    for x in (-.085,.078):
        for y in (-.033,.033):
            cyl('board_screw_'+str((x,y)),(x,y,board+.001),.0025,.002,'silver')
    for i in range(12):
        box('front_gpio_'+str(i),(-.003+i*.0025,.029,board+.004),(.001,.002,.007),'gold')
    return groups
