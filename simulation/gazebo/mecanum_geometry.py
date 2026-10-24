"""Estimated 63.5 x 32 mm X-pattern mecanum wheels, SI units.

Eight independently revolving 45 degree rollers make lateral compliance emerge
from contact mechanics, rather than imposing a chassis velocity. Dimensions are
photo estimates, not measured geometry. Positive hub rotation about +Y drives +X.
"""
import math

WHEEL_ORDER = ('front_left', 'front_right', 'rear_left', 'rear_right')
HANDEDNESS = dict(zip(WHEEL_ORDER, (1, -1, -1, 1)))


def wheel_parts(radius=.03175, width=.032):
    parts=[]
    def add(name, shape, pos, size, color, rot=(0,0,0), collision=False):
        parts.append(dict(name=name,shape=shape,pos=pos,size=size,color=color,rot=rot,collision=collision))
    add('hub_core','cylinder',(0,0,0),(.015,width*.65),'rubber',(math.pi/2,0,0),True)
    for side in (-1,1):
        y=side*width*.36
        add(f'alloy_plate_{side}','cylinder',(0,y,0),(.0235,.0015),'silver',(math.pi/2,0,0))
        add(f'axle_boss_{side}','cylinder',(0,y+side*.002,0),(.007,.004),'silver',(math.pi/2,0,0))
        for j in range(8):
            a=j*math.tau/8
            add(f'plate_bolt_{side}_{j}','cylinder',(.018*math.sin(a),y+side*.001,.018*math.cos(a)),(.0016,.0018),'chip',(math.pi/2,0,0))
    return parts


def add_passive_rollers(model, node, groups, radius=.03175, width=.032,
                        roller_mass=.002, friction=.85):
    """Append 32 roller links/joints to an existing four-hub SDF model.

    groups use assembly's wheel origins. Hub links must be named
    front_left_wheel etc. Roller inertias are separate from hub mass.
    Call after emitting hub/carrier links and before writing the SDF.
    """
    rr=.006; length=.024; ring=radius-rr
    for tag, hand in HANDEDNESS.items():
        hub=tag+'_wheel'; origin=groups[hub]['origin']
        for i in range(8):
            a=i*math.tau/8
            axis=(math.cos(a)/math.sqrt(2),hand/math.sqrt(2),-math.sin(a)/math.sqrt(2))
            # Rz(yaw) Ry(pitch) sends cylinder-local Z onto the axle.
            yaw=math.atan2(axis[1],axis[0]); pitch=math.acos(axis[2])
            pos=(origin[0]+ring*math.sin(a),origin[1],origin[2]+ring*math.cos(a))
            name=f'{hub}_roller_{i}'
            link=node(model,'link',name=name)
            node(link,'pose',' '.join(map(str,(*pos,0,pitch,yaw))))
            inertial=node(link,'inertial');node(inertial,'mass',roller_mass)
            tensor=node(inertial,'inertia')
            transverse=roller_mass*(3*rr*rr+length*length)/12
            for key,value in zip(('ixx','iyy','izz','ixy','ixz','iyz'),(transverse,transverse,roller_mass*rr*rr/2,0,0,0)):
                node(tensor,key,value)
            for kind in ('visual','collision'):
                obj=node(link,kind,name=name+'_'+kind)
                shape=node(node(obj,'geometry'),'cylinder')
                node(shape,'radius',rr);node(shape,'length',length)
                if kind=='visual':
                    mat=node(obj,'material');node(mat,'ambient','.018 .02 .023 1');node(mat,'diffuse','.018 .02 .023 1')
                else:
                    ode=node(node(node(obj,'surface'),'friction'),'ode')
                    node(ode,'mu',friction);node(ode,'mu2',friction)
            joint=node(model,'joint',name=name+'_joint',type='revolute')
            node(joint,'parent',hub);node(joint,'child',name)
            axle=node(joint,'axis');node(axle,'xyz','0 0 1')
            dynamics=node(axle,'dynamics');node(dynamics,'damping',.000002);node(dynamics,'friction',.000001)
    return 32


def inverse_kinematics(vx, vy, yaw_rate, radius=.03175, wheelbase=.1397, track=.14):
    k=(wheelbase+track)/2
    return tuple((vx-hand*vy+(-1 if i%2==0 else 1)*k*yaw_rate)/radius
                 for i,hand in enumerate(HANDEDNESS.values()))
