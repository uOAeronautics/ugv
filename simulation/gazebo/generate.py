"""Generate a photo-informed Freenove visual model; stdlib only, SI in SDF."""
from assembly import assembly, COLORS
from camera_model import add_camera_sensor
from environment import add_environment
from motor_physics import load_profile, add_rubber_surface, add_visual_material
from mass_properties import properties
from native_motor_parameters import apply_native_parameters
from pathlib import Path
import json
import hashlib
import argparse
import math
import xml.etree.ElementTree as E

HERE = Path(__file__).resolve().parent

def node(parent, tag, text=None, **attrs):
    el = E.SubElement(parent, tag, attrs)
    if text is not None:
        el.text = str(text)
    return el

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drone', action='store_true', help='Add the photo-informed drone docking target')
    parser.add_argument('--room', action='store_true', help='Add the photo-informed 24 by 12 metre lab')
    default_profile=HERE/'mecanum_native_profile_pilot02_v1.json'
    motors=parser.add_mutually_exclusive_group()
    motors.add_argument('--motor-profile',type=Path,default=default_profile if default_profile.exists() else None,help='Native motor/contact candidate JSON; defaults to pilot02-v1 when available')
    motors.add_argument('--nominal-motors',action='store_true',help='Use reference motor assumptions instead of the candidate calibration')
    parser.add_argument('--output', type=Path, default=HERE / 'freenove_world.sdf')
    args = parser.parse_args()
    p = json.loads((HERE / 'dimensions.json').read_text())
    motor=load_profile()
    empirical=json.loads((HERE/"real_robot_profile.json").read_text())
    motor["tire_slip_lateral_m_per_s_n_assumed"]=empirical["lateral_slip_m_per_s_per_n"]
    inch = .0254
    length, width, outer, diameter, overall = [p[k] * inch for k in
        ('chassis_length_in', 'chassis_width_in', 'overall_width_in', 'wheel_diameter_in', 'overall_length_in')]
    radius = diameter / 2
    tire = p.get('wheel_width_mm_assumed',(outer-width)*500)/1000
    track = p.get('wheel_track_mm_assumed',(outer-tire)*1000)/1000
    wheelbase = overall - diameter
    groups = assembly(p)
    masses, total = properties(p, groups)
    root = E.Element('sdf', version='1.9')
    world = node(root, 'world', name='freenove_lab')
    physics = node(world, 'physics', name='default', type='ignored')
    node(physics, 'max_step_size', .001)
    node(physics, 'real_time_factor', 1)
    for name, cls in [('physics', 'Physics'), ('user-commands', 'UserCommands'), ('scene-broadcaster', 'SceneBroadcaster'), ('imu', 'Imu')]:
        node(world, 'plugin', filename=f'gz-sim-{name}-system', name=f'gz::sim::systems::{cls}')
    sensors=node(world,'plugin',filename='gz-sim-sensors-system',name='gz::sim::systems::Sensors')
    node(sensors,'render_engine','ogre2')
    add_environment(world, node, floor=not args.room)
    if args.room:
        from room_model import add_room
        add_room(world, node)
    room_config = json.loads((HERE/'room_dimensions.json').read_text()) if args.room else {}
    if args.drone:
        from drone_model import add_drone
        add_drone(world, node, pose=room_config.get('drone_pose', (.75, 0, 0, 0, 0, 0)))
    model = node(world, 'model', name='freenove')
    node(model, 'pose', ' '.join(map(str,room_config.get('ugv_pose',(0,0,.001,0,0,0)))))

    for name, group in groups.items():
        link = node(model, 'link', name=name)
        node(link, 'pose', ' '.join(map(str, (*group['origin'], 0, 0, 0))))
        inertial = node(link, 'inertial')
        prop = masses[name]
        node(inertial, 'mass', prop['mass'])
        node(inertial, 'pose', ' '.join(map(str, (*prop['center'], 0, 0, 0))))
        inertia = node(inertial, 'inertia')
        for tag, value in zip(('ixx','iyy','izz','ixy','ixz','iyz'), prop['inertia']):
            node(inertia, tag, value)
        for part in group['parts']:
            for kind in (('visual','collision') if part['collision'] else ('visual',)):
                obj=node(link, kind, name=part['name']+'_'+kind)
                node(obj, 'pose', ' '.join(map(str, (*part['pos'],*part['rot']))))
                geom=node(obj, 'geometry')
                shape=node(geom, part['shape'])
                if part['shape']=='box':
                    node(shape,'size',' '.join(map(str,part['size'])))
                else:
                    node(shape,'radius',part['size'][0])
                    node(shape,'length',part['size'][1])
                if kind=='collision' and name!='chassis':
                    add_rubber_surface(obj,node,motor)
                if kind=='visual':
                    color=' '.join(map(str,(*COLORS[part['color']],1)))
                    mat=node(obj,'material')
                    node(mat,'ambient',color)
                    node(mat,'diffuse',color)
                    add_visual_material(mat,node,part['color'],motor)
        if name=='chassis':
            add_camera_sensor(link,node,p)
            imu_profile=json.loads((HERE/'imu_noise_profile.json').read_text())
            sensor=node(link,'sensor',name='nano_ideal_imu',type='imu')
            node(sensor,'pose',' '.join(map(str,(*imu_profile['position_body_m'],0,0,0))))
            node(sensor,'always_on','true')
            node(sensor,'update_rate',imu_profile['ideal_physics_rate_hz'])
            node(sensor,'topic','/model/freenove/imu/ideal')
            node(sensor,'imu')  # Ideal chassis axes; Nano mounting/noise applied once in imu_model.py.
        if name!='chassis':
            # A millimetre-scale equivalent radial tyre spring allows load to
            # distribute across all four tires instead of rigid diagonal rocking.
            carrier=node(model,'link',name=name+'_carrier')
            node(carrier,'pose',' '.join(map(str,(*group['origin'],0,0,0))))
            ci=node(carrier,'inertial')
            node(ci,'mass',masses[name+'_carrier']['mass'])
            tensor=node(ci,'inertia')
            for tag, value in zip(('ixx','iyy','izz','ixy','ixz','iyz'), masses[name+'_carrier']['inertia']):
                node(tensor,tag,value)
            spring=node(model,'joint',name=name+'_radial_compliance',type='prismatic')
            node(spring,'parent','chassis')
            node(spring,'child',name+'_carrier')
            sa=node(spring,'axis')
            node(sa,'xyz','0 0 1')
            sl=node(sa,'limit')
            node(sl,'lower',-motor['tire_radial_travel_m_assumed'])
            node(sl,'upper',motor['tire_radial_travel_m_assumed'])
            sd=node(sa,'dynamics')
            node(sd,'spring_reference',0)
            node(sd,'spring_stiffness',motor['tire_radial_stiffness_n_per_m_assumed'])
            node(sd,'damping',motor['tire_radial_damping_ns_per_m_assumed'])
            joint=node(model,'joint',name=name+'_joint',type='revolute')
            node(joint,'parent',name+'_carrier')
            node(joint,'child',name)
            axis=node(joint,'axis')
            node(axis,'xyz','0 1 0')
            limit=node(axis,'limit')
            node(limit,'lower','-1e16')
            node(limit,'upper','1e16')
            node(limit,'effort',motor['stall_nm'])
            node(limit,'velocity',motor['omega0'])
            dynamics=node(axis,'dynamics')
            node(dynamics,'friction',.0008)
            node(dynamics,'damping',.00002)
    from mecanum_geometry import add_passive_rollers
    add_passive_rollers(model,node,groups,radius=radius,width=tire)
    drive = node(model, 'plugin', filename='ugv_tt_motor', name='ugv::TTMotorSystem')
    node(drive,'mecanum','true')
    node(drive,'wheelbase',wheelbase)
    node(drive, 'wheel_separation', track)
    node(drive, 'wheel_radius', radius)
    node(drive,'stall_torque',motor['stall_nm'])
    node(drive,'brake_damping',motor['brake_damping_nm_s_per_rad_assumed'])
    node(drive,'brake_resistance',motor['brake_resistance_nm_assumed'])
    node(drive,'steering_brake_rise_tau',motor['steering_brake_rise_tau_s_empirical'])
    node(drive,'no_load_speed',motor['omega0'])
    node(drive,'velocity_feedback',motor['velocity_feedback_gain_assumed'])
    node(drive,'velocity_integral',motor['velocity_integral_gain_assumed'])
    node(drive,'empirical_pwm_map','\n'+'\n'.join(f"{r['forward']} {r['turn']} {r['pwm']} {r['effective_duty']:.9f}" for r in empirical['pwm_map']))
    node(drive,'pwm_rise_tau','\n'+'\n'.join(f"{r['forward']} {r['turn']} {r['tau_s']:.9f}" for r in empirical['pwm_rise_tau']))
    odom=node(model,'plugin',filename='gz-sim-odometry-publisher-system',name='gz::sim::systems::OdometryPublisher')
    node(odom,'dimensions',2)
    node(odom,'odom_topic','/model/freenove/odometry')
    if args.motor_profile is not None and not args.nominal_motors:
        profile=json.loads(args.motor_profile.read_text())
        apply_native_parameters(root,profile)
        node(drive,'calibration_profile',args.motor_profile.name)
        node(drive,'calibration_status',profile.get('status','CANDIDATE_NOT_FULLY_VALIDATED'))
        node(drive,'calibration_profile_sha256',hashlib.sha256(args.motor_profile.read_bytes()).hexdigest())
        print('Motor candidate: '+str(args.motor_profile)+'; '+profile.get('status','CANDIDATE_NOT_FULLY_VALIDATED'))
    else:
        node(drive,'calibration_status','NOMINAL_REFERENCE_PARAMETERS_UNCALIBRATED')
    E.indent(root)
    E.ElementTree(root).write(args.output, encoding='utf-8', xml_declaration=True)
    print(f"Mass: {total['mass']*1000:.1f} g; assembly COM (mm): {[round(v*1000, 3) for v in total['center']]}")
    print(f'Generated: wheelbase={wheelbase*1000:.2f}mm, track={track*1000:.2f}mm, tire width={tire*1000:.2f}mm')

if __name__ == '__main__':
    main()
