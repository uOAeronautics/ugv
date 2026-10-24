"""Apply identified native motor/contact parameters to an already built SDF.

Stdlib only, shared by the world generator and offline replay. Parameter metadata
is preserved in the JSON profile; only explicitly supported fields enter SDF.
"""
import math
import xml.etree.ElementTree as E

MOTOR_FIELDS=('wheel_duty_exponent','wheel_voltage_gain','wheel_reverse_voltage_gain',
              'wheel_deadzone','wheel_drive_tau','wheel_brake_tau','wheel_gain_stddev',
              'random_seed','no_load_speed','stall_torque','brake_damping','brake_resistance')


def apply_native_parameters(root, profile):
    """Mutate an ElementTree/Element; return the exact supported parameter set."""
    params=profile.get('gazebo_motor_parameters',profile)
    drive=root.find('.//model[@name="freenove"]/plugin[@name="ugv::TTMotorSystem"]')
    if drive is None:raise ValueError('Native motor plugin missing')
    used={}
    for key in MOTOR_FIELDS:
        if key not in params:continue
        value=params[key]
        if key.startswith('wheel_'):
            if not isinstance(value,list) or len(value)!=4:raise ValueError(key+' needs FL FR RL RR values')
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in value):raise ValueError('Invalid '+key)
            if key=='wheel_duty_exponent' and any(x<=0 or x>3 for x in value):raise ValueError('Invalid duty exponent')
            maximum={'wheel_deadzone':.95,'wheel_gain_stddev':1.,'wheel_duty_exponent':3.}.get(key,5.)
            if any(x>maximum for x in value):raise ValueError('Out of bounds '+key)
        elif key=='random_seed':
            if type(value) is not int or not 0<=value<2**32:raise ValueError('Invalid random seed')
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (key in ('no_load_speed','stall_torque') and value==0):raise ValueError('Invalid '+key)
        child=drive.find(key)
        if child is None:child=E.SubElement(drive,key)
        child.text=' '.join(map(str,value)) if isinstance(value,list) else str(value)
        used[key]=value
    model=root.find('.//model[@name="freenove"]')
    for joint in model.findall('joint'):
        name=joint.get('name','')
        if name.endswith('_wheel_joint'):
            if 'no_load_speed' in params:joint.find('axis/limit/velocity').text=str(params['no_load_speed'])
            if 'stall_torque' in params:joint.find('axis/limit/effort').text=str(params['stall_torque'])
        if '_roller_' not in name:continue
        for field in ('friction','damping'):
            key='roller_bearing_'+field
            if key not in params:continue
            value=params[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=.01:raise ValueError('Invalid '+key)
            joint.find('axis/dynamics/'+field).text=str(value)
            used[key]=value
    return used


def extract_native_parameters(root):
    """Record explicit native parameters embedded in an existing world."""
    drive=root.find('.//model[@name="freenove"]/plugin[@name="ugv::TTMotorSystem"]')
    if drive is None:raise ValueError('Native motor plugin missing')
    params={}
    for key in MOTOR_FIELDS:
        text=drive.findtext(key)
        if text is None:continue
        params[key]=list(map(float,text.split())) if key.startswith('wheel_') else int(text) if key=='random_seed' else float(text)
    for field in ('friction','damping'):
        values=[float(j.findtext('axis/dynamics/'+field)) for j in root.find('.//model[@name="freenove"]').findall('joint') if '_roller_' in j.get('name','')]
        if values and all(v==values[0] for v in values):params['roller_bearing_'+field]=values[0]
    return params
