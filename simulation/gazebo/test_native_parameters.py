"""Motor calibration must update physical constraints as well as plugin inputs."""
import unittest
import xml.etree.ElementTree as E
from native_motor_parameters import apply_native_parameters


class NativeParametersTest(unittest.TestCase):
    def setUp(self):
        self.world=E.fromstring('''<sdf><world><model name="freenove">
        <plugin name="ugv::TTMotorSystem"><mecanum>true</mecanum></plugin>
        <joint name="front_left_wheel_joint"><axis><limit><velocity>26</velocity><effort>.078</effort></limit></axis></joint>
        <joint name="front_left_wheel_roller_0_joint"><axis><dynamics><damping>.000002</damping><friction>.000001</friction></dynamics></axis></joint>
        </model></world></sdf>''')

    def test_fitted_motor_changes_cannot_leave_stale_joint_limits(self):
        applied=apply_native_parameters(self.world,dict(gazebo_motor_parameters=dict(no_load_speed=44,stall_torque=.09,wheel_duty_exponent=[.6]*4)))
        self.assertEqual(applied['no_load_speed'],44)
        self.assertEqual(self.world.findtext('.//joint[@name="front_left_wheel_joint"]/axis/limit/velocity'),'44')
        self.assertEqual(self.world.findtext('.//joint[@name="front_left_wheel_joint"]/axis/limit/effort'),'0.09')
        self.assertEqual(self.world.findtext('.//plugin/wheel_duty_exponent'),'0.6 0.6 0.6 0.6')

    def test_passive_bearing_load_is_applied_to_roller_not_hub_joint(self):
        apply_native_parameters(self.world,dict(roller_bearing_damping=.00003))
        self.assertEqual(self.world.findtext('.//joint[@name="front_left_wheel_roller_0_joint"]/axis/dynamics/damping'),'3e-05')
        self.assertIsNone(self.world.find('.//joint[@name="front_left_wheel_joint"]/axis/dynamics'))

    def test_other_room_or_drone_models_are_not_modified(self):
        other=E.SubElement(self.world.find('world'),'model',name='other_robot')
        other.append(E.fromstring('<joint name="front_left_wheel_roller_0_joint"><axis><dynamics><damping>.000001</damping></dynamics></axis></joint>'))
        apply_native_parameters(self.world,dict(roller_bearing_damping=.00003))
        self.assertEqual(other.findtext('joint/axis/dynamics/damping'),'.000001')

    def test_invalid_domain_randomization_and_nonfinite_parameters_rejected(self):
        for profile in (dict(wheel_gain_stddev=[2]*4),dict(no_load_speed=float('nan')),dict(wheel_duty_exponent=[0]*4),dict(roller_bearing_damping=-.1)):
            with self.subTest(profile=profile),self.assertRaises(ValueError):apply_native_parameters(self.world,profile)

if __name__=='__main__':unittest.main()
