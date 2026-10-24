"""Regression checks for motor units, generated force actuation, and tire axes."""
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as E
from motor_physics import load_profile

HERE=Path(__file__).resolve().parent

class MotorModelTest(unittest.TestCase):
    def setUp(self):
        self.profile=load_profile()
        self.model=E.parse(HERE/'freenove_world.sdf').find('./world/model[@name="freenove"]')

    def test_reference_units_are_output_shaft_units(self):
        self.assertAlmostEqual(self.profile['stall_nm'],.0784532)
        self.assertAlmostEqual(self.profile['omega0'],250*2*math.pi/60)

    def test_velocity_actuator_cannot_override_motor_forces(self):
        plugins={p.get('name') for p in self.model.findall('plugin')}
        self.assertIn('ugv::TTMotorSystem',plugins)
        self.assertNotIn('gz::sim::systems::DiffDrive',plugins)
        joints=[j for j in self.model.findall('./joint[@type="revolute"]') if '_roller_' not in j.get('name')]
        self.assertEqual(len(joints),4)
        for joint in joints:
            self.assertAlmostEqual(float(joint.findtext('axis/limit/effort')),self.profile['stall_nm'])

    def test_mecanum_contacts_are_passive_independent_rollers(self):
        springs=self.model.findall('./joint[@type="prismatic"]')
        self.assertEqual(len(springs),4)
        rollers=[l for l in self.model.findall('link') if '_roller_' in l.get('name')]
        self.assertEqual(len(rollers),32)
        for link in rollers:
            contact=link.find('collision/surface/friction/ode')
            self.assertGreater(float(contact.findtext('mu')),0)
            self.assertEqual(contact.findtext('mu'),contact.findtext('mu2'))
            joint=self.model.find('./joint[@name="'+link.get('name')+'_joint"]')
            self.assertEqual(joint.get('type'),'revolute')
            self.assertEqual(joint.findtext('axis/xyz'),'0 0 1')
            self.assertLess(float(joint.findtext('axis/dynamics/friction')),1e-4)
        drive=self.model.find('./plugin[@name="ugv::TTMotorSystem"]')
        self.assertEqual(drive.findtext('mecanum'),'true')

if __name__=='__main__':
    unittest.main()
