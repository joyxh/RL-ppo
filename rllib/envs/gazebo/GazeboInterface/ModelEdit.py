#!/usr/bin/env python3
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '4/12/2017'
__copyright__ = "RR"
__all__ = [
    "createBoxModel",
    "createBoxVisual",
    "createCylinderModel",
    "createCylinderVisual",
    "createSphereModel",
    "createSphereVisual",
]

import lxml.etree as ltr


def createCylinderVisual(radius, length, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="cylinder")
    Link = ltr.SubElement(MODEL, "link", name="cylinder_link")
    __CylinderVisual(Link, "cylinder", radius, length, color)
    return ltr.tostring(ROOT, pretty_print=True)


def createSphereModel(mass, radius, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="sphere")
    Link = ltr.SubElement(MODEL, "link", name="sphere_link")
    __SphereInertial(Link, mass, radius)
    __SphereCollision(Link, "sphere", radius)
    __SphereVisual(Link, "sphere", radius, color)
    return ltr.tostring(ROOT, pretty_print=True)


def createSphereVisual(radius, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="sphere")
    Link = ltr.SubElement(MODEL, "link", name="sphere_link")
    __SphereVisual(Link, "sphere", radius, color)
    return ltr.tostring(ROOT, pretty_print=True)


def __BoxVisual(root, visualName, x, y, z, color):
    VISUAL = ltr.SubElement(root, "visual", name=visualName + "_visual")
    GEOMETRY = ltr.SubElement(VISUAL, "geometry")
    BOX = ltr.SubElement(GEOMETRY, "box")
    SIZE = ltr.SubElement(BOX, "size")
    SIZE.text = str(x) + " " + str(y) + " " + str(z)
    MATERIAL = ltr.SubElement(VISUAL, "material")
    SCRIPT = ltr.SubElement(MATERIAL, "script")
    URI = ltr.SubElement(SCRIPT, "uri")
    URI.text = "model://materials/scripts/Pi.material"
    NAME = ltr.SubElement(SCRIPT, "name")
    NAME.text = "Pi/" + str(color)
    return root


def __BoxCollision(root, collisionName, x, y, z):
    COLLISION = ltr.SubElement(root, "collision", name=collisionName + "_collision")
    GEOMETRY = ltr.SubElement(COLLISION, "geometry")
    BOX = ltr.SubElement(GEOMETRY, "box")
    SIZE = ltr.SubElement(BOX, "size")
    SIZE.text = str(x) + " " + str(y) + " " + str(z)
    return root


def __BoxInertial(root, mass, x, y, z):
    INERTIAL = ltr.SubElement(root, "inertial")
    MASS = ltr.SubElement(INERTIAL, "mass")
    MASS.text = str(mass)
    INERTIA = ltr.SubElement(INERTIAL, "inertia")
    IXX = ltr.SubElement(INERTIA, "ixx")
    IXX.text = str(mass / 12.0 * (y ** 2 + z ** 2))
    IXY = ltr.SubElement(INERTIA, "ixy")
    IXY.text = str(0)
    IXZ = ltr.SubElement(INERTIA, "ixz")
    IXZ.text = str(0)
    IYY = ltr.SubElement(INERTIA, "iyy")
    IYY.text = str(mass / 12.0 * (x ** 2 + z ** 2))
    IYZ = ltr.SubElement(INERTIA, "iyz")
    IYZ.text = str(0)
    IZZ = ltr.SubElement(INERTIA, "izz")
    IZZ.text = str(mass / 12.0 * (y ** 2 + x ** 2))
    return root


def __CylinderCollision(root, collisionName, radius, length):
    COLLISION = ltr.SubElement(root, "collision", name=collisionName + "_collision")
    GEOMETRY = ltr.SubElement(COLLISION, "geometry")
    CYLINDER = ltr.SubElement(GEOMETRY, "cylinder")
    RADIUS = ltr.SubElement(CYLINDER, "radius")
    LENGTH = ltr.SubElement(CYLINDER, "length")
    RADIUS.text = str(radius)
    LENGTH.text = str(length)
    return root


def __CylinderVisual(root, visualName, radius, length, color):
    VISUAL = ltr.SubElement(root, "visual", name=visualName + "_visual")
    GEOMETRY = ltr.SubElement(VISUAL, "geometry")
    CYLINDER = ltr.SubElement(GEOMETRY, "cylinder")
    RADIUS = ltr.SubElement(CYLINDER, "radius")
    LENGTH = ltr.SubElement(CYLINDER, "length")
    RADIUS.text = str(radius)
    LENGTH.text = str(length)
    MATERIAL = ltr.SubElement(VISUAL, "material")
    SCRIPT = ltr.SubElement(MATERIAL, "script")
    URI = ltr.SubElement(SCRIPT, "uri")
    URI.text = "model://materials/scripts/Pi.material"
    NAME = ltr.SubElement(SCRIPT, "name")
    NAME.text = "Pi/" + str(color)
    return root


def __CylinderInertial(root, mass, radius, length):
    INERTIAL = ltr.SubElement(root, "inertial")
    MASS = ltr.SubElement(INERTIAL, "mass")
    MASS.text = str(mass)
    INERTIA = ltr.SubElement(INERTIAL, "inertia")
    IXX = ltr.SubElement(INERTIA, "ixx")
    IXX.text = str(1.0 / 12.0 * mass * (3 * radius ** 2 + length ** 2))
    IXY = ltr.SubElement(INERTIA, "ixy")
    IXY.text = str(0)
    IXZ = ltr.SubElement(INERTIA, "ixz")
    IXZ.text = str(0)
    IYY = ltr.SubElement(INERTIA, "iyy")
    IYY.text = str(1.0 / 12.0 * mass * (3 * radius ** 2 + length ** 2))
    IYZ = ltr.SubElement(INERTIA, "iyz")
    IYZ.text = str(0)
    IZZ = ltr.SubElement(INERTIA, "izz")
    IZZ.text = str(1.0 / 2.0 * mass * length ** 2)
    return root


def __SphereCollision(root, collisionName, radius):
    COLLISION = ltr.SubElement(root, "collision", name=collisionName + "_collision")
    GEOMETRY = ltr.SubElement(COLLISION, "geometry")
    SPHERE = ltr.SubElement(GEOMETRY, "sphere")
    RADIUS = ltr.SubElement(SPHERE, "radius")
    RADIUS.text = str(radius)


def __SphereVisual(root, visualName, radius, Colour_name):
    VISUAL = ltr.SubElement(root, "visual", name=visualName + "_visual")
    GEOMETRY = ltr.SubElement(VISUAL, "geometry")
    SPHERE = ltr.SubElement(GEOMETRY, "sphere")
    RADIUS = ltr.SubElement(SPHERE, "radius")
    RADIUS.text = str(radius)
    MATERIAL = ltr.SubElement(VISUAL, "material")
    SCRIPT = ltr.SubElement(MATERIAL, "script")
    URI = ltr.SubElement(SCRIPT, "uri")
    URI.text = "model://materials/scripts/Pi.material"
    NAME = ltr.SubElement(SCRIPT, "name")
    NAME.text = "Pi/" + str(Colour_name)


def __SphereInertial(root, mass, radius):
    INERTIAL = ltr.SubElement(root, "inertial")
    MASS = ltr.SubElement(INERTIAL, "mass")
    MASS.text = str(mass)
    INERTIA = ltr.SubElement(INERTIAL, "inertia")
    IXX = ltr.SubElement(INERTIA, "ixx")
    IXX.text = str(2.0 / 3.0 * mass * radius ** 2)
    IXY = ltr.SubElement(INERTIA, "ixy")
    IXY.text = str(0)
    IXZ = ltr.SubElement(INERTIA, "ixz")
    IXZ.text = str(0)
    IYY = ltr.SubElement(INERTIA, "iyy")
    IYY.text = str(2.0 / 3.0 * mass * radius ** 2)
    IYZ = ltr.SubElement(INERTIA, "iyz")
    IYZ.text = str(0)
    IZZ = ltr.SubElement(INERTIA, "izz")
    IZZ.text = str(2.0 / 3.0 * mass * radius ** 2)
    return root


def createBoxModel(mass, x, y, z, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="box")
    Link = ltr.SubElement(MODEL, "link", name="box_link")
    __BoxInertial(Link, mass, x, y, z)
    __BoxCollision(Link, "box", x, y, z)
    __BoxVisual(Link, "box", x, y, z, color)
    return ltr.tostring(ROOT, pretty_print=True)


def createBoxVisual(x, y, z, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="box")
    Link = ltr.SubElement(MODEL, "link", name="box_link")
    __BoxVisual(Link, "box", x, y, z, color)
    return ltr.tostring(ROOT, pretty_print=True)


def createCylinderModel(mass, radius, length, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="Cylinder")
    Link = ltr.SubElement(MODEL, "link", name="Cylinder_link")
    __CylinderInertial(Link, mass, radius, length)
    __CylinderCollision(Link, "cylinder", radius, length)
    __CylinderVisual(Link, "cylinder", radius, length, color)
    return ltr.tostring(ROOT, pretty_print=True)
