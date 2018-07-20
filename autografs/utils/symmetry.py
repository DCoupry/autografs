#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.4'
__status__  = "beta"

import scipy.spatial
import ase.visualize
import ase.build
import itertools
import numpy

from autografs.utils.operations import rotation, reflection, inertia

import logging
logger = logging.getLogger(__name__)

def is_valid_op(mol,
                symmop,
                epsilon=0.1):
    """Return True if the symmop is a valid symmetry 
    operation for mol.
    mol -- ASE Atoms object
    symmol -- 3x3 numpy array
    """
    distances = []
    mol0 = mol.copy()
    mol1 = mol.copy()
    mol1.positions = mol1.positions.dot(symmop)
    workmol = mol0+mol1
    other_indices = list(range(len(mol0),len(workmol),1))
    for atom_index in range(0,len(mol0),1):
        dist = workmol.get_distances(atom_index,other_indices,mic=False)
        distances.append(numpy.amin(dist))
    distances = numpy.array(distances)
    return (distances<epsilon).all()

def get_potential_axes(mol):
    """Return all potential symmetry axes.
    faces, nodes, midway points, etc.
    """
    potential_axes = []
    try:
        # full analysis needed
        qhull = scipy.spatial.ConvexHull(mol.get_positions())
        logger.debug("Using Convex Hull algorithm for axis detection.")
        for equation in qhull.equations:
            # normal of simplices
            axis = equation[0:3]
            potential_axes.append(axis)
        for node in qhull.vertices:
            # exterior points
            axis = qhull.points[node]
            norm = numpy.linalg.norm(axis)
            potential_axes.append(axis)
        for simplex in qhull.simplices:
            for side in itertools.combinations(list(simplex), 2):
                # middle of side
                side = numpy.array(side)
                axis = qhull.points[side].mean(axis=0)
                potential_axes.append(axis)
                # perpendicular to side
                a0,a1 = qhull.points[side]  
                axis = numpy.cross(a0,a1)
                potential_axes.append(axis)
    except scipy.spatial.qhull.QhullError:
        logger.debug("Planar connectivity detected.")
        #coplanarity detected
        positions = mol.get_positions()
        potential_axes += [axis for axis in positions]
        # since coplanar, any two simplices describe the plane
        for side in itertools.combinations(list(positions), 2):
            # middle of side
            axis = numpy.array(side).mean(axis=0)
            potential_axes.append(axis)             
            axis = numpy.cross(side[0],side[1])
            potential_axes.append(axis)     
        for a0,a1 in itertools.combinations(list(potential_axes),2):
            axis = numpy.cross(a0,a1)
            potential_axes.append(axis)     
    potential_axes = numpy.array(potential_axes)
    norm = numpy.linalg.norm(potential_axes,axis=1)
    norm[norm<1e-3] = 1.0
    potential_axes /= norm[:,None]
    # remove zero norms
    mask = numpy.isnan(potential_axes).any(axis=1)
    potential_axes = potential_axes[~mask]
    axes = unique_axes(potential_axes)
    return axes

def unique_axes(potential_axes,epsilon=0.1):
    """Return non colinear potential_axes only"""
    axes = [potential_axes[0],]
    indices = []
    for axis in potential_axes[1:]:
        colinear = False
        for seen_axis in axes:
            dot = abs(seen_axis.dot(axis))
            if dot>1.0-epsilon:
                colinear = True
                break
        if not colinear:
            axes.append(axis)
    return numpy.asarray(axes)

def get_symmetry_elements(mol, 
                          max_order=8,
                          epsilon=0.1):
    """Return an array counting the found symmetries
    of the object, up to axes of order max_order.
    Enables fast comparison of compatibility between objects:
    if array1 - array2 > 0, that means object1 fits the slot2 
    """
    if len(mol)==1:
        logger.debug("Point-symmetry detected.")
        symmetries = numpy.array([0,0,0,0,0,1])
        return symmetries
    if len(mol)==2:
        logger.debug("Linear connectivity detected.")
        # simplest, linear case
        symmetries = numpy.array([1,1,0,1,1,2])
        return symmetries
    mol.center(about=0)
    # ensure there is a sufficient distance between connections
    dist = mol.get_all_distances().mean()
    if dist<10.0:
        alpha = 10.0/dist
        mol.positions = mol.positions.dot(numpy.eye(3)*alpha)
    logger.debug("DIST {}".format(dist))
    axes = get_potential_axes(mol)
    # array for the operations:
    # size = (1inversion+rotationorders+rotinvorders+2planes+1multiplicity)
    symmetries = numpy.zeros(4+2*max_order)
    # inversion
    inv = -1.0*numpy.eye(3)
    has_inv = is_valid_op(mol,inv)
    symmetries[0] += int(has_inv)
    # rotations:
    principal_order = 1
    principal_axes  = []
    for axis in axes:
        for order in range(2,max_order+1):
            rot = rotation(axis,order)
            has_rot = is_valid_op(mol,rot)
            symmetries[order-1]+=int(has_rot)
            if has_rot:
                logger.debug("Detected: C{order}".format(order=order))
            # bookkeeping for the planes
            if has_rot and order>principal_order:
                principal_order = order
                principal_axes  = [axis,]
            elif has_rot and order==principal_order:
                principal_axes.append(axis)
    # planes
    for axis in axes:
        ref = reflection(axis)
        has_ref = is_valid_op(mol,ref)
        if has_ref:
            dots = [abs(axis.dot(max_axis)) 
                    for max_axis in principal_axes]
            if not dots:
                continue
            mindot = numpy.amin(dots)
            maxdot = numpy.amax(dots)
            if maxdot>1.0-epsilon:
                #sigma h
                symmetries[-2]+=1
                logger.debug("Detected: sigma h")
            elif mindot<epsilon:
                #sigma vd
                symmetries[-3]+=1
                logger.debug("Detected: sigma vd")
    # rotoreflections
    for axis in axes:
        ref = reflection(axis)
        for order in range(2,max_order+1):
            rot = rotation(axis,order)
            rr  = rot.dot(ref)
            has_rr = is_valid_op(mol,rr)
            symmetries[order+max_order-2]+=int(has_rr)
            if has_rr:
                logger.debug("Detected: S{order}".format(order=order))
    # multiplicity
    symmetries[-1]=len([x for x in mol if x.symbol=="X"])
    return numpy.array(symmetries,dtype=int)
