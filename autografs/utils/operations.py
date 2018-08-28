#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.3.0'
__status__  = "production"


import numpy

def inertia(xyz,
            W  ):
    """Return the inertia matrix of coordinates, weighted"""
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    x2 = x*x
    y2 = y*y
    z2 = z*z
    Ixx =  numpy.sum((y2+z2)*W)
    Iyy =  numpy.sum((x2+z2)*W)
    Izz =  numpy.sum((x2+y2)*W)
    Ixy = -numpy.sum((x *y )*W)
    Iyz = -numpy.sum((y *z )*W)
    Ixz = -numpy.sum((x *z )*W)
    I = numpy.array([[Ixx,Ixy,Ixz],
                     [Ixy,Iyy,Iyz],
                     [Ixz,Iyz,Izz]],dtype=numpy.float32)
    return I
    
def rotation(axis ,
             order):
    """Return a rotation matrix around the axis"""
    M = numpy.eye(3)
    norm = numpy.linalg.norm(axis)
    if norm<1e-3:
        norm = 1.0
    axis /= norm
    v0      = axis[0]
    v1      = axis[1]
    v2      = axis[2]
    theta   = 2.0*numpy.pi/order
    costh   = numpy.cos(theta)
    sinth   = numpy.sin(theta)
    M[0,0] = costh + (1.0-costh)*v0**2
    M[1,1] = costh + (1.0-costh)*v1**2
    M[2,2] = costh + (1.0-costh)*v2**2
    M[1,0] = (1.0-costh)*v0*v1 + v2*sinth
    M[0,1] = (1.0-costh)*v0*v1 - v2*sinth
    M[2,0] = (1.0-costh)*v0*v2 - v1*sinth
    M[0,2] = (1.0-costh)*v0*v2 + v1*sinth
    M[2,1] = (1.0-costh)*v1*v2 + v0*sinth
    M[1,2] = (1.0-costh)*v1*v2 - v0*sinth   
    return M
    
    
def reflection(axis):
    """Return a reflection matrix around the axis"""
    M = numpy.eye(3)    
    norm = numpy.linalg.norm(axis)
    if norm<1e-3:
        norm = 1.0
    axis /= norm
    v0      = axis[0]
    v1      = axis[1]
    v2      = axis[2]
    M[0,0] = 1.0-2.0*v0*v0
    M[1,1] = 1.0-2.0*v1*v1
    M[2,2] = 1.0-2.0*v2*v2
    M[1,0] =    -2.0*v0*v1 
    M[0,1] =    -2.0*v0*v1 
    M[2,0] =    -2.0*v0*v2 
    M[0,2] =    -2.0*v0*v2 
    M[2,1] =    -2.0*v1*v2 
    M[1,2] =    -2.0*v1*v2
    return M


def procrustes(X,
               Y,
               method = "Q"):
    """Return the optimal rotation between two sets of coordinates"""
    X = numpy.array(X,dtype=float)
    Y = numpy.array(Y,dtype=float)
    if method == "Q":
        #covariance matrix
        H = numpy.dot(X.T, Y)
        # four dimensional matrix for quaternion calculation
        P = numpy.array([[H[0,0]+H[1,1]+H[2,2], H[1,2]-H[2,1], H[2,0]-H[0,2], H[0,1]-H[1,0]],
                         [H[1,2]-H[2,1], H[0,0]-H[1,1]-H[2,2], H[0,1]+H[1,0], H[2,0]+H[0,2]],
                         [H[2,0]-H[0,2], H[0,1]+H[1,0], H[1,1]-H[0,0]-H[2,2], H[1,2]+H[2,1]],
                         [H[0,1]-H[1,0], H[2,0]+H[0,2], H[1,2]+H[2,1], H[2,2]-H[1,1]-H[0,0]]])
        # compute eigenvalues
        eigenvalues, eigenvectors = numpy.linalg.eigh(P)
        # quaternion
        q = eigenvectors[:, numpy.argmax(eigenvalues)]
        q /= numpy.linalg.norm(q)
        #orthogonal rotation matrix
        R = numpy.array([[(q[0]**2)+(q[1]**2)-(q[2]**2)-(q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                         [2*(q[1]*q[2]+q[0]*q[3]), (q[0]**2)-(q[1]**2)+(q[2]**2)-(q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
                         [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), (q[0]**2)-(q[1]**2)-(q[2]**2)+(q[3]**2)]])
        scale = 1.0
    elif method == "SVD":
        u, w, vt = numpy.linalg.svd(X.T.dot(Y).T)
        R = u.dot(vt)
        scale = w.sum()
    else:
        raise NotImplementedError("Unknown method. Implemented are SVD or Q")
    return R, scale

def is_valid_op(mol,  
                symmop,
                epsilon = 0.1):
    """Check if a particular symmetry operation is a valid symmetry operation
    for a molecule, i.e., the operation maps all atoms to another
    equivalent atom.
        -- mol : ASE Atoms object. subject of symmop
        -- symmop: Symmetry operation to test.
        -- epsilon : numerical tolerance of the
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