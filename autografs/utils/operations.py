#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.4'
__status__  = "beta"

import numpy

def inertia(xyz : numpy.ndarray,
            W   : numpy.ndarray) -> numpy.ndarray:
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
    
def rotation(axis  : numpy.ndarray,
             order : float) -> numpy.ndarray:
    """Return a rotation matrix around the axis"""
    M = numpy.eye(3)
    axis /= numpy.linalg.norm(axis)
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
    
    
def reflection(axis : numpy.ndarray) -> numpy.ndarray:
    """Return a reflection matrix around the axis"""
    M = numpy.eye(3)
    axis /= numpy.linalg.norm(axis)
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


def procrustes(X : numpy.ndarray,
               Y : numpy.ndarray,
               method : str = "Q") -> (numpy.ndarray,float):
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