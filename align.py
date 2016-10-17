"""
This module recels the marvels of 3D alignment. In particular, quaternions calculations, and minimizing of the error between objects.
There is no easily implemented anisotropic scaling method in procrustes analysis that would conserve systematically the shape.
The best scaling used here is thus isotropic, and then hail mary!
"""

from scipy.optimize import minimize, minimize_scalar
from itertools import *
from fragment import *
import numpy
import sys
import math


def procrustes(X, Y):

    """ 
    Returns the optimal rotation between the fragment and the slot, using quaternions
    """

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
    return R


def transfer_tags(fragment, slot):

    """ 
    Returns the list of tags transfered from a tagged slot to an untagged fragment
    """

    work = slot.copy()
    tags = numpy.zeros(len(fragment))
    if not slot.get_tags().any():
        return tags
    used_tags = []
    for dummy in fragment:
        # we only loop on untagged dummies
        if dummy.symbol == 'X' and dummy.tag == 0:
            distances = [(numpy.linalg.norm(dummy.position-s.position), s.tag) for s in work]
            i = 0
            found = False
            while not found:
                best = sorted(distances, key=lambda k:k[0])[i]
                if best[1] not in used_tags:
                    found = True
                    tags[dummy.index] = best[1]
                    used_tags.append(best[1])
                i += 1
    return tags


def rmse(fragment, slot):

    """ 
    Returns the Root Mean Square Error on the distance between each 
    dummy in the slot and it's closest fragment dummy
    """

    X = numpy.array([f.position for f in fragment if f.symbol == 'X'])
    Y = numpy.array([m.position for m in slot if m.symbol == 'X'])
    err_squared = 0
    l = 0
    for x in X:
        minimum = numpy.amin(numpy.linalg.norm(Y-numpy.tile(x, (Y.shape[0],1)), axis=1))
        index = numpy.where(numpy.linalg.norm(Y-numpy.tile(x, (Y.shape[0],1)), axis=1) == minimum)[0][0]
        Y = numpy.delete(Y, index, axis=0)
        err_squared += minimum**2
        l+=1
    rmse = math.sqrt(err_squared/l)
    return rmse


def scale_and_rotate(X, Y, LAMBDA):

    """ 
    Given two corresponding molecules and a scaling parameter,
    will return a scaled and rotated version of these molecules
    """

    XX = dummies_from_mol(X)
    YY = Y.copy()
    N = len(Y)
    #homogenous
    if len(LAMBDA.shape) == 1 and len(LAMBDA) == 1:
        YY.positions *= LAMBDA[0]
    #isotropic
    else:
        YY.set_positions([y.position * alpha for y, alpha in zip(Y, LAMBDA)])
    #passing positions and finding optimal correspondance
    Y.set_positions(YY.get_positions())
    #when X is already well placed, rotation gives bad results!
    if rmse(X, Y) < 0.1:
        return X, Y
    else:
        rotations = []
        tmp_XX = XX.positions.copy() 
        # this is a hacky way of getting the good results
        for i in range(N**2):
            XXX = XX.copy()
            numpy.random.shuffle(tmp_XX)
            R = procrustes(tmp_XX, YY.positions)
            XXX.positions = [numpy.dot(R, xxx.position) for xxx in XXX]
            rotations.append((R, rmse(XXX, YY)))
            # found a good candidate
            if rmse(XXX, YY) < 0.1:
                break
            # enough data
            if i > 50:
                break
        # get the best proposition
        R = sorted(rotations, key=lambda k:k[1])[0][0]

        X.set_positions([numpy.dot(R, x.position) for x in X])
        return X, Y


def fit_matrices(X, Y, LAMBDA):

    """ 
    Function to minimize for slot scaling
    """

    YY = Y.copy()
    XX = X.copy()
    XX, YY = scale_and_rotate(XX, YY, LAMBDA)
    RMSE = rmse(XX, YY)
    return RMSE


def fit_rotation(X, Y, AXIS, ANGLE):
    
    """ 
    Function to minimize for rotation
    """ 

    YY = Y.copy()
    XX = X.copy()
    XX.rotate(v=AXIS, a=ANGLE)
    RMSE = rmse(XX, YY)
    return RMSE


def dummies_from_mol(fragment):

    """ 
    Returns a pseudomolecule with only the dummies.
    """

    X = Fragment()
    for x in fragment:
        if x.symbol=='X':
            X.append(x)
            X.mmtypes.append(fragment.mmtypes[x.index])
        else:
            pass
    return X


def star_align(args):
    
    """ 
    Wrapper function for multiprocess alignment
    """

    return align(*args)


def align(fragment, slot):

    """ 
    Main function of this module. 
    Aligns a fragment on top of a scaled version of the slot
    """

    #create working molecules with only dummies present
    X = dummies_from_mol(fragment)
    Y = slot.copy()
    if not (X.positions.shape == Y.positions.shape):
        raise RuntimeError("incompatible connectivity between slot and building unit")
    # total points
    N = X.positions.shape[0]
    # center the points on zero
    centroid_X = numpy.mean(X.positions, axis=0)
    centroid_Y = numpy.mean(Y.positions, axis=0)
    X.positions -=  numpy.tile(centroid_X, (N, 1))
    Y.positions -=  numpy.tile(centroid_Y, (N, 1))
    #translation to origin for fragments and slots
    fragment.positions -= numpy.tile(centroid_X, (len(fragment),1))
    slot.positions -= numpy.tile(centroid_Y, (len(slot),1))
    #get maximum factor in scaling
    bound = 1.5 * numpy.linalg.norm(numpy.amax(X.get_positions(), axis=0)) / numpy.linalg.norm(numpy.amax(Y.get_positions(), axis=0)) 
    # get the best fit
    minfunction = lambda scale : fit_matrices(X, Y, scale)
    bounds = [(0,bound)]
    parameters = numpy.array([0.75*bound])
    rez = minimize(fun=minfunction, x0=parameters, bounds=bounds)
    fragment, slot = scale_and_rotate(fragment, slot,rez.x)
    # tagging
    tags = transfer_tags(fragment, slot)
    fragment.set_tags(tags)
    # progress bar
    sys.stdout.write('.')
    sys.stdout.flush()
    return fragment, slot


def embed_mil53(framework, model):

    """
    MIL53 is one of these weird topologies that need manual alignment.
    This function (and any other further added because of other weirdness), returns
    a nice framework, well aligned
    """


    transformations = [[[1,0,0], [0,1,0],[0,0,1]],
                       [[1,0,0],[0,-1,0],[0,0,-1]],
                       [[1,0,0], [0,1,0],[0,0,1]],
                       [[1,0,0],[0,-1,0],[0,0,-1]]]

    # we will manually align the first object, then use transformations
    mil_monomer_idxs  = [f.idx for f in framework.fragments if f.shape=='mil53']
    first_mil_monomer = framework[mil_monomer_idxs[0]].copy()
    first_mil_slot    = model[mil_monomer_idxs[0]].copy()
    first_mil_monomer.set_cell(framework.cell)
    first_mil_slot.set_cell(model.cell)

    fragX = Fragment([fx for fx in first_mil_monomer if (first_mil_monomer.mmtypes[fx.index]!="C_R" and fx.symbol=="X")])
    slotX = Fragment([sx for sx in first_mil_slot if first_mil_slot.mmtypes[sx.index]!="C_R"])
    
    centroid_fragX = numpy.mean(fragX.positions, axis=0)
    centroid_slotX = numpy.mean(slotX.positions, axis=0)
    fragX.positions -= centroid_fragX
    slotX.positions -= centroid_slotX
    R = procrustes(fragX.positions, slotX.positions)
    first_mil_monomer.positions -= centroid_fragX
    first_mil_slot.positions    -= centroid_slotX 
    first_mil_monomer.positions  = numpy.dot(first_mil_monomer.positions, R)
    axis = fragX.positions[0]-fragX.positions[1]
    minfunction = lambda angle : fit_rotation(first_mil_monomer, first_mil_slot, AXIS=axis, ANGLE=angle)
    bounds = [(0.0, 2*math.pi)]
    rez = minimize_scalar(fun=minfunction, bounds=bounds)
    first_mil_monomer.rotate(v=axis, a=rez.x)

    # now transform this one well aligned object
    for transformation, index in zip(transformations, mil_monomer_idxs):
        monomer = first_mil_monomer.copy()
        slot = model[index].copy()
        slot_centroid = numpy.mean([sx.position for sx in slot if slot.mmtypes[sx.index]!="C_R"], axis=0)
        monomer.positions  = numpy.dot(monomer.positions, numpy.array(transformation))
        monomer.positions += slot_centroid
        monomer.set_tags(numpy.zeros(len(monomer)))
        tags = transfer_tags(monomer, slot)
        monomer.set_tags(tags)
        monomer.positions -= slot_centroid
        framework[index] = monomer
        # progress bar
        sys.stdout.write('.')
        sys.stdout.flush()
    print " Embedding completed"
    return framework