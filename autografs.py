# -*- coding: utf-8 -*-
#!/usr/bin/python2.7

"""
This module is the main Autografs file. It contains both the Model and Autografs classes,
which are respectively the container and the constructor for framework crystals.
"""

from itertools import combinations_with_replacement, product
from ase.lattice.spacegroup import crystal
from ase.calculators.neighborlist import *
from scipy.optimize import *
import numpy
import platform
import pickle
import random
import sys
import os

from fragment import *
from topology import *
from align    import *
from utils    import *

from ase.visualize import view


class Model:

    """ This is a container class for Autografs fragments. 
        It is designed to keep track of both the structure, topology, and individual objects.
        the constructor awaits the following arguments (all arguments are optional):
            - label : string that will be used for identification and file naming.
            - cell  : 3x3 array. unit cell parameters.
            - pbc   : 3x1 boolean array. True for each periodic direction (a, b, c).
        usage:
            >>> myModel = Model()
    """

    def __init__(self, label="framework", cell=[[1,0,0],[0,1,0],[0,0,1]], pbc=[1,1,1]):

        self.label = label
        self.cell = numpy.around(cell, decimals=5)
        self.pbc=[1,1,1]
        self.fragments = []
        self.topology = None
        self.shapes = shapes = {'linear'              : 2,
                                'triangle'            : 3,
                                'square'              : 4,
                                'rectangle'           : 4,
                                'tetrahedral'         : 4,
                                'mil53'               : 4,
                                'trigonal_bipyramid'  : 5,
                                'octahedral'          : 6,
                                'tri_prism'           : 6,
                                'mfu4'                : 6,
                                'hexagonal'           : 6,
                                'cubic'               : 8,
                                'icosahedral'         : 12}

                                
    def __getitem__(self, index):
        
        """
        Getter function that returns the Fragment object at index.
        """

        return self.fragments[index]


    def __setitem__(self, key, val):

        """
        Setter function for the fragment list
        """

        assert isinstance(val, Fragment)
        self.fragments[key] = val


    def __delitem__(self, index):

        """
        Deletes all specified objects in the Fragment list.
        """

        if not isinstance(index, list):
            index = list(index)
        for i in index:
            del self.fragments[i]


    def __len__(self):

        """
        Intrinsic method for length of Fragment list.
        """

        return len(self.fragments)


    def __imul__(self, m):

        """
        Method defining the multiplication of components. 
        In practice, this will create an appropriate supercell out of the fragments.
        Inputs:
            -- m : integer, or 3x1 array for diirections in the supercell
        """

        #transforms simple number in three components
        if isinstance(m, int):
            m = (m, m, m)
        N =  numpy.prod(m)
        old_cell = self.cell.copy()
        self.set_cell(numpy.dot(m, self.cell))
        if self.topology is not None:
            self.topology *= m
        olen     = len(self)
        startlen = 0
        endlen   = len(self)
        if len(self) > 0:
            for m0 in range(m[0]):
                for m1 in range(m[1]):
                    for m2 in range(m[2]):
                        displacement = numpy.dot((m0, m1, m2), old_cell)
                        if sum(displacement)!=0:
                            for idx, oidx in zip(numpy.arange(startlen, endlen), numpy.arange(0,olen)):
                                newfrag = self.fragments[oidx].copy()
                                newfrag.positions += displacement
                                self.append(newfrag)
                        startlen += olen
                        endlen   += olen
        print "Supercell generated : ", m
        return self          


    def __mul__(self, m):

        """
        Wraps around imul method above
        """

        new = self.copy()
        new *= m
        return new


    def __rmul__(self, m):

        """
        Does the same as __mul__ above, but specifies commutative multiplication.
        """

        return __mul__(self, m)


    def __str__(self): 

        """
        Nicecly formatted information about the model
        """

        fragstring = '\n'.join(['\n'.join(['{0} {1}'.format(fragment.unit, fragment.idx),
                                            '{0} connections'.format(len(fragment)),
                                            '{0}'.format(fragment.shape),
                                            '\n'.join(['{0}\t{1}\t{2}\t{3}'.format(numpy.around(atom.position[0], decimals=3),
                                                                                   numpy.around(atom.position[1], decimals=3),
                                                                                   numpy.around(atom.position[2], decimals=3),
                                                                                   atom.tag) for atom in fragment])
                                          ]) for fragment in self.fragments if fragment is not None])
        string = '\n'.join(['Number of objects {0}'.format(len(self)),
                            'Cell : ',
                            '{0} {1} {2}'.format(numpy.around(self.cell[0,0], decimals=3),
                                                 numpy.around(self.cell[0,1], decimals=3),
                                                 numpy.around(self.cell[0,2], decimals=3)),
                            '{0} {1} {2}'.format(numpy.around(self.cell[1,0], decimals=3),
                                                 numpy.around(self.cell[1,1], decimals=3),
                                                 numpy.around(self.cell[1,2], decimals=3)),
                            '{0} {1} {2}'.format(numpy.around(self.cell[2,0], decimals=3),
                                                 numpy.around(self.cell[2,1], decimals=3),
                                                 numpy.around(self.cell[2,2], decimals=3)),
                            '\n',
                            '{0}'.format(fragstring)])
        return string


    def set_topology(self, topology, fill=True):

        """ 
        Takes care of the topology setup. this is exclusively called in the topology module.
        If fill is true, the length of the Model is set to the length of the topology.
        usage:
            >>> myTopology = Atoms(symbols=..., positions=...)
            >>> Model.set_topology(topology=myTopology)
        """

        self.topology = topology.copy()
        if fill:
            self.fragments = [None,]*len(topology)
        self.cell = topology.get_cell()
        self.pbc  = topology.pbc 
        return

    def view(self, clean=True):
        
        """ 
        This method opens an ASE-GUI window displaying the results of the Model.get_atoms function.
        Using this method in a script will pause the execution until the ASE-GUI window is closed.
        If clean is False, the dummy atoms will be kept and connectivity will not be correct.
        """
        
        atoms=self.get_atoms(clean=clean)
        view(atoms)
        return

            
    def write(self, name=None, clean=True, verbose=True):

        """ 
        This part writes a molecule file reflecting the properties of the Model.
        By default, a GULP UFF optimization will be written with reasonable parameters.
        If clean is False, the dummy atoms will be kept and connectivity will not be correct.
        the name keyword will be used as a filename. if it is not supplied, the Model.label property will be used instead.
        the verbose keyword controls the quantity of printing feedback.
        usage:
            >>> print Model.label
            "framework"
            >>> Model.write(name="myMOF", clean=True)
            >>> os.listdir("current/directory/") # list the contents of the directory
            ["framework.gin",...]
            >>>
        """

        if name is None:
            name=self.label
        fragment = self.get_atoms(clean=clean)
        fragment.write(name)
        if verbose:
            print "--> Framework {0} written to file.".format(name)
        return


    def copy(self):

        """ 
        Returns a safe copy of the full Model object.
        this copy can be modified without inpact on the original object
        usage:
            >>> newFramework = Model.copy()
        """

        copied = self.__class__()
        copied.cell = self.cell.copy()
        copied.pbc  = self.pbc.copy()
        copied.fragments = [frag.copy() for frag in self.fragments]
        if self.topology is not None:
            copied.topology = self.topology.copy()
        copied.label = self.label
        copied.shapes = self.shapes
        return copied


    def get_atoms(self, clean=False, indices=False):

        """ 
        This is the method that returns the fully connected and typed Fragment object from the Model's building units.
        if the clean statement is False, the dummy atoms are included, and no connectivity between units is shown. else, the 
        dummy atoms are replaced by the correct connectivity.
        If indices=True, an ordered list is also returned that gives information about which atom originally belongs to which fragment.
        The indices keyword should not be used outside of internal routines.
        This allows the user to manipulate the generated framework with the ASE Atoms object methods (calculators, etc.)
        usage:
            >>> myFramework = Model.get_atoms(clean=True)
        """

        atoms = Fragment(cell=self.cell, pbc=self.topology.pbc)
        idxs = []
        # add all fragments to the same Fragment
        for fragment in self:
            atoms.extend(fragment)
            for atom in fragment:
                idxs.append((fragment.idx, atom.index))
        if clean:
            # the dummy atopms specifying connectivity have to go
            # they are tagged by pairs, and replaced by a single bond before being deleted
            tags = list(set([t for t in atoms.get_tags() if t != 0]))
            pairs = []
            for tag in tags:
                sameTag_tmp = list(numpy.array(numpy.where(atoms.get_tags()==tag)).ravel())
                bigdist = []
                # some sorting by distance for each tag pair
                for t in sameTag_tmp:
                    distances = sorted([(a, atoms.get_distance(t, a, mic=True)) for a in sameTag_tmp], key=lambda k: k[1])
                    bigdist.append(distances[1][1])
                sameTag_tmp = list(zip(*sorted(zip(sameTag_tmp, bigdist), key=lambda k : k[1]))[0])
                while len(sameTag_tmp) > 0:
                    t = sameTag_tmp.pop(0)
                    distances = sorted([(a, atoms.get_distance(t, a, mic=True)) for a in sameTag_tmp], key=lambda k: k[1])
                    best = distances[0][0]
                    st = [t, best]
                    pairs.append(st)
                    sameTag_tmp.remove(best)
                    # dangling bit!
                    if len(sameTag_tmp)==1:
                        pairs.append(sameTag_tmp)
                        sameTag_tmp = []
            todel = []
            for sameTag in pairs:
                if len(sameTag) == 1:
                    # we need to cap dangling dummies
                    atoms[sameTag[0]].symbol = "H"
                    atoms.mmtypes[sameTag[0]]= "H_"
                else:
                    todel.extend(sameTag)
                    #tags are the same. verified.
                    these_bonds = atoms.bonds[sameTag[0]]
                    these_other_bonds = atoms.bonds[sameTag[1]]
                    indices0 = numpy.array(numpy.where(these_bonds!=0)).ravel()
                    indices1 = numpy.array(numpy.where(these_other_bonds!=0)).ravel()
                    #if an atom is connected to a dummy, it is connected to all the corresponoding atoms
                    if len(indices0)>len(indices1):
                        bondOrders = these_bonds[indices0]
                        maxindices = indices0
                        minindices = indices1
                    else:
                        bondOrders = these_other_bonds[indices1]
                        maxindices = indices1
                        minindices = indices0
                    maxindexcounter = 0
                    for maxindex in maxindices:
                        for minindex in minindices:
                            atoms.set_bond(maxindex, minindex, bondOrders[maxindexcounter])
                        maxindexcounter += 1
            del atoms[todel]
        if indices:
            idxs = numpy.array(idxs)
            return atoms, idxs
        else:
            return atoms
        

    def get_tags(self):
   
        """ 
        Returns a copy of the tags in the Model atoms. 
        The dummy Atoms are included. This should not be used outside of internal routines.
        """

        atoms = self.get_atoms()
        return atoms.get_tags()
   

    def get_cell(self):

        """ 
        Returns a 3x3 array containing a copy of the Model.cell attribute.
        usage:
            >>> cell = Model.get_cell()
            >>> print cell
            [[3.0  0.0  0.0]
             [0.0 10.5  0.0]
             [0.0  0.0 10.5]]
            >>>         
        """

        return self.cell.copy()


    def set_cell(self, cell):

        """ 
        Safely set the cell attribute. 
        If an array of 3 numbers is given, a diagonal matrix is implied.
        usage:
            >>> Model.set_cell(cell=[3,10.5,10.5])
            >>> print Model.cell
            [[3.0  0.0  0.0]
             [0.0 10.5  0.0]
             [0.0  0.0 10.5]]
            >>>
        """

        cell = numpy.array(cell)
        if cell.shape != (3,3):
            cell = make_diagonal(cell)
        self.cell = cell


    def get_linkers(self):

        """ 
        Returns a list of all the linker units as Fragment objects.
        The unit type is determined by the topology of the model.
        usage:
            >>> linkers = Model.get_linkers()
            >>> print linkers
            [0,4,5,6]
            >>>
        """

        return [f.idx for f in self if f.unit=="linker"]


    def get_centers(self):

        """ 
        Returns a list of all the center units as Fragment objects.
        The unit type is determined by the topology of the model.
        usage : 
            >>> centers = Model.get_centers()
            >>> print centers
            [1,2,3]
            >>>
        """

        return [f.idx for f in self if f.unit=="center"]


    def append(self, fragment):

        """ 
        This function appends a new Fragment object to the list of fragments.
        usage:
            >>> myNewFragment = Fragment(symbols=..., positions=...)
            >>> Model.append(myNewFragment)
        """

        assert isinstance(fragment, Fragment)
        fragment.idx = len(self)
        self.fragments.append(fragment)


    def pop(self, i=-1):
        
        """ 
        Remove and return the fragment at index i (default last).
        usage:
            >>> fragment1 = Model.pop(i=1)
        """
        
        fragment = self[i]
        del self[i]

        return fragment


    def get_form_factor(self):

        """ 
        This function returns a number used to have an idea of the size of the Model.
        This is used for automatic resizing of the cell in the Autografs.make method, and in the 
        Autografs.optimize_cell(). The more fragments in the model, and the bigger the fragments, the higher the form factor
        usage:
            >>> formFactor = Model.get_form_factor()
        """

        ff = 0.0
        for fragment in self:
            X = Fragment([x for x in fragment if x.symbol=="X"])
            centroid = numpy.mean(X.positions, axis=0)
            X.positions -= centroid
            ff += sum(numpy.linalg.norm(X.positions, axis=1))
        ff /= len(self)
        ff *= 6.0 - sum(self.pbc)
        return ff


    def insert_defect(self, indices=None):

        """ 
        Deletes a list of fragments in the MOF.
        the defects will cap with hydrogen the empty spaces.
        usage:
            >>> Model.insert_defect(indices=[1,25,30])
        """

        defects = []
        # get the indices of defects
        if isinstance(indices, list):
            defects = indices
        else: 
            defects = list(indices)
        for defect in defects:
            print "Deleted fragment number {0} for defect generation".format(defect)
        del self[defects]
        return self


    def flip(self, index=0):

        """ 
        will rotate the specified linear sbu 90 degrees around itself.
        usage:
            >>> Model.flip(index=0)
        """

        X = numpy.array([x.position for x in self[index] if x.symbol=='X'])
        # only works for linear objects. for the rest, do it manually
        if  X.shape[0] != 2:
            print "Cannot flip a non-linear object : ambiguous axis of rotation."
            print "please, use the GUI for complex operations."
            return
        else:
            center = numpy.mean(X, axis=0)
            # get the farthest atom from centroid of dummies
            farthest_atom = sorted([(numpy.linalg.norm(a.position-center), a.position) for a in self[index]], key = lambda k:k[0])[-1][1]
            # get cross vector
            v = numpy.cross(X[0]-X[1], center-farthest_atom)
            a = math.radians(180.0)
            self[index].rotate(v=v, a=a, center=center)
            print "Fragment {0} flipped.".format(index)
            return


    def rotate(self, index=0, angle=0.0):

        """ 
        will rotate the specified linear sbu around the axis made by
        dummy atoms by the specified angles in degrees.
        usage:
            >>> Model.rotate(index=0, angles=90.0)
        """

        X = numpy.array([x.position for x in self[index] if x.symbol=='X'])
        # only works for linear objects. for the rest, do it manually
        if  X.shape[0] != 2:
            print "Cannot rotate a non-linear object : ambiguous axis of rotation."
            print "please, use the GUI for complex operations."
            return
        else:
            center = numpy.mean(X, axis=0)
            v = X[0] - X[1]
            a = math.radians(angle)
            self[index].rotate(v=v, a=a, center=center)
            print "Fragment {0} rotated by {1} degrees.".format(index, angle)
            return


    def functionalize(self, functional_group, index=0, func_index=None):

        """ 
        Will add a functional group to the selected sbu on a random hydrogen
        for more precise handling, use the GUI for now.
        will replace the atom nr func_index in fragment nr index by the functional group
        will produce unexpected results if the atom replaced is not a cap (e.g: hydrogen).
        If func_index is None, a random hydrogen in the fragment will be used,
        If and only if the hydogen is connected to a carbon.
        usage:
            >>> Model.functionalize(functional_group="F", index=0, func_index=None)
        """

        if not isinstance(functional_group, Fragment):
            print "the functional groups need to be Fragment objects. These can be accessed in the autografs database"
            return
        
        if f_index is None:
            #get all functionalizable zones
            H = [atom for atom in self[index] if atom.symbol=="H"]
            functionalizable = []
            for h in H:
                attached_to = numpy.where(self[index].bonds[h.index] != 0)[0][0]
                if self[index][attached_to].symbol == "C":
                    functionalizable.append(h.index)
            if len(functionalizable) == 0:
                print "Sorry, no reasonable site for functionalization on chosen fragment."
                return
            # get a random hydrogen
            func_index = random.choice(functionalizable)

        original_length = len(self[index])
        original_position = self[index][func_index].position
        
        #get info about the attach point
        attach_point = numpy.where(self[index].bonds[func_index] != 0)[0][0]
        mat_self = numpy.vstack((self[index][func_index].position, self[index][attach_point].position))
        centroid_self = numpy.mean(mat_self, axis=0)
        self[index].positions -= centroid_self
        vec_original =  self[index][attach_point].position - self[index][func_index].position
        
        #get teh point group data
        dummy_index = [d.index for d in functional_group if d.symbol == 'X'][0]
        dummy_attach_point = numpy.where(functional_group.bonds[dummy_index] != 0)[0][0]
        mat_fg = numpy.vstack((functional_group[dummy_index].position, functional_group[dummy_attach_point].position))
        centroid_fg = numpy.mean(mat_fg, axis=0)
        functional_group.positions -= centroid_fg
        vec_dummy =  functional_group[dummy_index].position - functional_group[dummy_attach_point].position
        
        #align the connections in a realistic manner
        H = numpy.dot(numpy.array([vec_dummy]).T, numpy.array([vec_original]))
        U, S, Vt = numpy.linalg.svd(H)
        R = numpy.dot(Vt.T, U.T)
        functional_group.set_positions([numpy.dot(R, fg.position) + self[index][func_index].position for fg in functional_group])

        #book keeping on bonds
        self[index].extend(functional_group)
        self[index].set_bond(attach_point, dummy_attach_point+original_length, 1.0)
        del self[index][func_index]
        del self[index][dummy_index + original_length - 1]
        self[index].positions += centroid_self
        print "Fragment {0} functionalized with {1} group.".format(index, functional_group.name)
        return


    def make(self, topology):
    
        """ 
        Dispatch to the correct function from the topology module for making the model. 
        Returns a filled and tagged model.
        This function has to be modified to implement a new topology.
        """    

        if topology == 'pcu':
            self = make_pcu(self)
            return
        elif topology == 'sql':
            self = make_sql(self)
            return
        elif topology == 'hex_p':
            self = make_hexp(self)
            return
        elif topology == 'bcu':
            self = make_bcu(self)
            return
        elif topology == 'cds':
            self = make_cds(self)
            return
        elif topology == 'pyr':
            self = make_pyr(self)
            return
        elif topology == 'srs':
            self = make_srs(self)
            return
        elif topology == 'nbo':
            self = make_nbo(self)
            return
        elif topology == 'dia':
            self = make_dia(self)
            return
        elif topology == 'sra':
            self = make_sra(self)  
            return  
        elif topology == 'bor':
            self = make_bor(self)       
            return  
        elif topology == 'ctn':
            self = make_ctn(self)       
            return  
        elif topology == 'pto':
            self = make_pto(self)
            return  
        elif topology == 'tbo':
            self = make_tbo(self)
            return  
        elif topology == 'bnn':
            self = make_bnn(self)
            return  
        elif topology == 'pts':
            self = make_pts(self)
            return  
        elif topology == 'ptt':
            self = make_ptt(self)
            return  
        elif topology == 'pth':
            self = make_pth(self)
            return  
        elif topology == 'stp':
            self = make_stp(self)
            return  
        elif topology == 'gar':
            self = make_gar(self)
            return     
        elif topology == 'bto':
            self = make_bto(self)
            return     
        elif topology == 'soc':
            self = make_soc(self)
            return  
        elif topology == 'spn':
            self = make_spn(self)
            return  
        elif topology == 'ibd':
            self = make_ibd(self)
            return  
        elif topology == 'iac':
            self = make_iac(self)
            return  
        elif topology == 'ifi':
            self = make_ifi(self)
            return  
        elif topology == 'rtl':
            self = make_rtl(self)
            return  
        elif topology == 'sod':
            self = make_sod(self)
            return  
        elif topology == 'sqc19':
            self = make_sqc19(self)
            return
        elif topology == 'qom':
            self = make_qom(self)
            return 
        elif topology == 'rhr':
            self = make_rhr(self)
            return 
        elif topology == 'ntt':
            self = make_ntt(self)
            return 
        elif topology == 'mtn_e':
            self = make_mtne(self)
            return  
        elif topology == 'afw':
            self = make_afw(self)
            return  
        elif topology == 'hcp':
            self = make_hcp(self)
            return  
        elif topology == 'afx':
            self = make_afx(self)
            return 
        elif topology == 'reo':
            self = make_reo(self)
            return 
        elif topology == 'tsi':
            self = make_tsi(self)
            return 
        elif topology == 'flu':
            self = make_flu(self)
            return 
        elif topology == 'scu':
            self = make_scu(self)
            return 
        elif topology == 'bcs':
            self = make_bcs(self)
            return 
        elif topology == 'ics':
            self = make_ics(self)
            return 
        elif topology == 'icv':
            self = make_icv(self)
            return 
        elif topology == 'qtz':
            self = make_qtz(self)
            return 
        elif topology == 'ftw':
            self = make_ftw(self)
            return 
        elif topology == 'ocu':
            self = make_ocu(self)
            return 
        elif topology == 'she':
            self = make_she(self)
            return 
        elif topology == 'shp':
            self = make_shp(self)
            return 
        elif topology == 'the':
            self = make_the(self)
            return
        elif topology == 'toc':
            self = make_toc(self)
            return 
        elif topology == 'ttt':
            self = make_ttt(self)
            return  
        elif topology == 'icx':
            self = make_icx(self)
            return  
        elif topology == 'ivt':
            self = make_ivt(self)
            return    
        elif topology == 'mil53':
            print "MIL-53 is an embedded system. Special alignment procedures are used."
            self = make_mil53(self)
            return  
        elif topology == 'mfu4':
            print "MFU-4 is an embedded system. Special alignment procedures are used."
            self = make_mfu4(self)
            return  
        else:
            raise RuntimeError('Topology: '+topology+' not implemented!')
    

    def embed(self, topology, model):

        """ 
        dispatch to the correct function from the topology module for making the model. returns a filled and tagged model.
        This is used when weird topologies are specified.
        """    

        if topology=='mil53':
            self = embed_mil53(self, model)
            return
        if topology=='mfu4':
            return
        else:
            raise RuntimeError('Topology: '+topology+' not implemented as embedded system!')


    def tag(self, radius=1e-3):

        """ 
        This method tags corresponding dummy atoms ("X", or "Xx") for later transfer to aligned building parts.
        This is exclusively used in the topology module on the result of the Model.fill method.
        The closest dummies in different fragments form a pair and get assigned an identical tag. 
        The tag is negative if it crosses a periodic boundary. The radius should be varied to get the correct
        behaviour when implementing a new topology.
        """

        tag = 1
        #get the periodic molecule model
        fullModel, idxs = self.get_atoms(indices=True)
        #radius for bond detection: 
        skin = 0.1 * radius
        cutoffs = [radius] * len(fullModel)
        neigbourlist = NeighborList(cutoffs, skin=skin, self_interaction=False, bothways=True)
        neigbourlist.build(fullModel)
        #tag bonded dummies with the same tag, negative if goes over pbc
        for dummy in fullModel:
            #we loop over untagged dummies only
            if dummy.tag == 0 :
                indices, offsets = neigbourlist.get_neighbors(dummy.index)
                index_index = 0
                if len(indices) > 1:
                    print 'WARNING: Element tagged multiple times.'
                    print '\t Attempting tag with closest untagged dummy'
                    dummy_distances =  fullModel.get_distances(dummy.index, indices, mic=True)
                    index_index = dummy_distances.argmin()
                if len(indices) == 0 :
                    raise AssertionError('Element left untagged!')
                index = indices[index_index]
                offset = offsets[index_index]
                # book keeping on current full model molecule
                dummy.tag = tag
                fullModel[index].tag = tag
                # tag the fragments individually
                if abs(offset).any():
                    self[idxs[dummy.index][0]][idxs[dummy.index][1]].tag = -tag
                    self[idxs[index][0]][idxs[index][1]].tag = -tag
                else:
                    self[idxs[dummy.index][0]][idxs[dummy.index][1]].tag = tag
                    self[idxs[index][0]][idxs[index][1]].tag = tag
                tag += 1
            else:
                pass

    
    def fill(self, topology, C=None, K=None, N=None, O=None, P=None, radius=1e-3, scale=True):

        """ 
        This method takes a topology Atoms object, and deduces the positions of dummy atoms based on 
        its connectivity. Fragment objects are created with the correct shapes and amount of dummies, and are placed in the Model container.
        the radius is used for Neighborlist calculations, and can be varied when implementing new topologies.
        C and K keywords await the shape of the different centers in the topology, and N, O, P the shape of the different linkers.
        the scale keyword tries to gve a good approximation of connectivity by placing dummy atoms in the middle of the bonds.
        This is exclusively used in the topology module after set_topology is called.
        """
    
        print "\tAnalysis of the topology..."
        #build neigbourlist
        cutoffs = [radius] * len(topology)
        neigbourlist = NeighborList(cutoffs, skin=0.2*radius, self_interaction=False, bothways=True)
        neigbourlist.build(topology)
        #assign the correct value to symbols
        for symbol in ['C', 'K', 'N', 'O', 'P']:
            for point in topology:
                if point.symbol == symbol:
                    if symbol == 'C':
                        shape, multiplicity = C, self.shapes[C]
                        unit = 'center'
                    elif symbol == 'K':
                        shape, multiplicity = K, self.shapes[K]
                        unit = 'center'
                    elif symbol == 'N':
                        shape, multiplicity = N, self.shapes[N]
                        unit = 'linker'
                    elif symbol == 'O':
                        shape, multiplicity = O, self.shapes[O]
                        unit = 'linker'
                    elif symbol == 'P':
                        shape, multiplicity = P, self.shapes[P]
                        unit = 'linker'
                    #Aligning 
                    indices, offsets = neigbourlist.get_neighbors(point.index)
                    if len(indices) != multiplicity:
                        if unit == "center":
                            connects_to = ["N", "O", "P"]
                        else:
                            connects_to = ["C", "K"]
                        #mostly, linkers only connect with centers, etc...
                        indices_tmp, offsets_tmp = zip(*[io for io in zip(indices, offsets) if topology[io[0]].symbol in connects_to])
                        if len(indices_tmp) != multiplicity:
                            #sometimes, linkers only connect with linkers, etc...
                            indices, offsets = zip(*[io for io in zip(indices, offsets) if topology[io[0]].symbol not in connects_to])
                        else:
                            indices = indices_tmp
                            offsets = offsets_tmp
                    print "\t\t", symbol, "number", point.index
                    print "\t\t\t|"
                    for i in indices:
                        print "\t\t\t`-->", "{0} number {1}".format(topology[i].symbol, topology[i].index)
                    if len(indices) != multiplicity:
                        raise AssertionError('Connectivity is different in topology and fragment!')
                    else:
                        fragment = Fragment('X'*multiplicity, shape=shape, positions=numpy.zeros(3*multiplicity).reshape((multiplicity, 3)))
                        Y = []
                        # new coordinates
                        for index, offset in zip(indices, offsets):
                            if scale:
                                Y.append(topology.positions[index] + numpy.dot(offset, topology.get_cell()))
                            else:
                                middle = (point.position + (topology.positions[index] + numpy.dot(offset, topology.get_cell())))/2
                                Y.append(middle)
                        #here we use the mmtypes as a data holder for embedded systems
                        fragment.mmtypes = [topology[i].symbol for i in indices]
                        Y = numpy.array(Y)
                        #scale for the bonds
                        if scale:
                            Y = scale_object(coordinates=Y, alpha=0.5)
                        # Update the fragment's coordinates 
                        fragment.set_positions(Y)
                        # update the model
                        fragment.idx = point.index
                        fragment.unit = unit
                        self[point.index] = fragment



class Autografs:

    """ 
    Class for the Framework generator.
    Attributes awaited by the constructor:
        - verbose          : true or false
        - refresh_database : either reuses the binary collection, or refresh the full database
        - path             : list of the paths where the fragments can be found in .inp format
    Attributes for the make function
        -- label    : name given to the output files
        -- topology : string, name of the topology used to create structure.
        -- center   : name of the center to use in the given topology, or Fragment object, or list of fragment objects, or list of names.
        -- linker   : name of the linker to use in the given topology, or Fragment object, or list of fragment objects, or list of names.
        -- pillar   : name of the pillar to use in the given topology (optional), or Fragment object.
        -- parallel : if true, use parallel routines for alignement, else, stay serial.
    usage:
        >>> from scm.autografs import *
        >>>
        >>> mofgen = autografs.Autografs()
        >>> framework = mofgen.make(label="SURMOF", topology="pcu", center="Zn_paddlewheel_octahedral", linker="Benzene_linear", pillar="Bipyridine_linear")
        >>> framework
        <Model object at 0x... >
        >>>

    """

    def __init__(self, path, verbose=True, refresh_database=True ):

        self.verbose = verbose
        if self.verbose:
            print "Loading the default Building Units database"
        self.database = {}
        self.index = {}
        if not os.path.exists(path):
            raise RuntimeError("Path to database not valid!")
        if not isinstance(path, list):
            path = [os.path.abspath(path)]
        else:
            path = [os.path.abspath(p) for p in path]
        # load the database of building parts
        for db_loc in path:
            l_database = {}
            if refresh_database or ("db.pickle" not in os.listdir(db_loc)) or ("db_index.pickle" not in os.listdir(db_loc)):
                print "\tRefreshing database..."
                with open(os.path.join(db_loc, "db.pickle"), "wb") as dbfile:
                    for molfile in [mf for mf in os.listdir(db_loc) if mf.endswith(".inp")]:
                        molfile = os.path.join(db_loc, molfile)
                        newmol = read_inp(molfile)
                        l_database.update({newmol.name:newmol})
                    pickle.dump(l_database, dbfile)     
                self.database.update(l_database)
                print "\tRefreshing search index..."               
                with open(os.path.join(db_loc, "db_index.pickle"), "wb") as indexfile:
                    l_index = {m.name:m.shape for m in self.database.itervalues()}
                    pickle.dump(l_index,indexfile)
                self.index.update(l_index)
            else:
                print "\tLoading database..."
                with open(os.path.join(db_loc, "db.pickle"), "rb") as dbfile:
                    l_database = pickle.load(dbfile)
                    self.database.update(l_database)                
                print "\tLoading index file..."
                with open(os.path.join(db_loc, "db_index.pickle"), "rb") as indexfile:
                    l_index = pickle.load(indexfile)
                    self.index.update(l_index)


    def get_available_topologies(self, center=None, linker=None):

        """ 
        Returns a list of available topologies to go with the selected SBUs.
        usage:
            >>> linker = "Benzene_linear"
            >>> center = "UIO66_Zr_icosahedral"
            >>> Autografs.get_available_topologies(center=center, linker=linker)
            ['hcp', 'sqc19']
            >>>
        """

        if center is not None:
            if isinstance(center, list):
                center=center[0]
            if linker is not None:
                if isinstance(linker, list):
                    linker=linker[0]
                available_topologies = [t for t, d in references.iteritems() if (d[0]==self.index[center]
                                                                             and d[1]==self.index[linker])]
                # rectangle and squares are a particular case
                if self.index[center] == "square":
                    available_topologies.extend([t for t, d in references.iteritems() if (d[0]=="rectangle"
                                                                             and d[1]==self.index[linker])])
                if self.index[linker] == "square":
                    available_topologies.extend([t for t, d in references.iteritems() if (d[0]==self.index[center]
                                                                             and d[1]=="rectangle")])
            else:
                available_topologies = [t for t, d in references.iteritems() if  d[0]==self.index[center]]
                if self.index[center] == "square":
                    available_topologies.extend([t for t, d in references.iteritems() if  d[0]=="rectangle"])
        else:
            if linker is not None:
                if isinstance(linker, list):
                    linker=linker[0]                
                available_topologies = [t for t, d in references.iteritems() if  d[1]==self.index[linker]]
                if self.index[linker] == "square":
                    available_topologies.extend([t for t, d in references.iteritems() if  d[1]=="rectangle"])
            else:
                available_topologies = references.keys()
        return available_topologies


    def get_available_linkers(self, topology=None, center=None, combinations=False):

        """ 
        Returns a list of available linkers to go with the selected topology and/or center
        usage:
            >>> topology = "hcp"
            >>> center = "UIO66_Zr_icosahedral"
            >>> Autografs.get_available_linkers(center=center)
            ['H3BTB_Yaghi_triangle', 'TIPA_Yaghi_triangle', 'Benzene_triangle',...]
            >>> Autografs.get_available_linkers(topology=topology)
            ['Betabinaphtol_linear', 'Tetraphenyl_linear',...]
            >>>
        """
        
        if topology is None:
            combinations = False
            topologies = self.get_available_topologies(center=center, linker=None)
            if len(topologies) == 0:
                return []
            else:
                available_linkers = []
                for t in topologies:
                    available_linkers.extend([l for l in self.index.iterkeys() if  self.index[l]==references[t][1]])
                    if references[t][1]=="rectangle":
                        available_linkers.extend([l for l in self.index.iterkeys() if  self.index[l]=="square"])                         
        else:
            available_linkers = [l for l in self.index.iterkeys() if  self.index[l]==references[topology][1]]
            if references[topology][1]=="rectangle":
                available_linkers.extend([l for l in self.index.iterkeys() if  self.index[l]=="square"])
        if combinations:
            possibilities = references[topology][3]
            if possibilities > 1:
                return [list(item) for item in combinations_with_replacement(available_linkers, possibilities)]
            else:
                return available_linkers
        else:
            return available_linkers


    def get_available_pillars(self, topology=None, center=None):

        """ 
        Returns a list of available pillar linkers to go with the selected topology and center
        usage :
            >>> topology = "pcu"
            >>> center = Zn_paddlewheel_octahedral"
            >>> Autografs.get_available_pillars(topology=topology, center=center)
            ['DCDPBN_Yaghi_linear', 'Bipyridine_linear', 'Phenazine_linear']
            >>>
        """
        
        linkers = self.get_available_linkers(topology, center)
        available_pillars = []
        for l in linkers:
            mol = self.database[l]
            mmtypes = [ mol.mmtypes[x.index] for x in mol if x.symbol=="X"]
            if not set(mmtypes) <= set(["H_", "C_R"]):
                available_pillars.append(l)
        return available_pillars


    def get_available_centers(self, topology=None, linker=None, combinations=False):

        """ 
        Returns a list of available centers to go with the selected topology and/or linker
        usage :
            >>> topology = "hcp"
            >>> linker = "H3BTB_Yaghi_triangle"
            >>> Autografs.get_available_centers(topology=topology)
            ['UIO66_Zr_icosahedral']
            >>> Autografs.get_available_centers(linker=linker)
            ['CuCNS_square', 'Zn_porphyrin_square',...] 
        """
        
        if topology is None:
            combinations = False
            topologies = self.get_available_topologies(center=None, linker=linker)
            if len(topologies) == 0:
                return []
            else:
                available_centers = []
                for t in topologies:
                    available_centers.extend([c for c in self.index.iterkeys() if  self.index[c]==references[t][0]])
                    if references[t][0]=="rectangle":
                        available_centers.extend([c for c in self.index.iterkeys() if  self.index[c]=="square"])                      
        else:
            available_centers = [c for c in self.index.iterkeys() if  self.index[c]==references[topology][0]]
            if references[topology][0]=="rectangle":
                available_centers.extend([c for c in self.index.iterkeys() if  self.index[c]=="square"])
        if combinations:
            possibilities = references[topology][2]
            if possibilities > 1:
                return [list(item) for item in combinations_with_replacement(available_centers, possibilities)]
            else:
                return available_centers
        else:
            return available_centers
        

    def residual_error(self, framework):

        """ 
        Returns the RMSE of the distances between dummy atoms of the same tag.
        This gives a measure of the cell scaling on a model.
        """

        framework_tmpatoms = framework.get_atoms(clean=False)
        atoms = Fragment([a for a in framework_tmpatoms if a.symbol=='X'])
        atoms.cell = framework_tmpatoms.cell
        atoms.pbc = framework_tmpatoms.pbc
        tags = [t for t in atoms.get_tags() if t != 0]
        RMSE = 0.0
        L = 0.0
        for tag in tags:
            sameTag = numpy.array(numpy.where(atoms.get_tags()==tag)).ravel()
            if len(sameTag) == 2:
                distance = atoms.get_distance(sameTag[0], sameTag[1], mic=True)
                RMSE += distance**2
                L += 1.0
            elif len(sameTag) > 2:
                print "!!! WARNING !!!"
                print "Element tagged more than once!"
            elif len(sameTag) < 2:
                print "!!! WARNING !!!"
                print "Element left untagged!"                
        RMSE = math.sqrt(RMSE/L)
        return RMSE


    def translate_fragments(self, framework, cell, minimizing=False):

        """ 
        Scales the positions of fragments in an updated cell for the Model given by the framework keyword.
        If minimizing is true , returns an error measure on the distance between dummy atoms of same tag in the scaled framework.
        If minimizing is false, returns the scaled framework.
        the cell keyword should be given as three multipliers in a list.
        usage :
            >>> framework = Autografs.make(....)
            >>> framework.get_cell()
            [[1.0  0.0  0.0]
             [1.0  1.0  0.0]
             [0.0  1.0  1.0]]
            >>> Autografs.translate_fragment(framework, cell=[3.0,5.0,10.0])
            >>> framework.get_cell()
            [[3.0  0.0  0.0]
             [3.0  5.0  0.0]
             [0.0  5.0 10.0]]
        """

        #working copies
        work_framework = framework.copy()
        orig_topo = work_framework.topology.copy()  
        #Update cells
        work_framework.cell = numpy.dot(work_framework.cell, make_diagonal(cell))
        work_framework.topology.set_cell(work_framework.cell, scale_atoms=True)
        #Update distances
        for point in work_framework.topology:
            work_framework[point.index].set_positions([wfp.position + point.position for wfp in work_framework[point.index]])
        #different returns wether we are minimizing the error or just scaling
        if minimizing:
            RMSE = self.residual_error(work_framework)
            # progress bar
            if self.verbose:
                print "RMSE = ", RMSE
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
            return RMSE
        else:
            return work_framework


    def optimize_cell(self, framework, form_factor=None):

        """ 
        This is an automatic function for guessing the best unit cell parameters of the framework.
        The scale factor can be chosen manually, or obtained from the Model.get_form_factor() method.
        Returns a scaled Model object.
        A form factor too small wil result in a collapsing cell, while a too big one gives an exploded one.
        """

        if form_factor is None:
            form_factor=framework.get_form_factor()
        if self.verbose:
            print "Initial cell form factor : ", form_factor, "Angstroms"
            print "If the final cell is badly scaled, consider manually changing it."
        function = lambda cell : self.translate_fragments(framework, cell, minimizing=True)
        print "Scaling with minimum cell parameter = ", form_factor/3.0
        bounds = [(form_factor/3.0, form_factor*100.0),]*3
        parameters = numpy.ones(3)*form_factor
        rez = minimize(fun=function, x0=parameters, bounds=bounds, tol=1e-5)
        if rez.fun > 5.0:
            print "RMSE is too big. The cell should probably be smaller."
            print "Scaling with minimum cell parameter = ", form_factor/10.0
            bounds = [(form_factor/10.0, form_factor*100.0),]*3
            rez = minimize(fun=function, x0=parameters, bounds=bounds, tol=1e-5)
        if rez.fun > 5.0:
            print "RMSE is still too big. Reducing form factor again."
            print "Scaling with minimum cell parameter = ", form_factor/15.0
            bounds = [(form_factor/15.0, form_factor*100.0),]*3
            rez = minimize(fun=function, x0=parameters, bounds=bounds, tol=1e-5)
        if rez.fun > 5.0:
            print "RMSE is still too big. use very small form factor as last ditch attempt to automation."
            print "Scaling with minimum cell parameter = ", form_factor/50
            bounds = [(form_factor/15.0, form_factor*100.0),]*3
            rez = minimize(fun=function, x0=parameters, bounds=bounds, tol=1e-5)
        framework = self.translate_fragments(framework, rez.x, minimizing=False)
        return framework


    def get_sbu(self, slot=None, model=None, center=None, linker=None):
        
        """ 
        Returns the corresponding SBU for the slot given. This is an internal routine, not for external use...
        given two fragments from a "model" Model object, and the make function input for linker and center, 
        we get filling material for the framework Model object.
        """
        
        #centers
        if model.topology[slot.idx].symbol =='C':
            if isinstance(center[0], Fragment):
                sbu=center[0]
            else:
                sbu = self.database[center[0]].copy()
            sbu.unit="center"
        elif model.topology[slot.idx].symbol =='K':
            if len(center)>1:
                if isinstance(center[1], Fragment):
                    sbu=center[1]
                else:
                    sbu = self.database[center[1]].copy()
            else:
                if isinstance(center[0], Fragment):
                    sbu=center[0]
                else:
                    sbu = self.database[center[0]].copy()
            sbu.unit="center"
        #linkers
        elif model.topology[slot.idx].symbol =='N':
            if isinstance(linker[0], Fragment):
                sbu=linker[0]
            else:
                sbu = self.database[linker[0]].copy()
            sbu.unit="linker"
        elif model.topology[slot.idx].symbol =='O':
            if len(linker)>1:
                if isinstance(linker[1], Fragment):
                    sbu=linker[1]
                else:
                    sbu = self.database[linker[1]].copy()
            else:
                if isinstance(linker[0], Fragment):
                    sbu=linker[0]
                else:
                    sbu = self.database[linker[0]].copy()
            sbu.unit="linker"
        elif model.topology[slot.idx].symbol =='P':
            if len(linker)>2:
                if isinstance(linker[2], Fragment):
                    sbu=linker[2]
                else:
                    sbu = self.database[linker[2]].copy()
            else:
                if isinstance(linker[0], Fragment):
                    sbu=linker[0]
                else:
                    sbu = self.database[linker[0]].copy() 
            sbu.unit="linker"
        return sbu     


    def clean_tags(self, tags):

        """ 
        Internal routine. returns all unassigned tags from dummy atoms in the system.
        """

        tags = [t for t in tags if (t != 0 and tags.count(t)<2)]
        return tags 


    def make(self, label='framework', topology=None, center=None, linker=None, pillar=None):

        """ 
        Main function of the Autografs generator object.
        Returns a Model object containing the Framework structure, automatically scaled.
        """

        if not isinstance(center, list):
            center = [center]
        if not isinstance(linker, list):
            linker = [linker]        
        # information about choices is printed
        if self.verbose:
            print "Building blocks chosen:"
            print '\tTopology\t= ', topology
            for c in center:
                print '\tCenter  \t= ', c
            for l in linker:
                print '\tLinker  \t= ', l

        model = Model()
        print 'Creating the model molecule from the selected topolgy.'
        model.make(topology)
        # we need to fill the container
        print "\n"
        if self.verbose:
            print '\nMODEL : '
            print '------\n'
            print model
        # this will be the container molecule
        framework = Model(label=label)
        framework.set_topology(model.topology, fill=False)
        # we need to keep track of the tags positions
        if self.verbose:
            print '\nALIGNMENT OF TAGGED SBUS'
            print '------------------------\n'
        if pillar is None:
            if self.verbose:
                print "Using simultaneous alignment. No concern was given to chemical realism."
            for slot in model:
                sbu = self.get_sbu(slot, model, center, linker)
                # align the slot and the sbu in space
                sbu.center()
                sbu.idx = slot.idx
                sbu, slot = align(sbu, slot)
                framework.append(sbu)
            print ' Aligment completed'
        else:
            if self.verbose:
                print "Using sequential alignment for pillared MOFs."
            available_slots = range(len(model))
            present_tags = []
            pillar_type = self.database[pillar].mmtypes[self.database[pillar].get_chemical_symbols().index('X')]
            print " pillar_type", pillar_type
            while len(framework) != len(model):
                present_tags = self.clean_tags(present_tags)
                if len(framework)==0:
                    # first SBU to align
                    slot = model[available_slots.pop(0)]
                    sbu = self.get_sbu(slot, model, center, linker)
                    sbu, slot = align(sbu, slot)
                    framework.append(sbu)
                    present_tags.extend(slot.get_tags())
                else:
                    slot = None
                    # we select a connected sbu
                    for tag in present_tags:
                        partial_framework = framework.get_atoms(clean=False)
                        tag_index_f = numpy.array(numpy.where(partial_framework.get_tags() == tag)).ravel()[0]
                        tag_type = partial_framework.mmtypes[tag_index_f]
                        # we loop over the available slots until we find a corresponding tag
                        for i in available_slots:
                            if tag in model[i].get_tags():
                                slot = model[i]
                                available_slots.remove(i)
                                break
                        # if a match was found, we get either a pillar or a linker
                        if slot is not None:
                            if len(framework) == len(model):
                                break
                            if tag_type == pillar_type:
                                sbu = self.database[pillar]
                            else:
                                sbu = self.get_sbu(slot, model, center, linker)
                            sbu, slot = align(sbu, slot)
                            framework.append(sbu)
                            present_tags.extend(slot.get_tags())

        if topology in embedded_systems:
            print "Embedded system postprocessing."
            framework.embed(topology=topology, model=model)
        # now get the cell that minimizes distance between identical tags
        print "Unit cell scaling."
        framework = self.optimize_cell(framework)
        print 'Unit cell scaled\n'
        if self.verbose:
            print '\nNew unit cell : '
            print '\n{0} {1} {2}'.format(numpy.around(framework.cell[0,0], decimals=3),
                                         numpy.around(framework.cell[0,1], decimals=3),
                                         numpy.around(framework.cell[0,2], decimals=3))
            print '{0} {1} {2}'.format(numpy.around(framework.cell[1,0], decimals=3),
                                       numpy.around(framework.cell[1,1], decimals=3),
                                       numpy.around(framework.cell[1,2], decimals=3))
            print '{0} {1} {2}\n'.format(numpy.around(framework.cell[2,0], decimals=3),
                                         numpy.around(framework.cell[2,1], decimals=3),
                                         numpy.around(framework.cell[2,2], decimals=3))
        print 'Framework "{0}" generated.'.format(label)
        print '\n'
        print '!!!! DO NOT FORGET THE UFF OPTIMIZATION STEP !!!!'
        print '\n'

        return framework



class MyCompleter(object): 

    """ 
    Custom completer for interactive session
    """

    def __init__(self, options):
        self.options = sorted(options)

    def complete(self, text, state):
        # on first trigger, build possible matches
        if state == 0:  
            # cache matches (entries that start with entered text)
            if text:  
                self.matches = [s for s in self.options if s and s.startswith(text)]
            # no text entered, all matches possible
            else:
                  self.matches = self.options[:]
        # return match indexed by state
        try: 
            return self.matches[state]
        except IndexError:
            return None


if __name__ == '__main__':

    """ 
    Interactive session of the Automated Topological Generator of Frameworks (AuToGraFs)
    """
    import readline
    print 'Please, enter the path to a library of building parts:'
    db = raw_input("\n-----> ").strip()
    mofgen = Autografs(path=db)
    print 'Please, choose a topology in the following list.'
    print '\n\t'.join(["\n\t==========  ======================  =============",
                       "topology    center                  linker",
                       "==========  ======================  =============",
                       "\n\t".join(["{0}\t{1}\t{2}".format(t, d[0], d[1]) for t,d in references.iteritems()])])   

    topocompleter = MyCompleter(references.keys())
    readline.set_completer(topocompleter.complete)
    readline.parse_and_bind('tab: complete')
    topology = raw_input("\n-----> ").strip()
    print "You chose the {0} topology.".format(topology)
    print 'This topology accepts up to {0} centers and {1} linkers.'.format(references[topology][2], references[topology][3])
    print "Please, choose a center in the following list. Multiple selections must be comma separated."
    centers = mofgen.get_available_centers(topology=topology)       
    print ''.join(sorted(['\n\t{0}'.format(sbuname) for sbuname in centers]))
    centercompleter = MyCompleter(centers)
    readline.set_completer(centercompleter.complete)
    readline.parse_and_bind('tab: complete')
    center = [ric.strip() for ric in raw_input("\n-----> ").split(',')]
    if len(center)>1:
        print "You chose the following centers :"
        for c in center:
            print '\t', c
        print 'WARNING : multiple SBU with different sizes are often a source of errors. Check output structures carefully.'
    else:
        print "You chose the center {0}.".format(center[0])
    print "Please, choose a linker in the following list. Multiple selections must be comma separated"
    linkers = mofgen.get_available_linkers(topology=topology)
    print ''.join(sorted(['\n\t{0}'.format(sbuname) for sbuname in linkers]))
    linkercompleter = MyCompleter(linkers)
    readline.set_completer(linkercompleter.complete)
    readline.parse_and_bind('tab: complete')
    linker = [ril.strip() for ril in raw_input("\n-----> ").split(',')]
    if len(linker)>1:
        print "You chose the following linkers :"
        for c in linker:
            print '\t', c
        print 'WARNING : multiple SBU with different sizes are often a source of errors. Check output structures carefully.'
    else:
        print "You chose the linker {0}.".format(linker[0])
    pillars = mofgen.get_available_pillars(topology=topology, center=center)
    pillar = None
    if len(pillars) > 0:
        pillarq = raw_input("Is the framework pillared ? [y/n] : ")
        if pillarq.lower() == "y":
            print "Please, choose a pillar in the following list. Note that only N-connected linear linkers are available"
            print ''.join(sorted(['\n\t{0}'.format(sbuname) for sbuname in pillars]))
            pillarcompleter = MyCompleter(pillars)
            readline.set_completer(pillarcompleter.complete)
            readline.parse_and_bind('tab: complete')
            pillar = raw_input("\n-----> ").strip()
            print "You chose the following pillar :"
            print "\t", pillar

    label = raw_input("Please enter the name of this framework\n-----> ")
    print "Creating the framework"
    mof = mofgen.make(label=label, topology=topology, center=center, linker=linker, pillar=pillar)
    print "Framework created"
    mof.view(clean=True)
    mof.write("test.cif")









