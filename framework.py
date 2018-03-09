import os
import sys
import numpy
import scipy
import typing
import ase

# todo: refine cell + progress of alignment
# from scipy.optimize import minimize_scalar as minimize
# from progress.bar            import Bar

from autografs.utils.sbu        import read_sbu_database
from autografs.utils.topologies import read_topologies_database
from autografs.utils.mmanalysis import analyze_mm 


class Framework(object):
    """
    The Framework object contain the results of an Autografs run.

    Designed to store and post-process the aligned fragments and the
    corresponding topology of an Autografs-made structure. 
    Methods available are setters and getters for the topology, individual SBU,
    rescaling of the total structure, and finally the connecting and returning
    of a clean ASE Atoms object.
    The SBUs are assumed to already be aligned and tagged by the 
    Framework generator, and both bonds and mmtypes are assumed to be ordered.
    The bond matrices are also assumed to be block symmetrical.
    """

    def __init__(self,
                 topology : ase.Atoms = None,
                 SBU      : list      = [],
                 mmtypes  : numpy.ndarray = [],
                 bonds    : numpy.ndarray = []) -> None:
        """Constructor for the Framework class.

        Can be initialized empty and filled consecutively.
        topology -- ASE Atoms object.
        SBU      -- list of ASE Atoms objects.
        mmtypes  -- array of type names. e.g: 'C_R','O_3'...
        bonds    -- block-symmetric matrix of bond orders.
        """
        mmtypes = numpy.asarray(mmtypes)
        bonds   = numpy.asarray(bonds)
        self.topology : ase.Atoms     = topology
        self.SBU      : list          = SBU
        self.mmtypes  : numpy.ndarray = mmtypes
        self.bonds    : numpy.ndarray = bonds
        return None

    def set_topology(self,
                     topology : ase.Atoms) -> None:
        """Set the topology attribute with an ASE Atoms object."""
        self.topology = topology.copy()
        return None

    def get_topology(self) -> ase.Atoms:
        """Return a copy of the topology attribute as an ASE Atoms object."""
        return self.topology.copy()

    def append(self,
               sbu     : ase.Atoms,
               bonds   : numpy.ndarray,
               mmtypes : numpy.ndarray) -> None:
        """Append all data releted to a building unit in the framework.

        This includes the ASE Atoms object, the bonding matrix, and the 
        molecular mechanics atomic types as numpy arrays. These three objects 
        are related through indexing: sbu[i] has a MM type mmtypes[i] and 
        a bonding array of bonds[i,:] or bonds[:,i]
        sbu     -- the Atoms object
        bonds   -- the bonds numpy array, of size len(sbu) by len(sbu).
        mmtypes -- the MM atomic types.
        """
        # first append the atoms object to the list of sbu
        self.SBU.append(sbu)
        # make the bonds matrix with a new block
        self.bonds   = scipy.linalg.block_diag(self.bonds,bonds)
        # append the atom types
        self.mmtypes = numpy.hstack([self.mmtypes,mmtypes])
        return None

    def scale(self,
              alpha  : float = 1.0,
              update : bool  = False) -> None:
        """Scale the building units positions by a factor alpha.

        This uses the correspondance between the atoms in the topology
        and the building units in the SBU list. Indeed, SBU[i] is centered on 
        topology[i]. By scaling the topology, we obtain a new center for the 
        sbu.
        alpha -- scaling factor
        """
        if update:
            topology = self.topology
        else:
            topology = self.topology.copy()
        # get the scaled cell, normalized
        I    = numpy.eye(3)*alpha
        cell = topology.get_cell()
        cell = cell.dot(I/numpy.linalg.norm(cell,axis=0))
        topology.set_cell(cell,scale_atoms=True)
        # then center the SBUs on this position
        for i,sbu in enumerate(self.SBU):
            center = topology[i].position
            cop    =  sbu.positions.mean(axis=0)
            sbu.positions += center - cop
        return None

    def refine(self,
               x0 : numpy.ndarray = [1.0]) -> None:
        """Refine cell scaling to minimize distances between dummies.

        We already have tagged the corresponding dummies during alignment,
        so we just need to calculate the MSE of the distances between 
        identical tags in the complete structure
        x0 -- starting point of the scaling search algorithm
        """

        def MSE(x : numpy.ndarray) -> float:
            """Return cost of scaling as MSE of distances."""
            # scale with this parameter
            self.scale(alpha=x)
            atoms        = self.get_atoms(dummies=True)
            tags         = atoms.get_tags()
            # find the pairs...
            pairs = [numpy.argwhere(tags==tag) for tag in set(tags)]
            # ...and the distances
            d = [atoms.get_distance(i0,i1,mic=True) for i0,i1 in pairs]
            d = numpy.asarray(distances)
            return numpy.mean(distances**2)
        bounds = list(zip(0.5*x0,2.0*x0))
        print(bounds)
        result = scipy.optimize.minimize(fun    = MSE,
                                         x0     = x0,
                                         bounds = bounds)
        self.scale(alpha=result.x)
        return None

    def rotate(self):
        return None

    def flip(self):
        return None

    def functionalize(self):
        return None

    def get_atoms(self,
                  dummies : bool = False) -> ase.Atoms:
        """Return the concatenated Atoms objects.

        The concatenation can either remove the dummies and
        connect the corresponding atoms or leave hem in place
        clean -- remove the dummies if True
        """
        # concatenate every sbu into one Atoms object
        structure = ase.Atoms(cell=self.topology.cell)
        bonds     = self.bonds.copy()
        mmtypes   = self.mmtypes.copy()  
        for sbu in self.SBU:
            structure += sbu
        if not dummies:
            # keep track of dummies
            xis   = [x.index for x in structure if x.symbol=="X"]
            tags  = atoms.get_tags()
            pairs = [numpy.argwhere(tags==tag) for tag in set(tags)]
            for pair in pairs:
                # if lone dummy, cap with hydrogen
                if len(pair)==1:
                    xi0 = pair[0]
                    del xis[xi0]
                    structure.symbols[xi0] = "H"
                    mmtypes[xi0] = "H_" 
                else:
                    xi0,xi1 = pair
                    bonds0  = numpy.where(bonds[xi0,:]>0.0)[0]
                    bonds1  = numpy.where(bonds[xi1,:]>0.0)[0]
                    # the bond order will be the maximum one
                    bo      = max(numpy.amax(bonds[xi0,:]),
                                  numpy.amax(bonds[xi1,:]))
                    # change the bonds
                    ix        = numpy.ix_(bonds0,bonds1)
                    bonds[ix] = bo
                    ix        = numpy.ix_(bonds1,bonds0)
                    bonds[ix] = bo
            # book keeping on what has disappeared
            bonds   = numpy.delete(bonds,xis,axis=0)
            bonds   = numpy.delete(bonds,xis,axis=1)
            mmtypes = numpy.delete(mmtypes,xis)
            del structure[xis]
        return structure
