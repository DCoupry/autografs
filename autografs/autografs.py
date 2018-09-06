#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__ = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.3.2'
__status__ = "production"

import os
import sys
import ase
import numpy
import scipy
import logging
import itertools
import scipy.optimize

from collections import defaultdict

from autografs.utils.sbu import read_sbu_database
from autografs.utils.sbu import SBU
from autografs.utils.topology import read_topologies_database
from autografs.utils.topology import Topology
from autografs.framework import Framework

logger = logging.getLogger(__name__)


class Autografs(object):
    """Framework maker class to generate ASE Atoms objects from topologies.

    AuToGraFS: Automatic Topological Generator for Framework Structures.
    Addicoat, M., Coupry, D. E., & Heine, T. (2014).
    The Journal of Physical Chemistry. A, 118(40), 9607–14.
    """

    def __init__(self,
                 topology_path=None,
                 sbu_path=None,
                 use_defaults=True,
                 update=False):
        """Constructor for the Autografs framework maker.

        Parameters
        ----------
        topology_path: str, optional
            Path to a .cgd file for custom topologies.
            see the RCSR files for a description of
            the filetype.
        sbu_path: str, optional
            Path to a .xyz file for custom SBU.
        use_defaults: bool, optional
            If True (default), loads the default libraries
            of topologies (from RCSR) and SBU.
        update: bool, optional
            If True, all databases will be redownloaded
            and analysed anew.

        Returns
        -------
        None
        """
        # some pretty logging
        logger.info("{0:*^80}".format("*"))
        logger.info("* {0:^76} *".format("AuToGraFS"))
        logger.info("* {0:^76} *".format(("Automatic Topological Generator"
                                          "for Framework Structures")))
        logger.info("* {0:^76} *".format(("Addicoat, M., Coupry, "
                                          "D. E., & Heine, T. (2014)")))
        logger.info("* {0:^76} *".format(("The Journal of Physical Chemistry."
                                          " A, 118(40), 9607-14")))
        logger.info("{0:*^80}".format("*"))
        logger.info("")
        # read the furnished databases
        logger.info("Reading the topology database.")
        self.topologies = read_topologies_database(path=topology_path,
                                                   use_defaults=use_defaults)
        logger.info("Reading the building units database.")
        self.sbu = read_sbu_database(path=sbu_path,
                                     use_defaults=use_defaults,
                                     update=update)
        # container for current topology
        self.topology = None
        # container for current sbu mapping
        self.sbu_dict = None
        logger.info("")
        return None

    def set_topology(self,
                     topology_name,
                     supercell=(1, 1, 1)):
        """Create and store the topology object

        Parameters
        ----------
        topology_name: str
            Name of the topology, used as key for
            searching the database. often a three
            letters string: 'pcu','hcb', etc.
        supercell: int, or (int, int, int)
            multiplicator for generation of a supercell
            of the topology, done before any framework
            generation. Useful for statistical defect
            introduction.

        Returns
        -------
        None
        """
        logger.info(("Topology set to --> "
                     "{topo}").format(topo=topology_name.upper()))
        # make the supercell prior to alignment
        if isinstance(supercell, int):
            # always ensur a 3-int long multiplicator
            supercell = (supercell, supercell, supercell)
        # get atoms from database
        topology_atoms = self.topologies[topology_name]
        # only do the work if mult is not 1
        if supercell != (1, 1, 1):
            logger.info(("{0}x{1}x{2} supercell of the topology"
                         "is used.").format(*supercell))
            # function of ASE Atoms objects
            topology_atoms *= supercell
        # make the Topology object
        logger.info("Analysis of the topology.")
        topology = Topology(name=topology_name,
                            atoms=topology_atoms)
        # store it for use as attribute
        self.topology = topology
        logger.info("")
        return None

    def make(self,
             topology_name=None,
             sbu_names=None,
             sbu_dict=None,
             supercell=(1, 1, 1),
             coercion=False):
        """Create a framework using given topology and sbu.

        Main funtion of Autografs. The sbu names and topology's
        are to be taken from the compiled databases. The sbu_dict
        can also be passed for multiple components frameworks.
        If the sbu_names is a list of tuples in the shape
        (name,n), the number n will be used as a drawing probability
        when multiple options are available for the same shape.

        Parameters
        ----------
        topology_name: str, optional
            name of the topology to use. If not given,
            Autografs will try to use its stored
            topology attribute.
        sbu_names: [str,...], optional
            list of names of the sbu to use. The names are the
            keys used to search the SBU database. If not used,
            the sbu_dict option has to be given.
        sbu_dict: {int:str,...}, optional
            slot index to sbu name mapping. ASE Atoms can
            also be given. If used, this argument will take
            precedence over sbu_names.
        supercell: int or (int, int, int), optional
            multiplicator for generation of a supercell
            of the topology, done before any framework
            generation. Useful for statistical defect
            introduction.
        coercion: bool, optional
            force the compatibility detection to only consider
            the multiplicity of SBU: any 4 connected SBU
            can fit any 4-connected slot.

        Returns
        -------
        autografs.framework.Framework
            the scaled, aligned version of the framework
            built using the defined options.
        """
        logger.info("{0:-^50}".format(" Starting Framework Generation "))
        logger.info("")
        self.sbudict = None
        # only set the topology if not already done
        if topology_name is not None:
            self.set_topology(topology_name=topology_name,
                              supercell=supercell)
        # container for the aligned SBUs
        aligned = Framework()
        aligned.set_topology(self.topology)
        # identify the corresponding SBU
        if sbu_dict is None and sbu_names is not None:
            logger.info("Scheduling the SBU to slot alignment.")
            self.sbu_dict = self.get_sbu_dict(sbu_names=sbu_names,
                                              coercion=coercion)
        elif sbu_dict is not None:
            logger.info("SBU to slot alignment is user defined.")
            # the sbu_dict has been passed. if not SBU object, create them
            for k, v in sbu_dict.items():
                if not isinstance(v, SBU):
                    if not isinstance(v, ase.Atoms):
                        name = str(v)
                        v = self.sbu[name].copy()
                    elif "name" in v.info.keys():
                        name = v.info["name"]
                    else:
                        name = str(k)
                    sbu_dict[k] = SBU(name=name,
                                      atoms=v)
            self.sbu_dict = sbu_dict
        else:
            raise ValueError("Either supply sbu_names or sbu_dict.")
        # some logging for pretty information
        for idx, sbu in sbu_dict.items():
            logging.info("\tSlot {sl}".format(sl=idx))
            logging.info("\t   |--> SBU {sbn}".format(sbn=sbu.name))
        # carry on
        alpha = 0.0
        # should be parrallelized one of these days
        for idx, sbu in self.sbu_dict.items():
            # now align and get the scaling factor
            sbu, f = self.align(fragment=self.topology.fragments[idx],
                                sbu=sbu)
            alpha += f
            aligned.append(index=idx,
                           sbu=sbu)
        logger.info("")
        # refine the cell of the aligned object
        aligned.refine(alpha0=alpha)
        # inform user of finished generation
        logger.info("")
        logger.info("Finished framework generation.")
        logger.info("")
        logger.info("{0:-^50}".format(" Post-treatment "))
        logger.info("")
        return aligned

    def get_topology(self,
                     topology_name):
        """Generates and return a Topology object

        Parameters
        ----------
        topology_name: str
            The name of the topology to generate
            The name is the key used to search
            the database.

        Returns
        -------
        autografs.utils.topology.Topology
            The topology object corresponding to the
            ASE Atoms stored in the database under
            the topology_name key.
        """
        topology_atoms = self.topologies[topology_name]
        topology = Topology(name=topology_name,
                            atoms=topology_atoms)
        return topology

    def get_sbu_dict(self,
                     sbu_names,
                     coercion=False):
        """Return a dictionary of SBU by corresponding fragment.

        This stage get a one to one correspondance between
        each topology slot and an available SBU from the list of names.

        Parameters
        ----------
        topology: autografs.utils.topology.Topology
            the topology to use as map for creation
            of the framework.
        sbu_names: [str,...] or [(str,float),...]
            the list of SBU names as strings. accepts probabilistic
            SBU scheduling.
        coercion: bool, optional
            If True, force compatibility by coordination alone

        Returns
        -------
        dict
            {slot index: autografs.utils.sbu.SBU, ...}
            the dictionary is the map that will generate
            the final Framework object
        """
        assert self.topology is not None
        weights = defaultdict(list)
        by_shape = defaultdict(list)
        for name in sbu_names:
            # check if probabilities included
            if isinstance(name, tuple):
                name, p = name
                p = float(p)
                name = str(name)
            else:
                p = 1.0
            # create the SBU object
            sbu = SBU(name=name,
                      atoms=self.sbu[name])
            slots = self.topology.has_compatible_slots(sbu=sbu,
                                                       coercion=coercion)
            if not slots:
                continue
            for slot in slots:
                weights[slot].append(p)
                by_shape[slot].append(sbu)
        # now fill the choices
        sbu_dict = {}
        for index, shape in self.topology.shapes.items():
            # here, should accept weights also
            shape = tuple(shape)
            if shape not in by_shape.keys():
                logger.info("Unfilled slot at index {idx}".format(idx=index))
            p = weights[shape]
            # no weights means same proba
            p /= numpy.sum(p)
            sbu_chosen = numpy.random.choice(by_shape[shape],
                                             p=p).copy()
            sbu_dict[index] = sbu_chosen
        return sbu_dict

    def align(self,
              fragment,
              sbu):
        """Return an aligned SBU.

        The SBU is rotated on top of the fragment
        using the procrustes library within scipy.
        a scaling factor is also calculated for all three
        cell vectors.

        Parameters
        ----------
        fragment:  ase.Atoms
            the slot in the topology on which
            alignment is templated
        sbu: ase.Atoms
            object to align on top of the fragment

        Returns
        -------
        ase.Atoms
            the sbu from input, but scaled and aligned
        1x3 numpy.array(dtype=float)
            the cumulative scaling vector resulting from
            the size difference between the slot and the
            sbu.
        """
        # first, we work with copies
        fragment = fragment.copy()
        # normalize and center
        fragment_cop = fragment.positions.mean(axis=0)
        fragment.positions -= fragment_cop
        sbu.atoms.positions -= sbu.atoms.positions.mean(axis=0)
        # identify dummies in sbu
        sbu_Xis = [x.index for x in sbu.atoms if x.symbol == "X"]
        # get the scaling factor
        sbu_pos = sbu.atoms.get_positions()
        frag_pos = fragment.get_positions()
        size_sbu = numpy.linalg.norm(sbu_pos[sbu_Xis], axis=1)
        size_fragment = numpy.linalg.norm(frag_pos, axis=1)
        alpha_iso = size_sbu.mean()/size_fragment.mean()
        # initial scaling: isotropic.
        fragment.positions = frag_pos.dot(numpy.eye(3)*alpha_iso)
        # getting the rotation matrix
        X0 = sbu_pos[sbu_Xis]
        X1 = fragment.get_positions()
        # trick to get a well defined rotation even
        # when the object is highly symmetric or planar
        if X0.shape[0] > 5:
            X0 = self.get_vector_space(X0)
            X1 = self.get_vector_space(X1)
        # use the scipy implementation
        R, _ = scipy.linalg.orthogonal_procrustes(X0, X1)
        sbu.atoms.positions = sbu.atoms.positions.dot(R)
        # now that the alignment is made, it is pssible
        # to refine a bit the scaling procedure
        alpha = numpy.zeros(3)
        for sbu_xi in sbu_Xis:
            # find corresponding dummmies by distance
            xixi = fragment.get_positions()-sbu.atoms.get_positions()[sbu_xi]
            xixidist = numpy.linalg.norm(xixi, axis=1)
            frag_xi = numpy.argmin(xixidist)
            # calculate the scaling factor for this dummy pair
            size_frag = numpy.linalg.norm(frag_pos[frag_xi])
            size_sbu = numpy.linalg.norm(sbu_pos[sbu_xi])
            # add it, well normalized.
            alpha += numpy.abs(frag_pos[frag_xi]*size_sbu/size_frag)
        # un-center the objects
        sbu.atoms.positions += fragment_cop
        fragment.positions += fragment_cop
        # tag the atoms for connection purposes
        sbu.transfer_tags(fragment)
        return sbu, alpha

    def get_vector_space(self,
                         X):
        """Returns a vector space as four points.

        Parameters
        ----------
        X:  numpy.array(dtype-float)
            the positions of points from which to generate
            an orthogonal vector space

        Returns
        -------
        4x3 numpy.array(dtype=float)
            generated orthogonal vector space
        """
        # initialize
        x0 = X[0]
        # find the point most orthogonal
        dots1 = [x.dot(x0)for x in X]
        i1 = numpy.argmin(dots1)
        x1 = X[i1]
        # the second point maximizes the same with x1
        dots2 = [x.dot(x1) for x in X[1:]]
        i2 = numpy.argmin(dots2)+1
        x2 = X[i2]
        # we find a third point
        dots3 = [x.dot(x1)+x.dot(x0)+x.dot(x2) for x in X]
        i3 = numpy.argmin(dots3)
        x3 = X[i3]
        vs = numpy.asarray([x0, x1, x2, x3])
        return vs

    def list_available_frameworks(self,
                                  topology_name=None,
                                  from_list=[],
                                  coercion=False):
        """Return a list of sbu_dict covering all the database

        It is sometimes useful, for example to generate all
        multiple component versions of a Framework, to have an
        easy wrapper for the necessary loops. This function will
        return all possible permutations of topologically equivalent
        sites using the given topology and list of sbu names, or
        using the full database (very slow!).

        Parameters
        ----------
        topology_name:  str
            The name of the topology to generate
            The name is the key used to search
            the database.
        from_list: [str, ...]
            subset of sbu_names to use for the permutations
            if none is given, the permutatiosn cover all available
            databases elements
        coercion: bool, optional
            If True, force compatibility by coordination alone

        Returns
        -------
        [{slot index: autografs.utils.sbu.SBU, ...}, ...]
            list of all valid sbu_dict that can be passed
            to autografs.Autografs.make() to create a valid
            Fraework object
        """
        av_sbu = self.list_available_sbu(topology_name=topology_name,
                                         from_list=from_list,
                                         coercion=coercion)
        dicts = []
        for product in itertools.product(*av_sbu.values()):
            tmp_d = {}
            for k, v in zip(av_sbu.keys(), product):
                tmp_d.update({kk: v for kk in k})
            dicts.append(tmp_d)
        return dicts

    def list_available_topologies(self,
                                  sbu_names=[],
                                  full=True,
                                  max_size=100,
                                  from_list=[],
                                  pbc="all",
                                  coercion=False):
        """Return a list of topologies compatible with the SBUs

        For each sbu in the list given in input, refines first by coordination
        then by shapes within the topology. Thus, we do not need to analyze
        every topology.

        Parameters
        ----------
        sbu_names: [str,...]
            the list of SBU names as strings
        full: bool, optional
            if True, only list topologies fully
            filled with the given SBU names
        max_size: int
            maximum size of topologies to consider, in number of
            building units
        from_list: [str, ...]
            subset of topologies to consider
        coercion: bool, optional
            If True, force compatibility by coordination alone
        pbc: str, optional
            can be '2D', '3D' or 'all'. Restrits the
            search to topologies of the given periodicity

        Returns
        -------
        list
            list of topology names
        """
        these_topologies_names = self.topologies.keys()
        if max_size is None:
            max_size = 999999
        if from_list:
            these_topologies_names = from_list
        if pbc == "2D":
            logger.info("only considering 2D periodic topologies.")
            these_topologies_names = [tk for tk, tv in self.topologies.items()
                                      if sum(tv.pbc) == 2]
        elif pbc == "3D":
            logger.info("only considering 3D periodic topologies.")
            these_topologies_names = [tk for tk, tv in self.topologies.items()
                                      if sum(tv.pbc) == 3]
        elif pbc != "all":
            logger.info(("pbc keyword has to be '2D','3D'"
                         " or 'all'. Assumed 'all'."))
        these_topologies_names = sorted(these_topologies_names)
        if sbu_names:
            logger.info("Checking topology compatibility.")
            topologies = []
            sbu = [SBU(name=n,
                       atoms=self.sbu[n]) for n in sbu_names]
            for tk in these_topologies_names:
                tv = self.topologies[tk]
                if max_size is None or len(tv) > max_size:
                    continue
                topology = Topology(name=tk,
                                    atoms=tv)
                if topology is None:
                    continue
                # For now, no shape compatibilities
                filled = {shape: False for shape
                          in topology.get_unique_shapes()}
                fslots = [topology.has_compatible_slots(s, coercion=coercion)
                          for s in sbu]
                for slots in fslots:
                    for slot in slots:
                        filled[slot] = True
                if all(filled.values()):
                    logger.info(("\tTopology {tk}"
                                 " fully available.").format(tk=tk))
                    topologies.append(tk)
                elif any(filled.values()) and not full:
                    logger.info(("\tTopology {tk}"
                                 " partially available.").format(tk=tk))
                    topologies.append(tk)
        else:
            logger.info("Listing full database of topologies.")
            topologies = list(self.topologies.keys())
            topologies = sorted(topologies)
        return topologies

    def list_available_sbu(self,
                           topology_name=None,
                           from_list=[],
                           coercion=False):
        """Return the dictionary of compatible SBU.

        Filters the existing SBU by shape until only
        those compatible with a slot within the topology are left.

        Parameters
        ----------
        topology_name: str
            Name of the topology to analyse
        from_list: [str, ...]
            subset of SBU to consider
        coercion: bool, optional
            If True, force compatibility by coordination alone

        Returns
        -------
        list
            list of SBU names
        """
        av_sbu = defaultdict(list)
        if from_list:
            sbu_names = from_list
        else:
            sbu_names = list(self.sbu.keys())
        sbu_names = sorted(sbu_names)
        if topology_name is not None or self.topology is not None:
            if topology_name is not None:
                topology = Topology(name=topology_name,
                                    atoms=self.topologies[topology_name])
            else:
                topology = self.topology
            logger.info(("List of compatible SBU"
                         " with topology {t}:").format(t=topology.name))
            sbu_list = []
            logger.info(("\tShape analysis of"
                         " {le} available SBU...").format(le=len(self.sbu)))
            for sbuk in sbu_names:
                sbuv = self.sbu[sbuk]
                sbu = SBU(name=sbuk,
                          atoms=sbuv)
                if sbu is None:
                    continue
                sbu_list.append(sbu)
            for sites in topology.equivalent_sites:
                logger.info(("\tSites considered"
                             " : {s}").format(s=", ".join(map(str, sites))))
                shape = topology.shapes[sites[0]]
                for sbu in sbu_list:
                    is_compatible = sbu.is_compatible(shape, coercion=coercion)
                    if is_compatible:
                        logger.info("\t\t|--> {k}".format(k=sbu.name))
                        av_sbu[tuple(sites)].append(sbu.name)
            return dict(av_sbu)
        else:
            logger.info("Listing full database of SBU.")
            av_sbu = list(self.sbu.keys())
            av_sbu = sorted(av_sbu)
            return av_sbu


if __name__ == "__main__":
    # Toy example
    molgen = Autografs()
    sbu_names = ["Benzene_linear", "Zn_mof5_octahedral"]
    topology_name = "pcu"
    mof = molgen.make(topology_name=topology_name,
                      sbu_names=sbu_names)
    mof.view()
