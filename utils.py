"""
This module is a utilities module, with a few reading, writing, and some general geometry fuctions.
"""

from ase      import Atom
from fragment import *
import numpy as np
import os


def make_diagonal(array):

    """
    Takes an array of length N, returns a diagonal N*N matrix with the arrays
    values in the diagonal 
    """

    matrix = np.zeros((3,3))
    for i in range(3):
        matrix[i,i] = array[i]
    return matrix


def scale_object(coordinates, alpha):

    """
    Scales all values of an array by a factor alpha
    """

    com = np.mean(coordinates, axis=0)
    coordinates -= com
    coordinates *= alpha
    coordinates += com
    return coordinates


def unique_rows(data):

    """
    Returns an array of all the unique rows of the input data.
    """

    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """

    import unicodedata
    import re

    value = unicode(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    return value


def read_inp(f):

    """
    Read as input a file in deMonNano format, and returns a Fragment object as defined in the fragment module.
    """

    mol = Fragment()
    bonds = []
    mol.name = os.path.split(f)[1].split('.')[0]
    try:
        with open(f, 'rb') as i:
            lines = i.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('Data'):
                    dataline = [l.strip() for l in line.split(':')[1].split('=')]
                    if dataline[0] == 'shape':
                        mol.shape = dataline[1]
                    else:
                        pass
                elif line == 'GEOMETRY CARTESIAN':
                    pass
                elif line == 'END':
                    for bond in list(set(bonds)):
                        mol.set_bond(bond[0], bond[1], bond[2])
                else:
                    atomline = [a.strip() for a in line.split()]
                    if len(atomline) != 7:
                        pass
                    else:
                        atom = Atom(atomline[0], position=np.array([atomline[1],atomline[2],atomline[3]]))
                        mol.append(atom)
                        mol.mmtypes[-1] = atomline[4].split('=')[1]
                        for bond in atomline[6].split('=')[1].split(':'):
                            nindex = int(bond.split('/')[0])-1
                            order = round(float(bond.split('/')[1]),2)
                            if nindex > len(mol)-1:
                                bonds.append((len(mol)-1, nindex, order))
                            else:    
                                bonds.append((nindex, len(mol)-1, order))
    except Exception as e:
        raise IOError("Could not read fragment file {0}".format(f))
    return mol


def write_inp(f, mol):
    
    """  Write a DeMonNano .inp file from a Fragment object
    """

    with open(f, 'wb') as of:
        of.write("Data: shape = {0}\n".format(mol.shape))
        of.write("Data: name  = {0}\n".format(mol.name ))
        of.write("GEOMETRY CARTESIAN\n")
        for atom in mol:
            bondIndices = np.array(np.where(mol.bonds[atom.index]!=0)).ravel()
            bonds = [mol.bonds[atom.index, i] for i in bondIndices]
            sym = atom.symbol + " "*(5-len(atom.symbol))
            p0str = str(np.around(atom.position[0],decimals=6))
            p0 = p0str + " "*(10-len(p0str))
            p1str = str(np.around(atom.position[1],decimals=6))
            p1 = p1str + " "*(10-len(p1str))
            p2str = str(np.around(atom.position[2],decimals=6))
            p2 = p2str + " "*(10-len(p2str))
            of.write("{0}  {1}  {2}  {3}   MMTYPE={4}  QMMM=MM BOND={5}\n".format(sym,
                                                                                  p0,
                                                                                  p1,
                                                                                  p2,
                                                                                  mol.mmtypes[atom.index],
                                                                                  ":".join(["{0}/{1}".format(*b) for b in zip(bondIndices+1,bonds)])))
        of.write("END")
    return


def write_gin(f, images):
    
    """
    Write a GULP input file for geometry optimization using the UFF4MOF library.
    """

    with open(f, "wb") as fileobj:
        if not isinstance(images, (list, tuple)):
            images = [images]
        fileobj.write('%s\n' % 'opti conp noautobond fix molmec orthorhombic cartesian')
        if images[0].pbc.any():
            if images[0].pbc[2] == 0:
                fileobj.write('%s\n' % 'svectors')
                fileobj.write('%8.3f %8.3f %8.3f \n' %
                          (images[0].get_cell()[0][0],
                           images[0].get_cell()[0][1],
                           images[0].get_cell()[0][2]))
                fileobj.write('%8.3f %8.3f %8.3f \n' %
                          (images[0].get_cell()[1][0],
                           images[0].get_cell()[1][1],
                           images[0].get_cell()[1][2]))
            else:
                fileobj.write('%s\n' % 'vectors')
                fileobj.write('%8.3f %8.3f %8.3f \n' %
                          (images[0].get_cell()[0][0],
                           images[0].get_cell()[0][1],
                           images[0].get_cell()[0][2]))
                fileobj.write('%8.3f %8.3f %8.3f \n' %
                          (images[0].get_cell()[1][0],
                           images[0].get_cell()[1][1],
                           images[0].get_cell()[1][2]))
                fileobj.write('%8.3f %8.3f %8.3f \n' %
                          (images[0].get_cell()[2][0],
                           images[0].get_cell()[2][1],
                           images[0].get_cell()[2][2]))
        fileobj.write('%s\n' % 'cartesian')
        symbols = images[0].get_chemical_symbols()
        #We need to map MMtypes to numbers. We'll do it via a dictionary
        if any(images[0].get_mmtypes()):
            mmtypes=images[0].get_mmtypes()
            symb_types=[]
            mmdic={}
            types_seen=1
            for m,s in zip(mmtypes,symbols):
                if m not in mmdic:
                    mmdic[m]=s+`types_seen`
                    types_seen+=1
                    symb_types.append(mmdic[m])
                else:
                    symb_types.append(mmdic[m])
        else:
            pass
        for atoms in images:
            for s, (x, y, z), in zip(symb_types, atoms.get_positions()):
                fileobj.write('%-4s %-7s %15.8f %15.8f %15.8f \n' % (s, '   core',x, y, z))

        fileobj.write('%s\n' % '')
        bonds = images[0].get_bonds()
        #write the bonding
        for i in range(len(images[0])):
            for j in range(len(images[0])):
                if i > j and bonds[i, j]==4:
                    fileobj.write('%s %-3d %-3d %10s\n' % ('connect', j+1,i+1, ' quadruple'))
                elif i > j and bonds[i, j]==3:
                    fileobj.write('%s %-3d %-3d %10s\n' % ('connect', j+1,i+1, ' triple'))
                elif i > j and bonds[i, j]==2:
                    fileobj.write('%s %-3d %-3d %10s\n' % ('connect', j+1,i+1, ' double'))
                elif i > j and bonds[i, j]==1.5:
                    fileobj.write('%s %-3d %-3d %10s\n' % ('connect', j+1,i+1, ' resonant'))
                elif i > j and bonds[i, j]==0.5:
                    fileobj.write('%s %-3d %-3d %10s\n' % ('connect', j+1,i+1, ' half'))
                elif i > j and bonds[i, j]==0.25:
                    fileobj.write('%s %-3d %-3d %10s\n' % ('connect', j+1,i+1, ' quarter'))
                elif i > j and bonds[i, j]==1.0:
                    fileobj.write('%s %-3d %-3d \n' % ('connect', j+1,i+1))
                else:
                    pass
        fileobj.write('%s\n' % '')
        fileobj.write('%s\n' % 'species')
        for atoms in images:
            for  k,v in  mmdic.items():
                fileobj.write('%-5s %-5s\n' % (v,k))
        fileobj.write('%s\n' % '')
        fileobj.write('%s\n' % 'library uff4mof.lib')
    return