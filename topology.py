"""
This module stores information on topologies.
This is where new topologies should be implemented, both in references and in the maker functions.
Topologies themselves are taken from the Reticular Chemistry Structure Resource (RCSR)
relevant citations:
  -- O'Keeffe, M.; Peskov, M. A.; Ramsden, S. J.; Yaghi, O. M. ; Accts. Chem. Res. 2008, 41, 1782-1789. 
"""

from ase.lattice.spacegroup import crystal
from ase.visualize import view

# this dictionary stores the geometries available by topology and the number of different objects it takes in
# mainly used for checking and catching bad inputs in Autografs.
references = {"srs"       :  ["triangle"              ,  "linear"      , 1, 1],
              "sql"       :  ["square"                ,  "linear"      , 1, 2],
              "hex_p"     :  ["triangle"              ,  "linear"      , 1, 3],
              "bcu"       :  ["cubic"                 ,  "linear"      , 1, 1],
              "bto"       :  ["triangle"              ,  "linear"      , 1, 2],
              "dia"       :  ["tetrahedral"           ,  "linear"      , 1, 1],
              "sra"       :  ["tetrahedral"           ,  "linear"      , 1, 3],
              "nbo"       :  ["square"                ,  "linear"      , 1, 1],
              "cds"       :  ["square"                ,  "linear"      , 1, 2],
              "bnn"       :  ["trigonal_bipyramid"    ,  "linear"      , 1, 1],
              "pcu"       :  ["octahedral"            ,  "linear"      , 1, 3],
              "pts"       :  ["tetrahedral"           ,  "rectangle"   , 1, 1],
              "ptt"       :  ["tetrahedral"           ,  "rectangle"   , 1, 1],
              "bor"       :  ["tetrahedral"           ,  "triangle"    , 1, 1],
              "ctn"       :  ["tetrahedral"           ,  "triangle"    , 1, 1],
              "pth"       :  ["tetrahedral"           ,  "rectangle"   , 1, 1],
              "pto"       :  ["square"                ,  "triangle"    , 1, 1],
              "pyr"       :  ["octahedral"            ,  "triangle"    , 1, 1],
              "stp"       :  ["tri_prism"             ,  "rectangle"   , 1, 1],
              "soc"       :  ["octahedral"            ,  "rectangle"   , 1, 1],
              "tbo"       :  ["square"                ,  "triangle"    , 1, 1],
              "spn"       :  ["octahedral"            ,  "triangle"    , 1, 1],
              "gar"       :  ["tetrahedral"           ,  "octahedral"  , 1, 1],
              "ibd"       :  ["tetrahedral"           ,  "octahedral"  , 1, 1],
              "iac"       :  ["tetrahedral"           ,  "octahedral"  , 1, 1],
              "ifi"       :  ["tetrahedral"           ,  "triangle"    , 1, 1],
              "rtl"       :  ["octahedral"            ,  "triangle"    , 1, 1],
              "sod"       :  ["tetrahedral"           ,  "linear"      , 1, 1],
              "sqc19"     :  ["icosahedral"           ,  "linear"      , 1, 1],
              "qom"       :  ["octahedral"            ,  "triangle"    , 2, 3],
              "rhr"       :  ["square"                ,  "linear"      , 1, 1],
              "ntt"       :  ["rectangle"             ,  "triangle"    , 1, 2],
              "mil53"     :  ["mil53"                 ,  "linear"      , 1, 1],
              "mfu4"      :  ["mfu4"                  ,  "rectangle"   , 1, 1],
              "afw"       :  ["tetrahedral"           ,  "linear"      , 1, 2],
              "afx"       :  ["tetrahedral"           ,  "linear"      , 1, 1],
              "hcp"       :  ["icosahedral"           ,  "linear"      , 1, 1],
              "reo"       :  ["cubic"                 ,  "linear"      , 1, 1],
              "icx"       :  ["cubic"                 ,  "linear"      , 1, 1],
              "tsi"       :  ["rectangle"             ,  "tetrahedral" , 1, 1],
              "flu"       :  ["cubic"                 ,  "tetrahedral" , 1, 1],
              "scu"       :  ["cubic"                 ,  "rectangle"   , 1, 1],
              "bcs"       :  ["octahedral"            ,  "linear"      , 1, 1],
              "ics"       :  ["tetrahedral"           ,  "linear"      , 1, 1],
              "icv"       :  ["tetrahedral"           ,  "linear"      , 1, 1],
              "qtz"       :  ["tetrahedral"           ,  "linear"      , 1, 1],
              "ftw"       :  ["icosahedral"           ,  "rectangle"   , 1, 1],
              "ocu"       :  ["cubic"                 ,  "octahedral"  , 1, 1],
              "she"       :  ["hexagonal"             ,  "rectangle"   , 1, 1],
              "shp"       :  ["icosahedral"           ,  "rectangle"   , 1, 1],
              "the"       :  ["cubic"                 ,  "triangle"    , 1, 1],
              "toc"       :  ["tri_prism"             ,  "tetrahedral" , 1, 1],
              "ttt"       :  ["icosahedral"           ,  "triangle"    , 1, 1],
              "ivt"       :  ["rectangle"             ,  "linear"      , 1, 1]
              }
# some topologies are impossible to generate easily and call special methods. put'em here.
embedded_systems = ["mil53", "mfu4"]


def make_pcu(model):

    """
    http://rcsr.net/nets/pcu
    """

    print '\tCreating PCU crystal'
    pcu = crystal(['C', 'N', 'O', 'P'], [(0.0, 0.0, 0.0), (0.0,0.5,0.0), (0.5,0.0,0.0), (0.0,0.0,0.5)], 
                  cellpar=[1., 1., 1., 90., 90., 90.]) 
    model.set_topology(pcu)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=pcu, C='octahedral', K=None, N='linear',  O='linear', P='linear', radius=0.25)
    print '\tTagging connections...'
    model.tag(radius=1e-3)
    return

def make_sql(model):

    """
    http://rcsr.net/layers/sql
    """

    print '\tCreating SQL crystal'
    sqp = crystal(['C', 'N', 'O'], [(0.0, 0.0, 0.0), (0.0,0.5,0.0), (0.5,0.0,0.0)], 
                  cellpar=[1., 1., 10., 90., 90., 90.], pbc=(1,1,0))
    model.set_topology(sqp)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=sqp, C='square', K=None, N='linear',  O='linear', P=None, radius=0.25)
    print '\tTagging connections...'
    model.tag(radius=1e-3)
    return

def make_hexp(model):

    """
    http://rcsr.net/layers/hca
    """

    print '\tCreating HEX-P crystal'
    hex_p = crystal(['C', 'C', 'N','O', 'P' ], [(0., 0., 0.), (0.33,0.66,0.), (0.66,0.833,0.),(0.167,0.833,0.),(0.167,0.333,0.)], 
                  cellpar=[1., 1., 10., 90., 90., 120.], pbc=(1,1,0))
    model.set_topology(hex_p)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=hex_p, C='triangle', K=None, N='linear',  O='linear', P='linear', radius=0.25)
    print '\tTagging connections...'
    model.tag(radius=1e-2)
    return

def make_bcu(model):

    """
    http://rcsr.net/nets/bcu
    """

    print '\tCreating BCU crystal: spacegroup 229'
    bcu = crystal(['C', 'N'], [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)], 
                  spacegroup=229, cellpar=[1.1547, 1.1547, 1.1547, 90., 90., 90.])
    model.set_topology(bcu)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=bcu, C='cubic', K=None, N='linear',  O=None, P=None, radius=0.25)
    print '\tTagging connections...'
    model.tag(radius=1e-3)
    return

def make_cds(model):

    """
    http://rcsr.net/nets/cds
    """

    print '\tCreating CDS crystal: spacegroup 131'
    cds = crystal(['C', 'N', 'O'], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.0, 0.5, 0.0)],
                  spacegroup=131, cellpar=[1., 1., 2., 90., 90., 90.])
    model.set_topology(cds)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=cds, C='square', K=None, N='linear', O='linear', P=None, radius=0.25)
    print '\tTagging connections...'
    model.tag(radius=1e-3)           
    return

def make_pyr(model):

    """
    http://rcsr.net/nets/pyr
    """

    print '\tCreating PYR crystal: spacegroup 205...'
    #refined with basin hopping method.
    pyr = crystal(['N', 'C'], [(0.3626942, 0.3626942, 0.3626942), (0.0,0.0,0.0)], 
                  spacegroup=205, cellpar=[1, 1, 1, 90., 90., 90.])
    model.set_topology(pyr)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=pyr, C='octahedral', K=None, N='triangle',  O=None, P=None, radius=0.20)  
    print '\tTagging connections...'
    model.tag(radius=1e-1)         
    return  

def make_srs(model):

    """
    http://rcsr.net/nets/srs
    """

    print '\tCreating SRS crystal: spacegroup 214...'
    srs = crystal(['C', 'N'], [(0.125, 0.125, 0.125), (0.5,0.25,0.875)], 
                  spacegroup=214, cellpar=[2.8284, 2.8284, 2.8284, 90, 90, 90])
    model.set_topology(srs)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=srs, C='triangle', K=None, N='linear',  O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_nbo(model):

    """
    http://rcsr.net/nets/nbo
    """

    print '\tCreating NBO crystal: spacegroup 229...'
    nbo = crystal(['C', 'N'], [(0.0, 0.5, 0.5), (0.25,0.0,0.5)], 
                  spacegroup=229, cellpar=[2, 2, 2, 90, 90, 90])
    model.set_topology(nbo)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=nbo, C='square', K=None, N='linear',  O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_dia(model):

    """
    http://rcsr.net/nets/dia
    """

    print '\tCreating DIA crystal: spacegroup 227...'
    dia = crystal(['C', 'N'], [(0.125, 0.125, 0.125), (0.25,0.5,0.75)],
                  spacegroup=227, setting=2, cellpar=[2.3, 2.3, 2.3, 90, 90, 90])
    model.set_topology(dia)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=dia, C='tetrahedral', K=None, N='linear',  O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_sra(model):

    """
    http://rcsr.net/nets/sra
    """

    print '\tCreating SRA crystal: spacegroup 74...'
    sra = crystal(['C', 'N', 'O', 'P'], [(0.15340, 0.25000, 0.10240), (0.1534,0.5000,0.0000), (0.25, 0.25, 0.25), (0.0000,0.2500,0.1024)],
                     spacegroup=74, cellpar=[3.2592, 1.6842, 2.6331, 90, 90, 90])
    model.set_topology(sra)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=sra, C='tetrahedral', K=None, N='linear', O='linear', P='linear', radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return  

def make_bor(model):

    """
    http://rcsr.net/nets/bor
    """

    print '\tCreating BOR crystal: spacegroup 215...'
    bor = crystal(['N', 'C'], [(0.16667, 0.16667, 0.16667), (0.5,0.0,0.0)], 
                    spacegroup=215, cellpar=[2.4495, 2.4495, 2.4495, 90, 90, 90])
    model.set_topology(bor)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=bor, C='tetrahedral', K=None, N='triangle',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ctn(model):

    """
    http://rcsr.net/nets/ctn
    """

    print '\tCreating CTN crystal: spacegroup 220...'
    ctn = crystal(['N', 'C'], [(0.20830, 0.20830, 0.20830), (0.375,0.0,0.25)], 
                    spacegroup=220, cellpar=[3.7033, 3.7033, 3.7033, 90, 90, 90])
    model.set_topology(ctn)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ctn, C='tetrahedral', K=None, N='triangle',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_pto(model):        

    """
    http://rcsr.net/nets/pto
    """

    print '\tCreating PTO crystal: spacegroup 223...'
    pto = crystal(['N', 'C'], [(0.25, 0.25, 0.25), (0.25,0.0,0.5)],
                    spacegroup=223, cellpar=[2.8284, 2.8284, 2.8284, 90, 90, 90])
    model.set_topology(pto)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=pto, C='square', K=None, N='triangle',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_tbo(model):

    """
    http://rcsr.net/nets/tbo
    """

    print '\tCreating TBO crystal: spacegroup 225...'
    tbo = crystal(['N', 'C'], [(0.33333, 0.33333, 0.33333), (0.25,0.0,0.25)], 
                    spacegroup=225, cellpar=[4.8990, 4.8990, 4.8990, 90, 90, 90])
    model.set_topology(tbo)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=tbo, C='square', K=None, N='triangle',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_bnn(model):

    """
    http://rcsr.net/nets/bnn
    """

    print '\tCreating BNN crystal: spacegroup 191...'
    bnn = crystal(['C', 'N', 'O'], [(0.33333, 0.666667, 0.0), (0.5,0.5,0.0), (0.33333, 0.666667, 0.5)],
                    spacegroup=191, cellpar=[1.7321, 1.7321, 1.0, 90, 90, 120])
    model.set_topology(bnn)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=bnn, C='trigonal_bipyramid', K=None, N='linear', O='linear', P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_pts(model):

    """
    http://rcsr.net/nets/pts
    """

    print '\tCreating PTS crystal: spacegroup 131...'
    pts = crystal(['N', 'C'], [(0.0, 0.5, 0.0), (0.0,0.0,0.25)],
                    spacegroup=131, cellpar=[1.6331 , 1.6331 , 2.3093, 90, 90, 90])
    model.set_topology(pts)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=pts, C='tetrahedral', K=None, N='rectangle',  O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ptt(model):

    """
    http://rcsr.net/nets/ptt
    """

    print '\tCreating PTT crystal: spacegroup 66...'
    ptt = crystal(['C', 'N', 'O'], [(0.14010, 0.0, 0.75), (0.0,0.5,0.75), (0.25,0.25,0.5)],
                    spacegroup=66, cellpar=[4.6535, 1.5165, 3.0847, 90, 90, 90])
    model.set_topology(ptt)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ptt, C='tetrahedral', K=None, N='rectangle', O='rectangle', P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return  

def make_pth(model):

    """
    http://rcsr.net/nets/pth
    """

    print '\tCreating PTH crystal: spacegroup 180...'
    pth = crystal(['C', 'N'], [(0.0, 0.0, 0.5), (0.5,0.0,0.0)],
                    spacegroup=180, cellpar=[1.6330, 1.6330, 3.4641, 90, 90, 120])
    model.set_topology(pth)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=pth, C='tetrahedral', K=None, N='rectangle',  O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return  

def make_stp(model):

    """
    http://rcsr.net/nets/stp
    """

    print '\tCreating STP crystal: spacegroup 191...'
    stp = crystal(['N', 'C'], [(0.5, 0.0, 0.5), (0.33333,0.66667,0.0)],
                    spacegroup=191, cellpar=[2.8287, 2.8287, 1.1545, 90, 90, 120])
    model.set_topology(stp)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=stp, C='tri_prism', K=None, N='rectangle',  O=None, P=None, radius=0.55)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return  

def make_gar(model):

    """
    http://rcsr.net/nets/gar
    """

    print '\tCreating GAR crystal: spacegroup 230...'
    gar = crystal(['C', 'N'], [(0.375, 0.0, 0.25), (0.0,0.0,0.0)],
                    spacegroup=230, cellpar=[3.5770, 3.5770, 3.5770, 90, 90, 90])
    model.set_topology(gar)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=gar, C='tetrahedral', K=None, N='octahedral',  O=None, P=None, radius=0.55)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return     

def make_bto(model):  

    """
    http://rcsr.net/nets/bto
    """

    print '\tCreating BTO crystal: spacegroup 180...'
    bto = crystal(['C', 'N', 'O'], [(0.5000,  0.0000,  0.1111), (0.50,0.0,0.0), (0.5000, 0.2500, 0.1667)],
                    spacegroup=180, cellpar=[1.7321, 1.7321,  4.5000, 90, 90, 90])
    model.set_topology(bto)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=bto, C='triangle', K=None, N='linear', O='linear', P=None, radius=0.35)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return     

def make_soc(model):

    """
    http://rcsr.net/nets/soc
    """

    print '\tCreating STP crystal: spacegroup 229...'
    soc = crystal(['N', 'C'], [(0.25, 0, 0), (0.25, 0.25, 0.25)], 
                    spacegroup=229, cellpar=[2.8284, 2.8284, 2.8284, 90, 90, 90])
    model.set_topology(soc)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=soc, C='octahedral', K=None, N='rectangle',  O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return  

def make_spn(model): 

    """
    http://rcsr.net/nets/spn
    """

    print '\tCreating SPN crystal: spacegroup 227...'
    spn = crystal(['N', 'C'], [(0.16667, 0.16667, 0.16667), (0.0,0.0,0.0)],
                    spacegroup=227, setting=2, cellpar=[4.8990, 4.8990, 4.8990, 90, 90, 90])
    model.set_topology(spn)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=spn, C='octahedral', K=None, N='triangle',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ibd(model): 

    """
    http://rcsr.net/nets/ibd
    """

    print '\tCreating IBD crystal: spacegroup 230...'
    ibd = crystal(['C', 'N'], [(0.375, 0.0, 0.25), (0.25,0.25,0.25)],
                    spacegroup=230, cellpar=[3.266, 3.266, 3.266, 90, 90, 90])
    model.set_topology(ibd)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ibd, C='tetrahedral', K=None, N='octahedral',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_iac(model):

    """
    http://rcsr.net/nets/iac
    """

    print '\tCreating IAC crystal: spacegroup 230...'
    iac = crystal(['C', 'N'], [(0.125, 0.0, 0.25), (0.0,0.0,0.0)],
                    spacegroup=230, cellpar=[3.577, 3.577, 3.577, 90, 90, 90])
    model.set_topology(iac)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=iac, C='tetrahedral', K=None, N='octahedral',  O=None, P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ifi(model): 

    """
    http://rcsr.net/nets/ifi
    """

    print '\tCreating IFI crystal: spacegroup 214...'
    ifi = crystal(['N', 'C'], [(0.125, 0.125, 0.125), (0.625,0.0,0.25)],
                    spacegroup=214, cellpar=[2.1381, 2.1381, 2.1381, 90, 90, 90])
    model.set_topology(ifi)
    print "\t!!! WARNING !!!"
    print "\tThis topology is interpenetrated. Using linkers and centers of vastly different sizes will result in unconnected products."
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ifi, C='tetrahedral', K=None, N='triangle',  O=None, P=None, radius=0.35)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_rtl(model):

    """
    http://rcsr.net/nets/rtl
    """

    print '\tCreating RTL crystal: spacegroup 136...'
    rtl = crystal(['N', 'C'], [(0.3, 0.3, 0.0), (0.0, 0.0, 0.0)], 
                    spacegroup=136, cellpar=[2.3571, 2.3571, 1.4906, 90, 90, 90])
    model.set_topology(rtl)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=rtl, C='octahedral', K=None, N='triangle',  O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=5e-2)         
    return  

def make_sod(model):

    """
    http://rcsr.net/nets/sod
    """

    print '\tCreating SOD crystal: spacegroup 229...'
    sod = crystal(['C', 'N' ], [(0.25, 0.0, 0.5), (0.125, 0.5, 0.125)], 
                    spacegroup=229, cellpar=[2.8284, 2.8284, 2.8284, 90, 90, 90])
    model.set_topology(sod)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=sod, C='tetrahedral', K=None, N='linear',  O=None, P=None, radius=0.3)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_sqc19(model):

    """
    """

    print '\tCreating SQC19 crystal: spacegroup 225...'
    sqc19 = crystal(['C', 'N'], [(0.0, 0.0, 0.0), (-0.25, -0.25, 0.0)], 
            spacegroup=225, cellpar=[1.41421, 1.41421, 1.41421, 90, 90, 90])
    model.set_topology(sqc19)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=sqc19, C='icosahedral', K=None, N='linear',  O=None, P=None, radius=0.3)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)   
    return

def make_qom(model): 

    """
    http://rcsr.net/nets/qom
    """

    print '\tCreating QOM crystal: spacegroup 163...'
    qom = crystal(['C', 'K', 'N', 'O', 'P' ], [(0.33333, 0.66667, 0.75), (0.1667, 0.8333, 0.25), 
                                       (0.0, 0.0, 0.25), (0.33333, 0.66667, 0.25), (0.11110, 0.38890, 0.91670)], 
                                       spacegroup=163, cellpar=[3.4640, 3.4640, 2.8287, 90, 90, 120])
    model.set_topology(qom)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=qom, C='octahedral', K='octahedral', N='triangle', O='triangle', P='triangle', radius=0.50)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_rhr(model):

    """
    http://rcsr.net/nets/rhr
    """

    print '\tCreating RHR crystal: spacegroup 229...'
    rhr = crystal(['C', 'N' ], [(0.33333, 0.33333, 0.0), (0.25, 0.41667, 0.083333)],
                    spacegroup=229, cellpar=[3.4644, 3.4644, 3.4644, 90, 90, 90])
    model.set_topology(rhr)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=rhr, C='square', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_ntt(model):

    """
    http://rcsr.net/nets/ntt
    """

    print '\tCreating NTT crystal: spacegroup 225...'
    ntt = crystal(['C', 'N', 'O'], [(0.0, 0.16667, 0.16667), (0.1668, 0.1668, 0.3332), (0.1111, 0.1111, 0.2222 )],
                    spacegroup=225, cellpar=[7.3485, 7.3485, 7.3485, 90, 90, 90])
    model.set_topology(ntt)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ntt, C='rectangle', K=None, N='triangle', O='triangle', P=None, radius=0.45)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_afw(model):

    """
    http://rcsr.net/nets/afw
    """

    print '\tCreating AFW crystal: spacegroup 155...'
    afw = crystal(['C', 'N', 'O'], [(0.2, 0., 0.), (0.1, 0.1, 0.), (0.2667,  0.9334,  0.8333)],
                    spacegroup=155, cellpar=[2.8868, 2.8868,  2.2362, 90.0  , 90.0 ,   120.0])
    model.set_topology(afw)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=afw, C='tetrahedral', K=None, N='linear', O='linear', P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-1)         
    return 

def make_afx(model):

    """
    http://rcsr.net/nets/afx
    """

    print  '\tCreating AFX crystal: spacegroup 194...'
    afx = crystal(['C', 'C', 'N', 'N', 'N', 'N', 'N', 'N', 'N'], 
                  [(0.0000, 0.2291, 0.0774), (0.3333, 0.4376, 0.1726), (0.8855, 0.1145, 0.0774),
                   (0.1146, 0.2292, 0.0774), (0.0000, 0.2291, 0.0000), (0.0521, 0.3333, 0.1250),
                   (0.4479, 0.5521, 0.1726), (0.2188, 0.4376, 0.1726), (0.3333, 0.4376, 0.2500)],
                    spacegroup=194, cellpar=[4.3653, 4.3653, 6.4623, 90.0, 90.0, 120.0])
    model.set_topology(afx)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=afx, C='tetrahedral', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-1)         
    return 

def make_hcp(model):

    """
    http://rcsr.net/nets/hcp
    """

    print '\tCreating HCP crystal: spacegroup 194...'
    hcp = crystal(['C', 'N', 'N'], [(0.3333 , 0.6667 , 0.2500), (0.8333, 0.1667, 0.2500), (0.5, 0., 0.5)],
                    spacegroup=194, cellpar=[1., 1., 1.633, 90, 90, 120])
    model.set_topology(hcp)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=hcp, C='icosahedral', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=0.1)         
    return 

def make_reo(model):

    """
    http://rcsr.net/nets/reo
    """

    print '\tCreating HCP crystal: spacegroup 221...'
    reo = crystal(['C', 'N'], [(0.5 , 0. , 0.), (0.75, 0.25, 0.)],
                    spacegroup=221, cellpar=[1.4142, 1.4142,  1.4142, 90, 90, 90])
    model.set_topology(reo)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=reo, C='cubic', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=0.1)         
    return 

def make_tsi(model):

    """
    http://rcsr.net/nets/tsi
    """

    print '\tCreating TSI crystal: spacegroup 141...'
    tsi = crystal(['C', 'N', 'N'], [(0., 0.75, 0.125), (0., 0., 0.), (0.5, 0.75, 0.125)],
                    spacegroup=141, cellpar=[1.0000, 1.0000, 3.4640, 90, 90, 90])
    model.set_topology(tsi)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=tsi, C='rectangle', K=None, N='tetrahedral', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=0.1)         
    return 

def make_flu(model):

    """
    http://rcsr.net/nets/flu
    """

    print '\tCreating FLU crystal: spacegroup 225...'
    flu = crystal(['C', 'N'], [(0., 0., 0.), (0.25, 0.25, 0.25)],
                    spacegroup=225, cellpar=[2.3094, 2.3094, 2.3094, 90, 90, 90])
    model.set_topology(flu)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=flu, C='cubic', K=None, N='tetrahedral', O=None, P=None, radius=0.55)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  


def make_scu(model):

    """
    http://rcsr.net/nets/scu
    """

    print '\tCreating SCU crystal: spacegroup 123...'
    scu = crystal(['C', 'N'], [(0., 0., 0.), (0.5, 0., 0.5)],
                    spacegroup=123, cellpar=[1.6330, 1.6330, 1.1547, 90, 90, 90])
    model.set_topology(scu)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=scu, C='cubic', K=None, N='rectangle', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_bcs(model):

    """
    http://rcsr.net/nets/bcs
    """

    print '\tCreating BCS crystal: spacegroup 230...'
    bcs = crystal(['C', 'N'], [(0., 0., 0.), (0.1250, 0.3750, 0.8750)],
                    spacegroup=230, cellpar=[2.3094, 2.3094, 2.3094, 90, 90, 90])
    model.set_topology(bcs)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=bcs, C='octahedral', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ics(model):

    """
    http://rcsr.net/nets/ics
    """

    print '\tCreating ICS crystal: spacegroup 230...'
    ics = crystal(['C', 'N'], [(0.3750, 0., 0.25), (0.4375, 0.8750, 0.3125)],
                    spacegroup=230, cellpar=[3.2663, 3.2663, 3.2663, 90, 90, 90])
    model.set_topology(ics)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ics, C='tetrahedral', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_icv(model):

    """
    http://rcsr.net/nets/icv
    """

    print '\tCreating ICV crystal: spacegroup 214...'
    icv = crystal(['C', 'N'], [(0.125, 0., 0.25), (0.1875, 0.0625, 0.1250)],
                    spacegroup=214, cellpar=[3.2663, 3.2663, 3.2663, 90, 90, 90])
    model.set_topology(icv)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=icv, C='tetrahedral', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_qtz(model):

    """
    http://rcsr.net/nets/qtz
    """

    print '\tCreating QTZ crystal: spacegroup 180...'
    qtz = crystal(['C', 'N'], [(0.5, 0., 0.), (0.25, 0.5, 0.5)],
                    spacegroup=180, cellpar=[1.6330, 1.6330, 1.7321, 90, 90, 120])
    model.set_topology(qtz)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=qtz, C='tetrahedral', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ftw(model):

    """
    http://rcsr.net/nets/ftw
    """

    print '\tCreating FTW crystal: spacegroup 221...'
    ftw = crystal(['C', 'N'], [(0.0, 0., 0.0), (0.5, 0.0, 0.5)],
                    spacegroup=221, cellpar=[1.4142, 1.4142, 1.4142, 90, 90, 90])
    model.set_topology(ftw)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ftw, C='icosahedral', K=None, N='rectangle', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_ocu(model):

    """
    http://rcsr.net/nets/ocu
    """

    print '\tCreating OCU crystal: spacegroup 229...'
    ocu = crystal(['C', 'N'], [(0.5, 0., 0.), (0.25, 0.25, 0.25)],
                    spacegroup=229, cellpar=[2.3094, 2.3094, 2.3094, 90, 90, 90])
    model.set_topology(ocu)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ocu, C='cubic', K=None, N='octahedral', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_she(model):

    """
    http://rcsr.net/nets/she
    """

    print '\tCreating SHE crystal: spacegroup 229...'
    she = crystal(['C', 'N'], [(0.25, 0.25, 0.25), (0.25, 0., 0.5)],
                    spacegroup=229, cellpar=[2.8284, 2.8284, 2.8284, 90, 90, 90])
    model.set_topology(she)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=she, C='hexagonal', K=None, N='rectangle', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_shp(model):

    """
    http://rcsr.net/nets/shp
    """

    print '\tCreating SHP crystal: spacegroup 191...'
    shp = crystal(['C', 'N'], [(0.0, 0., 0.0), (0.5, 0.0, 0.5)],
                    spacegroup=191, cellpar=[1.6330, 1.6330, 1.1547, 90, 90, 120])
    model.set_topology(shp)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=shp, C='icosahedral', K=None, N='rectangle', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_the(model):

    """
    http://rcsr.net/nets/the
    """

    print '\tCreating THE crystal: spacegroup 221...'
    the = crystal(['C', 'N'], [(0.5, 0., 0.), (0.1667, 0.1667, 0.1667)],
                    spacegroup=221, cellpar=[2.4995, 2.4995, 2.4995, 90, 90, 90])
    model.set_topology(the)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=the, C='cubic', K=None, N='triangle', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return  

def make_toc(model):

    """
    http://rcsr.net/nets/toc
    """

    print '\tCreating THE crystal: spacegroup 224...'
    toc = crystal(['C', 'N'], [(0., 0., 0.), (0.25, 0.25, 0.75)],
                    spacegroup=224, setting=2, cellpar=[2.3094, 2.3094, 2.3094, 90, 90, 90])
    model.set_topology(toc)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=toc, C='tri_prism', K=None, N='tetrahedral', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_ttt(model):

    """
    http://rcsr.net/nets/ttt
    """

    print '\tCreating TTT crystal: spacegroup 216...'
    ttt = crystal(['C', 'N'], [(0., 0., 0.), (0.333, 0.333, 0.333)],
                    spacegroup=216, cellpar=[2.4495, 2.4495, 2.4495, 90, 90, 90])
    model.set_topology(ttt)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ttt, C='icosahedral', K=None, N='triangle', O=None, P=None, radius=0.5)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_icx(model):

    """
    http://rcsr.net/nets/icx
    """

    print '\tCreating ICX crystal: spacegroup 223...'
    icx = crystal(['C', 'N'], [(0., 0., 0.), (0.333, 0.333, 0.333)],
                    spacegroup=223, cellpar=[2.4495, 2.4495, 2.4495, 90, 90, 90])
    model.set_topology(icx)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=icx, C='cubic', K=None, N='linear', O=None, P=None, radius=0.34)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_ivt(model):
 
    """
    http://rcsr.net/nets/ivt
    """

    print '\tCreating IVT crystal: spacegroup 141...'
    ivt = crystal(['C', 'N'], [(0., 0., 0.), (0.8750, 0.8750, 0.125)],
                    spacegroup=141, setting=2, cellpar=[2.3099, 2.3099, 2.3099, 90, 90, 90])
    model.set_topology(ivt)
    view(ivt)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=ivt, C='rectangle', K=None, N='linear', O=None, P=None, radius=0.25)  
    print '\tTagging connections...'
    model.tag(radius=1e-3)         
    return 

def make_mtne(model): 

    """
    http://rcsr.net/nets/mtn-e
    """

    print '\tCreating MTN-E crystal: spacegroup 227...'
    print '\t!!! WARNING !!!'
    print '\tThis is an enormous topology. The process will be long.'
    mtn_e = crystal(['C', 'C', 'C', 'C',
             'N', 'N', 'N', 'N', 'N', 'N', 'N' ], 
             [(0.17140, 0.17140, 0.17140), (0.125,0.125,0.37270), (0.20140, 0.20140, 0.29500), (0.25000, 0.15660, 0.40660),
              (0.1864,  0.2332,  0.1864),  
              (0.125 ,  0.1714,  0.125),
              (0.0868,  0.0868,  0.33385),
              (0.1092,  0.0625,  0.38965),
              (0.179 ,  0.2257,  0.3508),
              (0.2014,  0.2482,  0.2482),#N positions are made by taking edges of systre:mtn-e.cgd and taking the midpoint of them with their corresponding Atom. 
              (0.2967,  0.1566,  0.4533)],#Duplicates and equivalents removed
            spacegroup=227, setting=2, cellpar=[7.5837, 7.5837, 7.5837, 90, 90, 90])
    model.set_topology(mtn_e)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=mtn_e, C='tri_prism', N='linear', radius=0.35)  
    print '\tTagging connections...'
    model.tag(radius=1e-2)         
    return  

def make_mil53(model): 
 
    """
    """

    from ase import Atom, Atoms
    print '\tEmbedding hard-coded MIL-53 crystal...'
    mil53 = Atoms('C4N4', positions= [[1.65212500,       4.00000000,       3.00000000], 
                                      [1.65212500,      12.00000000,       9.00000000], 
                                      [4.95637500,      12.00000000,       9.00000000], 
                                      [4.95637500,       4.00000000,       3.00000000], 
                                      [1.65212500 ,      0.00000000 ,      0.00000000],
                                      [4.95637500 ,      8.00000000 ,      6.00000000],
                                      [4.95637500 ,      0.00000000 ,      6.00000000],
                                      [1.65212500 ,      8.00000000 ,      0.00000000]], 
                            cell=[6.6085, 16.0, 12.0], pbc=[1,1,1])
    model.set_topology(mil53)
    print '\tSuperposing shapes on the crystal...'
    model.fill(topology=mil53, C='mil53', N='linear', radius=2.25, scale=False)
    print '\tSpecifying special connections...'
    for fragment in model:
        fragment.mmtypes = ["C_R" if mmtype=="N" else mmtype for mmtype in fragment.mmtypes]
    print '\tTagging connections...'
    model.tag(radius=1e-3)   
    return  

def make_mfu4(model):

    """
    """

    from ase.calculators.neighborlist import NeighborList
    from ase.data import covalent_radii
    from fragment import Fragment
    import numpy as np

    print '\tCreating MFU-4 crystal: spacegroup 225...'
    mfu4 = crystal(['Zn', 'N', 'C'], [(0.25, 0.25, 0.25), (0.0, 0.25, 0.25), (0.0539, 0.7248, 0.7752 )],
                    spacegroup=225, cellpar=[20, 20, 20, 90, 90, 90])
    mfu4_partial = crystal(['C', 'N'], [(0.25, 0.25, 0.25), (0.0, 0.25, 0.25)],
                    spacegroup=225, cellpar=[20, 20, 20, 90, 90, 90])
    model.set_topology(mfu4_partial)
    print '\tManual superposition shapes on the crystal...'
    cutoffs = [covalent_radii[atom.number]*3 if atom.symbol=="Zn" else covalent_radii[atom.number] for atom in mfu4]
    nl = NeighborList(cutoffs, skin=0.1, self_interaction=False, bothways=True)
    nl.build(mfu4)
    for atom in [a for a in mfu4 if a.symbol!="C"]:
        indices, offsets = nl.get_neighbors(atom.index)
        if atom.symbol=="N":
            multiplicity = 4
            shape = "rectangle"
            name = "rectangle"
            unit = "linker"
        elif atom.symbol=="Zn":
            multiplicity = 12
            shape = "mfu4"
            name = "mfu4"
            unit = "center"
        coordinates = [mfu4[index].position + np.dot(mfu4.cell, offset) for index, offset in zip(indices, offsets)]
        idx = [ap.index for ap in mfu4_partial if all(ap.position == atom.position)][0]
        fragment = Fragment("X"*multiplicity, name=name, shape=shape, unit=unit, positions=coordinates, idx=idx)
        model[idx] = fragment
    print '\tTagging connections...'
    model.tag(radius=1e-2)         
    return  
