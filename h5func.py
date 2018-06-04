# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:36:40 2017

@author: cledezma
"""

def export_h5(filename, varnames, var, attr):
    """Export list-like arrays to file using :mod:`h5py`.

    Args:
        filename (str): Filename to export to.
        varnames (str or strs): Name(s) of variable(s) to export.
        var (list or lists): Variable(s) to export.
        attrs (list or lists): Attribute(s) corresponding to variables, given 
        (for each variable) as [[name1, value1], [name2, value2],...]
    """
    import h5py
    with h5py.File(filename, 'w') as f:
        for i in range(len(varnames)):
            dset = f.create_dataset(varnames[i], var[i].shape, var[i].dtype)
            dset[...] = var[i]
            for att in attr[i]:
                dset.attrs.create(att[0],att[1])
                                


def import_h5(filename, varnames):
    """Import list-like arrays to file using :mod:`h5py`.
    Function written by Eric Chen
    
    Args:
        filename (str): Filename to import from.
        varnames (str or strs): Name(s) of variable(s) to import.

    Return:
        vars (list or lists): Imported variable(s).
        attr (list of tuples): Attribute(s) of the imported variable(s)
    """
    import h5py
    dout = []
    attr = []
    with h5py.File(filename, 'r') as f:
        for v in varnames:
            dout += [f[v][...]]
            attBuf = []
            for att in f[v].attrs.items():
                attBuf += [att]
                
            attr += [attBuf]
    return dout, attr