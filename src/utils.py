'''
Helper function from rasp.cbrain
'''
import numpy as np

def return_var_idxs(ds, var_list, var_cut_off=None):
    """
    To be used on stacked variable dimension. Returns indices array

    Parameters
    ----------
    ds: xarray dataset
    var_list: list of variables

    Returns
    -------
    var_idxs: indices array

    """
    if var_cut_off is None:
        var_idxs = np.concatenate([np.where(ds.var_names == v)[0] for v in var_list])
    else:
        idxs_list = []
        for v in var_list:
            i = np.where(ds.var_names == v)[0]
            if v in var_cut_off.keys():
                i = i[var_cut_off[v]:]
            idxs_list.append(i)
        var_idxs = np.concatenate(idxs_list)
    return var_idxs
