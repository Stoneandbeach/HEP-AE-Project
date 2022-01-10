"""
This is a set of helper functions for ATLAS-collective-AE.ipynb
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from fastai.data import core
import sys

def normalize(data, scheme='zero_one', no_params=False):
    """
    Normalize data before AE encoding. The data is assumed to be in the column order of [pt, eta, phi, E].
    
    pt and E are assumed to be given in MeV
    
    Normalization schemes available:
    'one_zero': norm_var = orig_var - min(orig_var) / (max(orig_var) - min(orig_var))
    Returns values in range [0, 1]
    
    'div_by_range': norm_var = orig_var / (max(orig_var) - min(orig_var))
    
    'log': [pt, E] norm_var = log10(orig_var * 1e-3)
           [eta, phi] norm_var = orig_var / 3
    """
    if scheme == 'zero_one':
        assert isinstance(data, pd.core.frame.DataFrame), __name__ + ': Please provide input as a pandas.DataFrame.'
        outdata = data.copy()
        norm_params = {}
        for column in outdata.columns:
            dmax = outdata[column].max()
            dmin = outdata[column].min()
            norm_params[column] = (dmin, dmax) # Store min and max values for denormalization
            outdata[column] = (outdata[column] - dmin) / (dmax - dmin)
        if no_params:
            return outdata
        else:
            return outdata, norm_params
    if scheme == 'div_by_range':
        assert isinstance(data, pd.core.frame.DataFrame),  __name__ + ': Please provide input as a pandas.DataFrame.'
        outdata = data.copy()
        norm_params = {}
        for column in outdata.columns:
            dmax = outdata[column].max()
            dmin = outdata[column].min()
            norm_params[column] = (dmin, dmax) # Store min and max values for denormalization
            outdata[column] = outdata[column] / (dmax - dmin)
        if no_params:
            return outdata
        else:
            return outdata, norm_params
    elif scheme == 'log':
        if isinstance(data, pd.core.frame.DataFrame):
            outdata = data.copy()
            outdata.iloc[:, [0, 3]] = np.log10(outdata.iloc[:, [0, 3]] * 1e-3)
            outdata.iloc[:, [1, 2]] = outdata.iloc[:, [1, 2]] / 3
        else:
            outdata = data.copy()
            outdata[:, [0, 3]] = np.log10(outdata[:, [0, 3]] * 1e-3)
            outdata[:, [1, 2]] = outdata[:, [1, 2]] / 3
        if no_params:
            return outdata
        else:
            return outdata, None
    else:
        print('Aborting! Unknown scheme:', scheme)
        sys.exit()

def unnormalize(data, norm_params=None, scheme='zero_one'):
    """
    Unnormalize data after AE decoding, to recover initial magnitudes.
    
    Normalization schemes available:
    'zero_one': requires input (norm_data, norm_params), where
    norm_data is the data to be unnormalized, and
    norm_params is the dictionary returned by normalize()
    
    'div_by_range': requires input (norm_data, norm_params)
    
    'log': takes input (norm_data)
    Returns pt and E in [GeV].
    """
    if scheme == 'zero_one':
        assert norm_params != None, 'Please provide norm_params given by normalize().'
        assert isinstance(data, pd.core.frame.DataFrame),  __name__ + ': Please provide input as a pandas.DataFrame.'
        outdata = data.copy()
        for column in outdata.columns:
            dmax = norm_params[column][1]
            dmin = norm_params[column][0]
            multiplier = dmax - dmin
            outdata[column] = outdata[column] * multiplier + dmin
        return outdata
    if scheme == 'div_by_range':
        assert norm_params != None, 'Please provide norm_params given by normalize().'
        assert isinstance(data, pd.core.frame.DataFrame),  __name__ + ': Please provide input as a pandas.DataFrame.'
        outdata = data.copy()
        for column in outdata.columns:
            multiplier = norm_params[column][1] - norm_params[column][0]
            outdata[column] = outdata[column] * multiplier
        return outdata
    elif scheme == 'log':
        if isinstance(data, pd.core.frame.DataFrame):
            outdata = data.copy()
            outdata.iloc[:, [0, 3]] = 10**outdata.iloc[:, [0, 3]]
            outdata.iloc[:, [1, 2]] = outdata.iloc[:, [1, 2]] * 3
            return outdata
        else:
            print('Please use a pandas.DataFrame as input.')
            sys.exit()
    else:
        print('Aborting! Unknown scheme', scheme)
        sys.exit()

def make_DataLoaders(train, test, batch_size=256, shuffle=True):
    """
    Bundle training and testing data in DataLoaders from torch.utils.data
    
    Since this is intended to be used with an autoencoder, training input is the same as training target. Same goes for testing input.
    
    Input needs to be numpy.array or torch.tensor.
    """
    if type(train) != torch.Tensor:
        train = torch.tensor(train, dtype=torch.float)
    if type(test) != torch.Tensor:
        test = torch.tensor(test, dtype=torch.float)
    train_x = train
    train_y = train_x
    train_ds = TensorDataset(train_x, train_y)
    
    test_x = test
    test_y = test_x
    test_ds = TensorDataset(test_x, test_y)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    dls = core.DataLoaders(train_dl, test_dl)
    return dls

def group(data, group_size=2):
    """
    Create groups by concatenating a number of consecutive rows equal to group_size.
    
    Takes either numpy.ndarray or torch.tensor as input.
    """
    if group_size == 1:
        print('WARNING! Group size is = 1. Is this intentional?')
    data_rem = len(data) % group_size
    if not data_rem == 0:
        print(f'Number of rows in data not divisible by group size {group_size}. Truncating. Size before truncation: {len(data)}')
        data = data[:-data_rem]
        print(f'After truncation: {len(data)}')
    data_cat = data[::group_size]
    for i in range(1, group_size):
        if isinstance(data, np.ndarray):
            data_cat = np.hstack([data_cat, data[i::group_size]])
        else:
            data_cat = torch.hstack([data_cat, data[i::group_size]])
    return data_cat

def ungroup(pred, group_size=2):
    """
    Separate grouped data into individual instances.
    
    Takes either numpy.ndarray or torch.tensor as input.
    """
    if group_size == 1:
        print('WARNING! Group size is = 1. Is this intentional?')
    # Calculate shape of ungrouped data
    num_rows = int(pred.shape[0] * group_size)
    num_cols = int(pred.shape[1] / group_size)
    assert pred.shape[1] % group_size == 0, 'Group size {} does not go evenly into input number of columns {}. Has the data been altered after decoding?'.format(group_size, pred.shape[1])
    
    # Ungroup
    if isinstance(pred, np.ndarray):
        ungrouped_pred = np.zeros([num_rows, num_cols])
    else:
        ungrouped_pred = torch.zeros([num_rows, num_cols])
    for i in range(group_size):
        ungrouped_pred[i::group_size, :] = pred[:, i*num_cols:i*num_cols + num_cols]
    return ungrouped_pred