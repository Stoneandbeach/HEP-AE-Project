# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:03:07 2021

@author: Stoneandbeach

This script trains several autoencoder models at once. It was used to run AE training on the HTCondor cluster service hosted by CERN.
It is very much a piece of work that has grown organically with the needs of the project, and as such is likely somewhat hard to make sense of.
The basic operating principle is the same as in the project found at https://github.com/Stoneandbeach/ATLAS-collective-AE, so a study of that
project is probably the way to build an understanding of what this script produces. In essence, this script is only built to run the ATLAS-collective-AE
process a number of times with different input parameters.

Please feel free to contact me, the author, with any questions at sten@stoneandbeach.com!
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import fastai
from fastai import learner
from fastai.data import core
from fastai.metrics import mse
from fastai.callback import schedule # Spyder thinks this package isn't being used, but the learner does not work without it

from satools.satools import normalize, unnormalize, group, ungroup, make_DataLoaders

import glob

import vector

def main(*inargs):
    
    print()
    print('- - - - - - - -')
    print()
    
    if sys.version_info[0] > 3 or sys.version_info[0] == 3 and sys.version_info[1] > 8:
        print('Newer version of python than 3.8 detected. This may affect performance.')
    if sys.version_info[0] == 3 and sys.version_info[1] < 8:
        print('Earlier version than python 3.8 detected. Importing pickle5 to compensate.')
    
    global run_nr, norm_params, root_path, storage_path, device, lr_override, branches, dimensions, d2lep, test
    
    intermediate = False
    test = False
    train = False
    evaluate = False
    lxplus = False
    lr_override = False
    use_mass = False
    calculate_E = False
    d2lep = False
    
    print(inargs)
    
    selection = int(inargs[0])
    
    # Set dimensions of input data. Default = 31
    dimensions = 31
    if inargs[1][:4] == 'dim=':
        dimensions = int(inargs[1][4:])
    if 'int' in inargs or 'intermediate' in inargs:
        intermediate = True
    
    test_label = ''
    if 'test' in inargs:
        test = True
        test_label = 'TEST_'
    if 'train' in inargs:
        train = True
        # Using timestamp as run number
        run_nr = r''
        for stamp in list(time.localtime())[:6]:
            stamp_str = str(stamp)
            if len(stamp_str) == 1:
                stamp_str = '0' + stamp_str
            run_nr += stamp_str
        
    if 'evaluate' in inargs:
        evaluate = True
        if not train:
            for i, inarg in enumerate(inargs):
                if inarg == 'evaluate':
                    try:
                        run_nr = inargs[i+1]
                    except:
                        print('Please provide a run number as the next input after "evaluate".')
                        sys.exit()
                    try:
                        run_nr = int(run_nr)
                        run_nr = str(run_nr)
                    except:
                        print('The argument following "evaluate" must be an integer value for run number if not running training.')
                        sys.exit()
    if 'lxplus' in inargs: # This tells the script that it is running on the HTCondor cluster and not locally on my own machine, which affects file paths among other things.
        lxplus = True
        print('Running on lxplus')
    if 'd2lep' in inargs:
        if dimensions != 4:
            print('Dataset "d2lep" can only be used with 4-dimensional input! Aborting.')
            sys.exit()
        d2lep = True
        dataset_name = 'd2lep'
    else:
        dataset_name = 'data18'
    if 'mass' in inargs:
        if d2lep:
            print('Using mass is only valid for input from the "data18" dataset! Aborting.')
            sys.exit()
        use_mass = True
        print('Using mass instead of energy.')
    if 'calculate_E' in inargs:
        if d2lep:
            print('Using calculated energy is only valid for input from the "data18" dataset! Aborting.')
            sys.exit()
        elif dimensions != 4:
            print('31-dimensional input from the "data18" dataset uses calculated energy by default.')
        calculate_E = True
        print('Using calculated energy from mass.')
    if 'scheme' in inargs:
        if not train:
            print('Selecting normalization scheme is only possible for training runs. Aborting.')
            sys.exit()
        for i, inarg in enumerate(inargs):
            if inarg == 'scheme':
                try:
                    scheme = inargs[i+1]
                    print('Normalization scheme chosen:', scheme)
                except:
                    print('Input argument "scheme" must be followed by either "div_by_range", "zero_one" or "log". Aborting.')
                    sys.exit()
    else:
        # Default normalization scheme (options are div_by_range, zero_one, log)
        scheme = 'div_by_range'
        print('Using default normalization scheme:', scheme)
    if 'lr_override' in inargs:
        lr_override = True
        
       
    # Choose device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Set up paths
    if lxplus:
        root_path = r'.'
    else:
        root_path = r'F:\Master\Data'
    
    if train:
        dirs_not_ready = True
        while dirs_not_ready:
            run_found = False
            prev_runs = glob.glob(os.sep.join([root_path, 'storage', '*run_*'])) # Grab a list of directory names of previous runs
      
            for prev_run in prev_runs:
                ix = prev_run.rfind('run_')
                prev_run_nr = prev_run[ix+4:ix+18] # Extract previous run number from directory name
                if prev_run_nr == run_nr:
                    run_found = True
            
            if run_found:
                print('\n- - - WARNING - - -')
                print('Directory with run number', run_nr, 'already exists! Retrying in one second...\n')
                time.sleep(1)
                run_nr = r''
                for stamp in list(time.localtime())[:6]:
                    stamp_str = str(stamp)
                    if len(stamp_str) == 1:
                        stamp_str = '0' + stamp_str
                    run_nr += stamp_str
            else:
                storage_path = os.sep.join([root_path, 'storage', test_label+'run_'+str(run_nr)+'_'+str(selection)+'_'+dataset_name+'_dim_'+str(dimensions)])
                os.mkdir(storage_path)
                dirs_not_ready = False
            
        for path in ['collective-losses', 'model-info', 'models', 'orig-pred', 'run-info']:
            os.mkdir(os.sep.join([storage_path, path]))
        print('Directories created successfully.')
    
    elif evaluate:
        storage_path = os.sep.join([root_path, 'storage', test_label+'run_'+str(run_nr)+'_'+str(selection)+'_'+dataset_name+'_dim_'+str(dimensions)])
    
    # Store selection info for current run
    if train:
        run_info_file = os.sep.join([storage_path, 'run-info', str(run_nr) + '.run'])
        with open(run_info_file, 'w') as file:
            args = ''
            for arg in inargs:
                args += arg + ' '
            file.write(args)
    
    # Load data  
    t = time.perf_counter()
    
    data_path = os.sep.join([root_path, 'pickles'])
    if d2lep:
        filename = r'data_D.2lep-jet.pkl'
    else:
        filename = r'data18_13TeV_pandas_filtered.pkl'
    
    jet_data_filename = os.sep.join([data_path, filename])
    with open(jet_data_filename, 'br') as jet_data_file:
        jet_data = pickle.load(jet_data_file)
    
    print('Jet data read from file:', jet_data_filename)
    print('Data load time:', time.perf_counter() - t, 'seconds.')
    print('Number of jets in dataset:', len(jet_data))
    
    if dimensions == 4:
        if use_mass:
            branch_root = 'JetEtaJESScaleMomentum_'
            jet_data = jet_data[[branch_root + suffix for suffix in ['pt', 'eta', 'phi', 'm']]]
            branches = ['pt', 'eta', 'phi', 'm']
        elif calculate_E:
            branch_root = 'JetEtaJESScaleMomentum_'
            jet_data = jet_data[[branch_root + suffix for suffix in ['pt', 'eta', 'phi']] + ['Calculated_E']]
            branches = ['pt', 'eta', 'phi', 'Calculated_E']
        else:    
            if d2lep:
                jet_data = jet_data[['jet_pt', 'jet_eta', 'jet_phi', 'jet_E']]
                branches = ['pt', 'eta', 'phi', 'E']
            else:
                print('4-dimensional input, requiers either the d2lep dataset (provide input arg "d2lep") or to use mass ("mass") or calculated energy ("calculate_E"). Aborting.')
                sys.exit()
        jet_data.columns = branches
    elif dimensions == 31:
        if use_mass:
            b_file = open('branches_m.txt', 'r')
        else:
            b_file = open('branches_E.txt', 'r')
        branches = []
        for line in b_file.readlines():
            if line[0] != '#':
                branches.append(line.replace('\n', ''))
        b_file.close()
        print('Number of branches selected:', len(branches), '\n')
        #for branch in branches:
        #    print(branch)
    else:
        print('Anomalous input dimension chosen. Please use either 4 or 31. Aborting.')
        sys.exit()
    
    jet_data = jet_data[branches]
    print(jet_data.columns)
    
# =============================================================================
#     each param is a dictionary with
#      n_gs_ls_combinations = len(group_sizes) (which is also = len(latent_space_sizes))
#      n_features : 3 for compression of 4-momenta with intermediate compression, 4 for compression of 4-momenta, 31 for compression of full variable set
#      group_size : 2 or 3 or ...
#      latent_space_size : 5 or 6 or ...
#      jet selection mode : 'same_event', 'random' or 'pt_sort'
#      intermediate : True or False
#      name : str(mode _ group_size _ latent_space_size)
# =============================================================================
        
    model_type = 'collective'
    
    # Select model properties set
    if selection == -1: # This is a testing mode for N x (4-3) compression
        n_gs_ls_combinations = 1
        n_features = [len(branches) - intermediate] * n_gs_ls_combinations
        group_sizes = [2]
        latent_space_sizes = [5]
        modes = ['same_event'] #, 'random', 'pt_sort']
    if selection == 0: # This is a testing mode for N x 31 compression
        n_gs_ls_combinations = 1
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [4]
        latent_space_sizes = [96]
        modes = ['same_event', 'random', 'pt_sort']
        
    if selection == 1:
        n_gs_ls_combinations = 6
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [2, 3, 3, 4, 5, 5]
        latent_space_sizes = [5, 7, 8, 10, 12, 13]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 2:
        n_gs_ls_combinations = 14
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        latent_space_sizes = [3, 6, 5, 9, 8, 7, 12, 11, 10, 9, 15, 14, 13, 12]
        modes = ['same_event', 'random', 'pt_sort']
    
    if selection == 3:
        n_gs_ls_combinations = 20
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        latent_space_sizes = [12, 16, 20, 24, 24, 32, 40, 48, 36, 48, 60, 72, 48, 64, 80, 96, 60, 80, 100, 120]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 4:
        n_gs_ls_combinations = 4
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [1, 1, 1, 1]
        latent_space_sizes = [12, 16, 20, 24]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 5:
        n_gs_ls_combinations = 4
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [2, 2, 2, 2]
        latent_space_sizes = [24, 32, 40, 48]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 6:
        n_gs_ls_combinations = 4
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [3, 3, 3, 3]
        latent_space_sizes = [36, 48, 60, 72]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 7:
        n_gs_ls_combinations = 4
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [4, 4, 4, 4]
        latent_space_sizes = [48, 64, 80, 96]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 8:
        n_gs_ls_combinations = 4
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [5, 5, 5, 5]
        latent_space_sizes = [60, 80, 100, 120]
        modes = ['same_event', 'random', 'pt_sort']
    
    if selection == 9:
        assert dimensions == 4, 'This selection can only be used with 4-dimensional input! Aborting.'
        n_gs_ls_combinations = 1
        n_features = [4]
        group_sizes = [1]
        latent_space_sizes = [3]
        modes = ['same_event']
        model_type = 'single'
        
    if selection == 10:
        n_gs_ls_combinations = 5
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [1, 2, 3, 4, 5]
        latent_space_sizes = [24, 48, 72, 96, 120]
        modes = ['same_event', 'random', 'pt_sort']
    
    if selection == 11:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [2]
        latent_space_sizes = [5]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 12:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [3]
        latent_space_sizes = [7]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 13:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [3]
        latent_space_sizes = [8]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 14:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [4]
        latent_space_sizes = [10]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 15:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [5]
        latent_space_sizes = [12]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 16:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [5]
        latent_space_sizes = [13]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 17:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [4]
        latent_space_sizes = [11]
        modes = ['same_event', 'random', 'pt_sort']
    if selection == 18:
        n_gs_ls_combinations = 1
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [5]
        latent_space_sizes = [14]
        modes = ['same_event', 'random', 'pt_sort']
    
    if selection == 2:
        n_gs_ls_combinations = 14
        n_features = [len(branches)-intermediate] * n_gs_ls_combinations
        group_sizes = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        latent_space_sizes = [3, 6, 5, 9, 8, 7, 12, 11, 10, 9, 15, 14, 13, 12]
        modes = ['same_event', 'random', 'pt_sort']
    if selection > 20 and selection < 40:
        i = selection - 21
        n_gs_ls_combinations = 1
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [[1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5][i]]
        latent_space_sizes = [[3, 6, 5, 9, 8, 7, 12, 11, 10, 9, 15, 14, 13, 12][i]]
        modes = ['same_event', 'random', 'pt_sort']
    
    if selection == 126:
        n_gs_ls_combinations = 1
        n_features = [len(branches)] * n_gs_ls_combinations
        group_sizes = [3]
        latent_space_sizes = [7]
        modes = ['pt_sort']
    
    assert len(group_sizes) == len(latent_space_sizes), 'group_sizes and latent_space_sizes vectors not of same length! Aborting.'
    
    # Training epochs. Note that d2lep is much larger than data18.
    if test:
        n_epochs_base = 1
    elif d2lep:
        n_epochs_base = 500
    else:
        n_epochs_base = 10000
    
    params = []
    for i in range(n_gs_ls_combinations):
        for mode in modes:
            n_epochs = n_epochs_base
            if not test:
                n_epochs = int(n_epochs * group_sizes[i]**0.9)
            param = {'n_features' : n_features[i],
                     'group_size' : group_sizes[i],
                     'latent_space_size' : latent_space_sizes[i],
                     'mode' : mode,
                     'intermediate' : intermediate,
                     'name' : mode + '_' + str(group_sizes[i]) + '_' + str(latent_space_sizes[i]),
                     'n_epochs' : n_epochs,
                     'model_type' : model_type,
                     'scheme' : scheme}
            params.append(param)
        if i == 0:
            print(param)

    
    
    # Split data into training and testing sets
    jet_data, norm_params = normalize(jet_data, scheme=scheme)
    datasets = get_sets(jet_data)
    
    
    # Training loop
    t = time.perf_counter()
    if train:
        losses = {}
        for param in params:
            print()
            print('- - - - - - - - - - -')
            print('Starting training with mode:', param['mode'], ', group size:', param['group_size'], 'latent space size:', param['latent_space_size'])
            if intermediate:
                print('Using intermediate compression.')
            print()
            
            # Run training on this set of parameters
            t_part = time.perf_counter()
            losses[param['name']] = run_training(param, datasets)
            training_time = time.perf_counter() - t_part
            with open(run_info_file, 'a') as f:
                f.write('\n\n')
                for key in param.keys():
                    f.write(key + ' ' + str(param[key]) + '\n')
                f.write('Training time: ' + str(training_time) + '\n')
                f.write('Loss at end of training: ' + str(losses[param['name']]))
                f.write('\n')
                
            if n_epochs_base > 2:
                time.sleep(60)
            
        print('\n ----- \n')
        total_training_time = time.perf_counter() - t
        print('Total training time:', total_training_time, 'seconds.')
        print()
        for param in params:
            print('Losses at end of training of', param['name'], 'was:')
            print(losses[param['name']])
            print()
    
    if evaluate:
        print()
        print('- - - - - - - - - - -')
        print('EVALUATION MODE')
        
        if train:
            print('Evaluating results from this training run.')
        else:
            print('Evaluating results from run number', run_nr)
        
        model_names = get_model_names(run_nr)
        print(model_names)
        
        for model_name in model_names:
            
            print('Evaluating model', model_name)
            print()
            
            run_evaluation(datasets, model_name)
            
        print()
        print('Evaluation complete.')
    
    print()
    print('Total run time:', time.perf_counter() - t, 'seconds.')

def run_training(param, datasets):
    
    datasets = mode_setup(param=param, datasets=datasets)
    
    if param['intermediate']:
        datasets = single_compression(datasets, mode='encode')
        datasets = (datasets[0].detach(), datasets[1].detach())
    
    dls = set_up_dls(param, datasets)
    loss = train(param=param, dls=dls)
    
    return loss

def run_evaluation(datasets, model_name):
    
    # Get model info
    info = get_info(model_name)
    
    # Prepare datasets
    datasets = mode_setup(param=info, datasets=datasets)
    datasets_orig = datasets
    
    if info['intermediate']:
        print('Using intermediate compression.')
        datasets = single_compression(datasets, mode='encode')
        datasets = (datasets[0].detach().numpy(), datasets[1].detach().numpy())
    
    # Get collective model
    if info['model_type'] == 'single':
        model = get_single_model(model_name)
    else:
        model = get_collective_model(model_name)
    
    orig = unnormalize(datasets_orig[1], scheme=info['scheme'], norm_params=norm_params)
    testing_set = datasets[1]
    if isinstance(testing_set, pd.core.frame.DataFrame):
        testing_set = testing_set.values
    pred = ungroup(model(group(torch.tensor(testing_set).to(device), group_size=info['group_size'])), group_size=info['group_size']).detach().numpy()
    if info['intermediate']:
        pred = single_compression(pred, mode='decode')
    pred = pd.DataFrame(pred)
    pred.columns = orig.columns
    
    # Truncate sets if grouping results in them being of different length
    set_length = min(len(pred), len(orig))
    orig = orig.iloc[:set_length]
    pred = pred.iloc[:set_length]
    
    pred.index = orig.index
    pred = unnormalize(pred, scheme=info['scheme'], norm_params=norm_params)
    
    path = os.sep.join([storage_path, 'orig-pred'])
    storage_filename = model_name + '_orig-pred.pkl'
    pd.to_pickle((orig, pred), os.sep.join([path, storage_filename]))
    print('Saving evaluation to', os.sep.join([path, storage_filename]))

def get_model_names(from_run):
    info_path = os.sep.join([storage_path, 'model-info'])
    print()
    print('Loading models from path:', info_path)
    print()

    full_names = glob.glob(os.sep.join([info_path, 'run_'+str(from_run)+'*']))
    
    model_names = []
    for model_name in full_names:
        model_names.append(model_name[model_name.rfind('run'):model_name.rfind('.info')])
        
    assert model_names != [], 'List of model names empty! Is the chosen model name incorrect or not present?'
    
    return model_names

def get_info(model_name):
    info_path = os.sep.join([storage_path, 'model-info'])
    with open(os.sep.join([info_path, model_name + '.info']), 'br') as info_file:
        info = pickle.load(info_file)[0]
        
    return info

def get_sets(data):
    
    data_len = len(data)
    split_idx = int(4/5 * data_len)
    while data.iloc[split_idx].name[1] != 0:
        split_idx += 1
    training_set = data.iloc[:split_idx]
    testing_set = data.iloc[split_idx:]
    return (training_set, testing_set)

def mode_setup(param, datasets):
    
    if param['mode'] == 'same_event':
        datasets = truncate(datasets, param['group_size'])
    
    training_set = datasets[0]
    testing_set = datasets[1]
    
    if param['mode'] == 'random':
        if d2lep:
            if os.path.isfile('cmt_random_training_order.pkl'):
                print('Random training order found.')
                with open('cmt_random_training_order.pkl', 'br') as training_order_file:
                    training_order = pickle.load(training_order_file)
                training_set = training_set.iloc[training_order]
            else:
                print('Random training order not found - creating and storing.')
                training_order = np.array(list(range(len(datasets[0]))))
                np.random.shuffle(training_order)
                pd.to_pickle(training_order, 'cmt_random_training_order.pkl')
                training_set = training_set.iloc[training_order]
            
            if os.path.isfile('cmt_random_testing_order.pkl'):
                print('Random testing order found.')
                with open('cmt_random_testing_order.pkl', 'br') as testing_order_file:
                    testing_order = pickle.load(testing_order_file)
                testing_set = testing_set.iloc[testing_order]
            else:
                print('Random testing order not found - creating and storing.')
                testing_order = np.array(list(range(len(datasets[1]))))
                np.random.shuffle(testing_order)
                pd.to_pickle(testing_order, 'cmt_random_testing_order.pkl')
                testing_set = testing_set.iloc[testing_order]
        else:
            if os.path.isfile('training_random_order.pkl'):
                print('Random training order found.')
                with open('training_random_order.pkl', 'br') as training_order_file:
                    training_order = pickle.load(training_order_file)
                training_set = training_set.iloc[training_order]
            else:
                print('Random training order not found - creating and storing.')
                training_order = np.array(list(range(len(datasets[0]))))
                np.random.shuffle(training_order)
                pd.to_pickle(training_order, 'training_random_order.pkl')
                training_set = training_set.iloc[training_order]
            
            if os.path.isfile('testing_random_order.pkl'):
                print('Random testing order found.')
                with open('testing_random_order.pkl', 'br') as testing_order_file:
                    testing_order = pickle.load(testing_order_file)
                testing_set = testing_set.iloc[testing_order]
            else:
                print('Random testing order not found - creating and storing.')
                testing_order = np.array(list(range(len(datasets[1]))))
                np.random.shuffle(testing_order)
                pd.to_pickle(testing_order, 'testing_random_order.pkl')
                testing_set = testing_set.iloc[testing_order]
    
    if param['mode'] == 'pt_sort':
        
        training_set = training_set.sort_values(by=branches[0])
        testing_set = testing_set.sort_values(by=branches[0])
        
    datasets = (training_set, testing_set)
    
    return datasets

def truncate(datasets, group_size):
    
    training_set = datasets[0]
    testing_set = datasets[1]
    print('\nTotal number of jets in training and testing sets:')
    print(len(training_set), len(testing_set))
    
    check_idxs =  True
    while check_idxs:
        full_mask = np.array([True] * len(training_set))
        sub_idxs = np.array([idx[1] for idx in training_set.index])
        sub_cuts = sub_idxs.copy()
        sub_cuts[:-1] = sub_cuts[:-1] - sub_cuts[1:]
        sub_max = np.where(sub_cuts >= 0)[0]
        jets_per_event = sub_max.copy()
        jets_per_event[1:] = jets_per_event[1:] - jets_per_event[:-1]
        jets_per_event[0] += 1
        mask = np.array(jets_per_event % group_size != 0)
        full_mask[sub_max[mask]] = False
        training_set = training_set[full_mask]
        if sum(mask) == 0:
            check_idxs = False
    
    check_idxs =  True
    while check_idxs:
        full_mask = np.array([True] * len(testing_set))
        sub_idxs = np.array([idx[1] for idx in testing_set.index])
        sub_cuts = sub_idxs.copy()
        sub_cuts[:-1] = sub_cuts[:-1] - sub_cuts[1:]
        sub_max = np.where(sub_cuts >= 0)[0]
        jets_per_event = sub_max.copy()
        jets_per_event[1:] = jets_per_event[1:] - jets_per_event[:-1]
        jets_per_event[0] += 1
        mask = np.array(jets_per_event % group_size != 0)
        full_mask[sub_max[mask]] = False
        testing_set = testing_set[full_mask]
        if sum(mask) == 0:
            check_idxs = False
    
    print('Number of jets remaining after mode "same_event" setup:')
    print(len(training_set), len(testing_set))
    
    return (training_set, testing_set)

def set_up_dls(param, datasets):
       
    batch_size = 256
    if isinstance(datasets[0], pd.DataFrame):
        datasets = (datasets[0].values, datasets[1].values)
    training_data = group(datasets[0], group_size=param['group_size'])
    testing_data = group(datasets[1], group_size=param['group_size'])
    dls = make_DataLoaders(training_data, testing_data, batch_size)
    return dls
    
def single_compression(datasets, mode='both'):
    
    single_model = get_single_model()
    if isinstance(datasets, tuple):
        if isinstance(datasets[0], pd.core.frame.DataFrame):
            datasets = tuple(dataset.values for dataset in datasets)
        compressed_datasets = [single_model(torch.tensor(dataset).to(device), mode=mode) for dataset in datasets]
    else:
        if isinstance(datasets, pd.core.frame.DataFrame):
            datasets = datasets.values
        compressed_datasets = single_model(torch.tensor(datasets).to(device), mode=mode)
    return compressed_datasets
      
def get_single_model(model_name=None):
        
    # Create dummy DataLoaders for use with single compression learner
    loss_func = nn.MSELoss()
    temptrain = TensorDataset(torch.Tensor([1]).to(device), torch.Tensor([1]).to(device))
    temptest = TensorDataset(torch.Tensor([1]).to(device), torch.Tensor([1]).to(device))
    dls = core.DataLoaders(temptrain, temptest)
    model = SingleModel()
    if torch.cuda.is_available():
        model.to('cuda')
    if model_name:
        learn = learner.Learner(dls, model=model, loss_func=loss_func, model_dir=os.sep.join([storage_path, 'models']))
        learn.load(model_name)
    else:
        learn = learner.Learner(dls, model=model, loss_func=loss_func, model_dir=os.sep.join([root_path, 'storage/int-model']))
        learn.load('run_20211025142811_dim_d2lep_4_same_event_1_3')
    return model

def get_collective_model(model_name):
    
    # Create dummy DataLoaders for use with single compression learner
    loss_func = nn.MSELoss()
    temptrain = TensorDataset(torch.Tensor([1]).to(device), torch.Tensor([1]).to(device))
    temptest = TensorDataset(torch.Tensor([1]).to(device), torch.Tensor([1]).to(device))
    dls = core.DataLoaders(temptrain, temptest)
    
    info = get_info(model_name)
    
    model = CollectiveModel(n_features=info['n_features'], group_size=info['group_size'], latent_space_size=info['latent_space_size'])
    if torch.cuda.is_available():
        print('Using GPU.')
        model.to('cuda')
    learn = learner.Learner(dls, model=model, loss_func=loss_func, model_dir=os.sep.join([storage_path, 'models']))
    learn.load(model_name)
    return model

def train(param, dls, model_name=None):
    
    # Set up the model and learner
    
    if param['model_type'] == 'single':
        model = SingleModel()
    else:
        model = CollectiveModel(n_features=param['n_features'], group_size=param['group_size'], latent_space_size=param['latent_space_size'])
    if torch.cuda.is_available():
        print('Using GPU.')
        model.to('cuda')
    
    loss_func = nn.MSELoss()
    
    weight_decay = 1e-6
    
    recorder = learner.Recorder()
    learn = learner.Learner(dls, model=model, wd=weight_decay, loss_func=loss_func, cbs=recorder, model_dir=os.sep.join([storage_path, 'models']))
    
    # Use Learner to find good learning rates

    print()
    print(model.describe())
    print()
    
    if fastai.__version__[:3] == '2.5':
        print('fastai version 2.5.x detected, attempting to import SuggestionFunction "minimum".')
        try:
            from fastai.callback.schedule import minimum
            lrs = learn.lr_find(suggest_funcs=minimum)
            print('Import successful. Using learning rate "minimum".')
        except:
            print('Import failed! Using default learning rate "valley" instead.')
            lrs = learn.lr_find()
    else:
        lrs = learn.lr_find()
    
    print('Learning rate tuple')
    print(lrs)
    print()
    lr_min = lrs[0]
    plt.title(param['name'])
    plt.show();
    
    print('Learning rate with the minimum loss:', lr_min)
    
    # Train the model

    lr_max = lr_min
    print('Training collective AE:', model.describe())
    t = time.perf_counter()
    if lr_override:
        print('Using fixed learning rate', lr_max)
        learn.fit(n_epoch=param['n_epochs'], lr=lr_max)
    else:
        learn.fit_one_cycle(n_epoch=param['n_epochs'], lr_max=lr_max)
    print('Training took', time.perf_counter() - t, 'seconds.')
    
    if model_name == None:
        if d2lep:
            dataset_name = 'd2lep'
        else:
            dataset_name = 'data18'
        model_name = 'run_' + str(run_nr) + '_' + dataset_name + '_dim_' + str(dimensions) + '_' + param['name']
        if test:
            model_name = model_name + '_TEST'
        # for i in range(6):
        #     if i == 3:
        #         model_name = model_name + '_'
        #     model_name = model_name + str(time.localtime()[i])
    
    learn.save(model_name)
    print('Model saved as', model_name)
    
    # Save model information
    model_info = (param, model_name, model.describe())
    info_path = os.sep.join([storage_path, 'model-info'])
    pd.to_pickle(model_info, os.sep.join([info_path, model_name + '.info']))
    
    # Save losses from training run
    values = recorder.values
    loss_path = os.sep.join([storage_path, 'collective-losses'])
    pd.to_pickle(values, os.sep.join([loss_path, model_name + '_losses']))
    
    return learn.validate()

#

class SingleModel(nn.Module):
    def __init__(self, n_features=4, latent_space_size=3):
        super(SingleModel, self).__init__()      
        
        self.n_features = n_features
        self.latent_space_size = latent_space_size
        
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, latent_space_size)
        self.de1 = nn.Linear(latent_space_size, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.act = nn.Tanh()
    
    def encode(self, x):
        return self.en4(self.act(self.en3(self.act(self.en2(self.act(self.en1(x)))))))
    
    def decode(self, x):
        return self.de4(self.act(self.de3(self.act(self.de2(self.act(self.de1(self.act(x))))))))
    
    def forward(self, x, mode = 'both'):
        assert mode in ['both', 'encode', 'decode'], 'model expects keyword mode to be \'encode\', \'decode\' or \'both\'.'
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
        else:
            z = self.encode(x)
            return self.decode(z)
    
    def describe(self):
        description = ''
        for module in [self.en1, self.en2, self.en3, self.en4, self.de1, self.de2, self.de3, self.de4]:
            description += str(module.in_features) + '-'
        description += str(module.out_features) + ', activation function: ' + str(self.act)
        return description

class CollectiveModel(nn.Module):
    def __init__(self, n_features=3, group_size=2, latent_space_size=5):
        super(CollectiveModel, self).__init__()
        
        self.n_features = n_features
        self.group_size = group_size
        self.latent_space_size = latent_space_size
        
        # Layers used for N x 4 and N x (4-3)
        if n_features <= 4:
            self.en1 = nn.Linear(n_features * group_size, 200)
            self.en2 = nn.Linear(200, 200)
            self.en3 = nn.Linear(200, 100)
            self.en4 = nn.Linear(100, self.latent_space_size)
            self.de1 = nn.Linear(latent_space_size, 100)
            self.de2 = nn.Linear(100, 200)
            self.de3 = nn.Linear(200, 200)
            self.de4 = nn.Linear(200, n_features * group_size)
            self.act = nn.Tanh()
        else:
            self.en1 = nn.Linear(n_features * group_size, 300)
            self.en2 = nn.Linear(300, 300)
            self.en3 = nn.Linear(300, 300)
            self.en4 = nn.Linear(300, self.latent_space_size)
            self.de1 = nn.Linear(latent_space_size, 300)
            self.de2 = nn.Linear(300, 300)
            self.de3 = nn.Linear(300, 300)
            self.de4 = nn.Linear(300, n_features * group_size)
            self.act = nn.Tanh()
    
    def encode(self, x):
        return self.en4(self.act(self.en3(self.act(self.en2(self.act(self.en1(x)))))))
    
    def decode(self, x):
        return self.de4(self.act(self.de3(self.act(self.de2(self.act(self.de1(self.act(x))))))))
    
    def forward(self, x, mode = 'both'):
        assert mode in ['both', 'encode', 'decode'], 'model expects keyword mode to be \'encode\', \'decode\' or \'both\'.'
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
        else:
            z = self.encode(x)
            return self.decode(z)
    
    def describe(self):
        description = ''
        for module in [self.en1, self.en2, self.en3, self.en4, self.de1, self.de2, self.de3, self.de4]:
            description += str(module.in_features) + '-'
        description += str(module.out_features) + ', activation function: ' + str(self.act)
        return description

#

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please provide input arguments.')
    else:
        main(*sys.argv[1:])
else:
    print('Please run from command line.')