# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:40:46 2016
Most recently edited: 16J27
@author: Scott

This is the core file of the package. Includes functions for combining EC and Xray
copied from EC_MS on 17L09 as last commited to EC_MS with code c1c6efa
"""

# make python2-compatible:
from __future__ import print_function
from __future__ import division

import numpy as np
import re
import os #, sys    

def synchronize(data_objects, t_zero='start', append=None, cutit=False, 
                override=None, update=True, verbose=True):
    '''
    This will combine array data from multiple dictionaries into a single 
    dictionary with all time variables aligned according to absolute time.
    Data will be retained where the time spans overlap, unless cutit = 0, in 
    which case all data will be retained, but with t=0 at the start of the overlap.
    if t_zero is specified, however, t=0 will be set to t_zero seconds after midnight
    If append=1, data columns of the same name will be joined and filled with
    zeros for sets that don't have them so that all columns remain the same length
    as their time columns, and a data_column 'file number' will be added to 
    keep track of where the data comes from. It's really quite nice... 
    But it's a monster function.
    Could likely benifit from being written from the ground up some time, but not now.
    
    ----  inputs -----
        data_objects: traditionally a list of dictionaries, each of which has
    a key ['data_cols'] pointing to a list of keys for columns of data to be
    synchronized. data_objects can also contain objects with attribute data, in
    which data_objects[i].data is used in the same way as data_objects normally is.
        t_zero: a string or number representing the moment that is considered t=0
    in the synchronized dataset. If t_zero is a number, it is interpreted as time
    in seconds since midnight on the day the datasets were started. t_zero='start'
    means it starts at the start of the overlap. 'first' means t=0 at the earliest 
    datapoint in any data set.
        append: True if identically named data columns should be appended. False
    if the data from the individual sets should be kept in separate columns. By 
    default (append=None), append will get inside the function to True of all of
    the data sets have the same 'data type'.
        cutit: True if data from outside the timespan where all input datasets
    overlap should be removed. This can sometimes make things a bit cleaner to 
    work with in the front-panel scripts, but is not recommended.
        override: True if you don't want the function to pause and ask for your
    consent to continue in the case that there is no range of overlap in the datasets.
    override = False helps you catch errors if you're importing the wrong datasets.
    By default, override gets set to (not append). 
        update: True if you want object.data to be replaced with the synchronized
    dataset for any non-dictionary objects in data_objects
        verbose: True if you want the function to talk to you. Recommended, as it
    helps catch your errors and my bugs. False if you want a clean terminal or stdout
        
    ---- output ----
        the combined and synchronized data set, as a dictionary
    '''
    if verbose:
        print('\n\nfunction \'synchronize\' at your service!')    
    
    if type(data_objects) is not list:
        print('''The first argument to synchronize should be a list of datasets! 
                You have instead input a dictionary as the first argument. 
                I will assume that the first two arguments are the datasets you
                would like to synchronize with standard settings.''')
        data_objects = [data_objects, t_zero]
        t_zero = 'start'
    
    datasets = []
    objects_with_data = []
    for i, dataset in enumerate(data_objects):
        if type(dataset) is dict:
            datasets += [dataset]
        else:
            try:
                data = dataset.data
            except AttributeError:
                print('can\'t get data from data_object number ' + str(i))
                continue
            objects_with_data += [dataset]
            datasets += [data]       
    
    if append is None: #by default, append if all datasets are same type, 17G28
        append = len({d['data_type'] for d in datasets}) == 1
    if override is None:
        override = not append
    if verbose:
        print('append is ' + str(append))
        
                              #prepare to collect some data in the first loop:
    recstarts = []            #first recorded time in each file in seconds since midnight
    t_start = 0               #latest start time (start of overlap) in seconds since midnight
    t_finish = 60*60*24*7     #earliest finish time (finish of overlap) in seconds since midnight 
    t_first = 60*60*24*7      #earliest timestamp in seconds since midnight
    t_last = 0                #latest timestamp in seconds since midnight
    hasdata = {}              #'combining number' of dataset with 0 if its empty or 1 if it has data
    
    combined_data = {'data_type':'combined', 'data_cols':[]}
    title_combined = ''
    
    #go through once to generate the title and get the start and end times of the files and of the overlap
    if verbose:
        print('---------- syncrhonize entering first loop -----------')

    for nd, dataset in enumerate(datasets):
        dataset['combining_number'] = nd
        if 'data_cols' not in dataset or len(dataset['data_cols']) == 0:
            print(dataset['title'] + ' is empty')
            hasdata[nd] = 0
            recstarts += [60*60*24*7] # don't want dataset list to be shortened when sorted later according to recstarts!
            continue
        hasdata[nd] = 1    
        title_combined += dataset['title'] + '__as_' + str(nd) + '__and___'
        if verbose:
            print('working on ' + dataset['title'])
            print('timestamp is ' + dataset['timestamp'])
        t_0 = timestamp_to_seconds(dataset['timestamp'])
        
        t_f = 0
        t_s = 60*60*24*7
        
        for col in dataset['data_cols']:
            if is_time(col):
                try:
                    t_s = min(t_s, t_0 + dataset[col][0])   #earliest start of time data in dataset
                    t_f = max(t_f, t_0 + dataset[col][-1])  #latest finish of time data in dataset
                except IndexError:
                    print(dataset['title'] + ' may be an empty file.')
                    hasdata[nd] = 0
                    
        if hasdata[nd] == 0:
            continue
        recstarts += [t_s]               #first recorded time
    
        t_first = min([t_first, t_0])    #earliest timestamp  
        t_last = max([t_last, t_0])      #latest timestamp 
        t_start = max([t_start, t_s])    #latest start of time variable overall
        t_finish = min([t_finish, t_f])  #earliest finish of time variable overall
    
    title_combined = title_combined[:-6]
    combined_data['title'] = title_combined
    combined_data['tspan_0'] =    [t_start, t_finish] #overlap start and finish times as seconds since midnight
    combined_data['tspan_1'] = [t_start - t_first, t_finish - t_first]    # start and finish times as seconds since earliest start   

    if t_zero == 'start':
        t_zero = t_start
    elif t_zero == 'first':
        t_zero = t_first
    elif t_zero == 'last':
        t_zero = t_last
        
    combined_data['first'] = t_first - t_zero
    combined_data['last'] = t_last - t_zero
    combined_data['start'] = t_start - t_zero
    combined_data['finish'] = t_finish - t_zero
    
    combined_data['timestamp'] = seconds_to_timestamp(t_zero) #we want that timestamp refers to t=0, duh!

    combined_data['tspan'] = [t_start - t_zero, t_finish - t_zero]    #start and finish times of overlap as seconds since t=0  
    combined_data['tspan_2'] = combined_data['tspan'] #old code calls this tspan_2.
    
    if verbose:
        print('first: ' + str(t_first) + ', last: ' + str(t_last) + 
        ', start: ' + str(t_start) + ', finish: ' + str(t_finish))
        
    
    if t_start > t_finish and not override:
        print('No overlap. Check your files.\n')
        offerquit()
    
    print(hasdata)
    N_notempty = len([1 for v in hasdata.values() if v==1])
    if N_notempty == 0:
        print('First loop indicates that no files have data!!! Synchronize will return an empty dataset!!!')
    elif N_notempty == 1:
        print('First loop indicates that only one dataset has data! Synchronize will just return that dataset!')
        combined_data = next(datasets[nd] for nd, v in hasdata.items() if v==1)
        print('\nfunction \'synchronize\' finished!\n\n')
        return combined_data
    
        #this is how I make sure to combine the datasets in the right order!     
    I_sort = np.argsort(recstarts)    
    datasets = [datasets[I] for I in I_sort]       
        #sort by first recorded absolute time. This is so that successive datasets of same type can be joined,
        #with the time variables increasing all the way through 
        #(note: EC lab techniques started together have same timestamp but different recstart)
    
    #and loop again to synchronize the data and put it into the combined dictionary.
    
    if (append and 
        'EC' in {d['data_type'] for d in datasets}): #condition added 17I21
        #add a datacolumn that can be used to separate again afterwards by file
        #as of now (17G28) can't think of an obvious way to keep previous file numbers
        combined_data['file number'] = [] #'file_number' renamed 'file number' for consistency 17H09

    if verbose:
        print('---------- syncrhonize entering second loop -----------')
        
    for i, dataset in enumerate(datasets):
        nd = dataset['combining_number']
        if hasdata[nd] == 0:
            if verbose:
                print('skipping this dataset, because its combining number, ' + str(nd) + ' is in the empty files list')
            continue
        elif verbose:
            print('proceding with this dataset, with combining number ' + str(nd))
            
        if verbose:
            print('cols in ' + dataset['title'] + ':\n' + str(dataset['data_cols']))
            print('cols in combined_data:\n' + str(combined_data['data_cols']))
        #this way names in combined_data match the order the datasets are input with
        t_0 = timestamp_to_seconds(dataset['timestamp'])
        offset = t_0 - t_zero
        
        #first figure out where I need to cut, by getting the indeces striclty corresponding to times lying within the overlap
            
        I_keep = {}    
            #figure out what to keep:
        for col in dataset['data_cols']:     
                #fixed up a bit 17C22, but this whole function should just be rewritten.
            if is_time(col):
                if verbose:
                    print('timeshifting time column ' + col)
                t = dataset[col] + t_0 #absolute time
                I_keep[col] = [I for (I, t_I) in enumerate(t) if t_start < t_I < t_finish]
        
        #then cut, and put it in the new data set
        #first, get the present length of 'time/s' to see how much fill I need in the case of appending EC data from different methods.
        if append:
            print('append is True')
            if 'time/s' in dataset:
                l1 = len(dataset['time/s'])
                fn = np.array([i]*l1)
                if dataset['data_type'] == 'EC':
                    combined_data['file number'] = np.append(combined_data['file number'], fn) 
                    print('len(combined_data[\'file number\']) = ' + str(len(combined_data['file number'])))
            else:
                print('\'time/s\' in Dataset is False')
            if 'time/s' in combined_data:
                l0 = len(combined_data['time/s'])     
            else:
                l0 = 0
        for col in dataset['data_cols']:
            #print(col)
            data = dataset[col] 
            if cutit:           #cut data to only return where it overlaps
                data = data[I_keep[get_time_col(col)]]  
                    #fixed up a bit 17C22, but this whole function should just be rewritten. 
            if is_time(col):
                data = data + offset
            
            if append:
                if col in combined_data:
                    data_0 = combined_data[col]
                else:
                    data_0 = np.array([])
                
                if get_time_col(col) == 'time/s': #rewritten 17G26 for fig3 of sniffer paper
                    #I got l0 before entering the present loop.
                    l2 = len(data_0)
                    if l0>l2:
                        fill = np.array([0]*(l0-l2))
                        #print('filling ' + col + ' with ' + str(len(fill)) + ' zeros')
                        data_0 = np.append(data_0, fill)
                
                #print('len(data_0) = ' + str(len(data_0)) + ', len(data) = ' + str(len(data)))
                data = np.append(data_0, data)
                #print('len(data_0) = ' + str(len(data_0)) + ', len(data) = ' + str(len(data)))
                    
            else:
                if col in combined_data:
                    print('conflicting versions of ' + col + '. adding subscripts.')
                    col = col + '_' + str(nd)                        
                    
            combined_data[col] = data
            if col not in combined_data['data_cols']:
                combined_data['data_cols'].append(col)  
        #offerquit()          
        
        #keep all of the metadata from the original datasets (added 16J27)
        for col, value in dataset.items():
            if col not in dataset['data_cols'] and col not in ['combining_number', 'data_cols']:     #fixed 16J29
                if col in combined_data.keys():
                    combined_data[col + '_' + str(nd)] = value
                else:
                    combined_data[col] = value
                    
    #17G28: There's still a problem if the last dataset is missing columns! Fixing now.  
    if 'time/s' in combined_data:
        combined_data['tspan_EC'] = [combined_data['time/s'][0], combined_data['time/s'][-1]] #this is nice to use
        l1 = len(combined_data['time/s'])
        for col in combined_data['data_cols']:   
            if get_time_col(col) == 'time/s': #rewritten 17G26 for fig3 of sniffer paper
                l2 = len(combined_data[col])
                if l1>l2:
                    fill = np.array([0]*(l1-l2))
                    print('filling ' + col + ' with ' + str(len(fill)) + ' zeros')
                    combined_data[col] = np.append(combined_data[col], fill)
        if append and 'file number' in combined_data.keys():
            combined_data['data_cols'].append('file number') #for new code
            combined_data['file_number'] = combined_data['file number'] #for old code
            combined_data['data_cols'].append('file_number') #for old code

            
    if len(combined_data['data_cols']) == 0:
        print('The input did not have recognizeable data! Synchronize is returning an empty dataset!')    

    if update:
        for instance in objects_with_data:
            instance.data = combined_data

        
    if verbose:
        print('function \'synchronize\' finsihed!\n\n')   
    
    return combined_data        


    
def cut(x, y, tspan=None, returnindeces=False, override=False):
    '''
    Vectorized 17L09 for EC_Xray. Should be copied back into EC_MS
    '''
    if tspan is None:
        return x, y
    
    if np.size(x) == 0:
        print('\nfunction \'cut\' received an empty input\n')
        offerquit()
        
    mask = np.logical_and(tspan[0]<x, x<tspan[-1])
    
    if True not in mask and not override:
        print ('\nWarning! cutting like this leaves an empty dataset!\n' +
               'x goes from ' + str(x[0]) + ' to ' + str(x[-1]) + 
                ' and tspan = ' + str(tspan) + '\n')
        offerquit()
        
    x = x.copy()[mask]
    y = y.copy()[mask]
    
    if returnindeces:
        return x, y, mask #new 17H09
    return x, y


def cut_dataset(dataset_0, tspan=None):
    '''
    Makes a time-cut of a dataset. Written 17H09.
    Unlike time_cut, does not ensure all MS data columns are the same length.
    '''
    print('\n\nfunction \'cut dataset\' at your service!\n') 
    dataset = dataset_0.copy()
    if tspan is None:
        return dataset
    #print(dataset['title'])
    # I need to cunt non-time variables irst
    indeces = {} #I imagine storing indeces improves performance
    for col in dataset['data_cols']:
        #print(col + ', length = ' + str(len(dataset[col])))
        if is_time(col): #I actually don't need this. 
            continue #yes I do, otherwise it cuts it twice.
        timecol = get_time_col(col)
        #print(timecol + ', timecol length = ' + str(len(dataset[timecol])))
        if timecol in indeces.keys():
            #print('already got indeces, len = ' + str(len(indeces[timecol])))
            
            dataset[col] = dataset[col].copy()[indeces[timecol]]
        else:
            #print('about to cut!')
            dataset[timecol], dataset[col], indeces[timecol] = (
             cut(dataset[timecol], dataset[col], 
                 tspan=tspan, returnindeces=True) 
             )
    print('\nfunction \'cut dataset\' finsihed!\n\n') 
    return dataset


def offerquit():
    yn = input('continue? y/n\n')
    if yn == 'n':
        raise SystemExit

    
def is_time(col, verbose=False):
    '''
    determines if a column header is a time variable, 1 for yes 0 for no
    '''
    if verbose:
        print('\nfunction \'is_time\' checking \'' + col + '\'!')
    col_type = get_type(col)
    if col_type == 'EC':
        if col[0:4]=='time':
            return True
        return False
    elif col_type == 'MS':
        if col[-2:] == '-x': 
            return True
        return False
    elif col_type == 'Xray':
        if col == 't':
            return True
        return False
    #in case it the time marker is just burried in a number suffix:
    ending_object = re.search(r'_[0-9][0-9]*\Z',col) 
    if ending_object:
        col = col[:ending_object.start()]
        return is_time(col)
    print('can\'t tell if ' + col + ' is time. Returning False.')
    return False

def is_MS_data(col):
    if re.search(r'^M[0-9]+-[xy]', col):
        return True
    return False

def is_EC_data(col):
    if col in ['mode', 'ox/red', 'error', 'control changes', 'time/s', 'control/V', 
               'Ewe/V', '<I>/mA', '(Q-Qo)/C', 'P/W', 'loop number', 'I/mA', 'control/mA',
               'Ns changes', 'counter inc.', 'cycle number', 'Ns', '(Q-Qo)/mA.h', 
               'dQ/C', 'Q charge/discharge/mA.h', 'half cycle', 'Capacitance charge/µF', 
               'Capacitance discharge/µF', 'dq/mA.h', 'Q discharge/mA.h', 'Q charge/mA.h', 
               'Capacity/mA.h', 'file number', 'file_number', 
               'U vs RHE / [V]', 'J /[mA/cm^2]', 'J / [mA/cm^2]']:
    #this list should be extended as needed
        return True
    return False

def is_Xray_data(col):
    if is_EC_data(col):
        return False
    if is_MS_data(col):
        return False
    return True

def get_type(col):
    if is_MS_data(col):
        return 'MS'
    if is_EC_data(col):
        return 'EC'
    return 'Xray'

def get_time_col(col, verbose=False):
    if is_time(col):
        time_col = col
    elif is_EC_data(col): 
        time_col = 'time/s'
    elif is_Xray_data(col):
        time_col = 't' #I do not like this. I will try and get them to fix this in SPEC
    elif is_MS_data(col):
        time_col = col.replace('-y','-x')        
    else:
        print('don\'t know what ' + col + ' is or what it\'s time col is.')
        time_col = None
    if verbose:
        print('\'' + col + '\' should correspond to time col \'' + str(time_col) +'\'')
    return time_col

def timestamp_to_seconds(timestamp):
    '''
    seconds since midnight derived from timestamp hh:mm:ss
    '''
    h = int(timestamp[0:2])
    m = int(timestamp[3:5])
    s = int(timestamp[6:8])
    seconds = 60**2 *h + 60 *m + s
    return seconds
    
def seconds_to_timestamp(seconds):
    '''
    timestamp hh:mm:ss derived from seconds since midnight
    '''
    h = int(seconds/60**2)
    seconds = seconds - 60**2 *h
    m = int(seconds/60)
    seconds = seconds - 60 *m
    s = int(seconds)
    timestamp = '{0:2d}:{1:2d}:{2:2d}'.format(h,m,s)
    timestamp = timestamp.replace(' ','0')
    return timestamp

def dayshift(dataset, days=1):
    ''' Can work for up to 4 days. After that, hh becomes hhh... 
    '''
    dataset['timestamp'] = seconds_to_timestamp(timestamp_to_seconds(dataset['timestamp']) + days*24*60*60) 
    return dataset

def sort_time(dataset, data_type='EC', verbose=False):
    #17K11: This now operates on the original dictionary, so
    #that I don't need to read the return.
        if verbose:
            print('\nfunction \'sort_time\' at your service!\n\n')
        
        if 'NOTES' in dataset.keys():
            dataset['NOTES'] += '\nTime-Sorted\n'
        else: 
            dataset['NOTES'] = 'Time-Sorted\n'
        
        if data_type == 'all':
            data_type = ['EC','MS']
        elif type(data_type) is str:
            data_type = [data_type]

        sort_indeces = {} #will store sort indeces of the time variables
        data_cols = dataset['data_cols'].copy()
        dataset['data_cols'] = []
        for col in data_cols:
            if verbose:
                print('working on ' + col)
            data = dataset[col] #do I need the copy?
            if get_type(col) in data_type: #retuns 'EC' or 'MS', else I don't know what it is.
                time_col = get_time_col(col, verbose)
                if time_col in sort_indeces.keys():
                    indeces = sort_indeces[time_col]
                else:
                    print('getting indeces to sort ' + time_col)
                    indeces = np.argsort(dataset[time_col])
                    sort_indeces[time_col] = indeces
                if len(data) != len(indeces):
                    if verbose:
                        print(col + ' is not the same length as its time variable!\n' +
                              col + ' will not be included in the time-sorted dataset.')
                else:
                    dataset[col] = data[indeces]
                    dataset['data_cols'] += [col]
                    print('sorted ' + col + '!')
            else: #just keep it without sorting.
                dataset['data_cols'] += [col]
                dataset[col] = data
                

        if verbose:
            print('\nfunction \'sort_time\' finished!\n\n')    
        
        #return dataset#, sort_indeces  #sort indeces are useless, 17J11
    #if I need to read the return for normal use, then I don't want sort_indeces
    
    
    