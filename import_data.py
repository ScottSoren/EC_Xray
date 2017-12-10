#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:17:41 2017

@author: scott
"""
import os, platform, re, codecs
from datetime import datetime
import numpy as np

float_match = r'[-]?\d+[\.]?\d*(e[-]?\d+)?'     #matches floats like -3.54e4 or 7 or 245.13 or 1e-15
#note, no white space included on the ends! Seems to work fine.

def numerize(data):
    for col in data['data_cols']: #numerize!
        data[col] = np.array(data[col])    

def get_empty_set(cols, **kwargs):
        #get the colheaders and make space for data   
    data = {}
    data.update(kwargs)
    for col in cols:
        data[col] = []
    data['data_cols'] = cols
    return data

def timetag_to_timestamp(filename):
    '''
    Converts a time tag of format _<hh>h<mm>m<ss>_ to timestamp of format 
    <hh>:<mm>:<ss>, which is what synchronize reads (I should maybe change 
    synchronize to work with the unix epoch timestamp instead...)
    The time tag is something we write in the file names to give an approximate
    sense of when a measurement is started. It can be used as the measurement
    start time if there really is no better way. 
    I can't beleive SPEC doesn't save time. I will pressure them to fix this.
    '''
    hm_match = re.search(r'_[0-9]{2}h[0-9]{2}', filename)
    hm_str = hm_match.group()
    hh = hm_str[1:3]
    mm = hm_str[-2:]
    ss_match = re.search(r'_[0-9]{2}h[0-9]{2}m[0-9]{2}', filename)
    if ss_match is None:
        ss = '00'
    else:
        ss = ss_match.group()[-2:]
    return hh + ':' + mm + ':' + ss
    

def get_creation_timestamp(filepath):
    '''
    Returns creation timestamp of a file in the format that 
    combining.syncrhonize reads
    '''
    t = get_creation_time(filepath)
    time = datetime.fromtimestamp(t)
    hh = str(time.year)
    mm = str(time.minute)
    ss = str(time.second)
    return hh + ':' + mm + ':' + ss
    

def get_creation_time(filepath):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(filepath)
    else:
        stat = os.stat(filepath)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime    


def load_from_csv(filepath, multiset=False, timestamp=None):
    '''
    This function is made a bit more complicated by the fact that some csvs
    seem to have multiple datasets appended, with a new col_header line as the
    only indication. If multiset=True, this will separate them and return them
    as a list.
    if timestamp = None, the timestamp will be the date created
    I hate that SPEC doesn't save absolute time in a useful way.
    '''
    if timestamp is None:
        a = re.search('[0-9]{2}h[0-9]{2}', filepath)
        if a is None:
            print('trying to read creation time')
            timestamp = get_creation_timestamp(filepath)
        else:
            print('getting timestamp from filename ' + filepath)
            timestamp = timetag_to_timestamp(filepath)
            
    with open(filepath,'r') as f: # read the file!
        lines = f.readlines()
    colheaders = [col.strip() for col in lines[0].split(',')]
    data = get_empty_set(colheaders, title=filepath, 
                         timestamp=timestamp, data_type='SPEC')
    datasets = []
    
    for line in lines[1:]: #put data in lists!
        vals = [val.strip() for val in line.split(',')]
        not_data = []
        newline = {}
        for col, val in zip(colheaders, vals):
            if col in data['data_cols']:
                try:
                    val = float(val)
                except ValueError:
                    print('value ' + val + ' of col ' + col + ' is not data.')
                    not_data += [col]   
            newline[col] = val
        if len(not_data) == len(data['data_cols']):
            print('it looks like there is another data set appended!')
            if multiset:
                print('continuing to next set.')
                numerize(data)
                datasets += [data.copy()]
                colheaders = [val.strip() for val in vals]
                data = get_empty_set(colheaders, 
                         timestamp=timestamp, data_type='SPEC')
                continue
            else:
                print('returning first set.')
                numerize(data)
                return data
        else:    
            for col in not_data:
                data['data_cols'].remove(col)
                print('column ' + col + ' removed from \'data_cols \'.')

        for col, val in zip(colheaders, vals):
            data[col] += [newline[col]]
    
    numerize(data)
    datasets += [data]
    if multiset:
        return datasets
    return data
            

def read_macro(file):
    with open(file) as macro:
        lines = macro.readlines()
    lines = remove_comments(lines)
    settings = {'tth':[], 'alpha':[], 'savepath':[], 'newfile':[], 'measurements':[]}
    for line in lines:
        #print(line)
        tth_match = re.search('umv tth ' + float_match, line)
        if tth_match:
            #print('got tth!')
            settings['tth'] += [float(tth_match.group()[7:])]
            continue
        alpha_match = re.search('umv th ' + float_match, line)
        if alpha_match:
            settings['alpha'] += [float(alpha_match.group()[6:])]
            continue
        if 'pd savepath' in line:
            settings['savepath'] += [line[12:]]
            continue
        if 'newfile ' in line:
            settings['newfile'] += [line[8:]]
            continue
        if '_timescan ' in line or 'ascan ' in line or 'pdascan ' in line:
            settings['measurements'] += [line]
            continue
    return settings

def remove_comments(lines):
    new_lines = []
    for line in lines:
        if '#' in line:
            line = re.search('^.*\#', line).group()[:-1]
            if re.search(r'\w', line): #to drop lines that only have comments
                new_lines += [line]
        else:
            new_lines += [line] #I don't want to get rid of empty lines here
    return new_lines



def import_EC_data(full_path_name, title='get_from_file',
                 data_type='EC', N_blank=10, verbose=True,
                 header_string=None, timestamp=None, ):
    file_lines = import_text(full_path_name, verbose=verbose)
    dataset = text_to_data(file_lines, title='get_from_file',
                           data_type='EC', N_blank=10, verbose=True,
                           header_string=None, timestamp=None)
    numerize(dataset)
    return dataset

'''
The following couple functions are adapted from EC_MS on 17L09 
as last commited to EC_MS with code c1c6efa
They might benifit from a full rewrite, but not now.
'''


def import_text(full_path_name='current', verbose=True):   
    '''
    This method will import the full text of a file selected by user input as a 
    list of lines.
    When I first wrote it for EC_MS, way back in the day, I made it so you can 
    call it without any arguments, and then input. probably unecessary.
    '''    
    if verbose:
        print('\n\nfunction \'import_text\' at your service!\n')
    
    if full_path_name == 'input':
        full_path_name = input('Enter full path for the file name as \'directory' + os.sep + 'file.extension\'')
    if full_path_name == 'current':
        full_path_name = os.getcwd()

    [directory_name, file_name] = os.path.split(full_path_name)
    original_directory = os.getcwd()
    os.chdir(directory_name)
            
    if os.path.isdir(full_path_name) and not os.path.isfile(file_name):
        directory_name = full_path_name
        os.chdir(directory_name)
        ls_string = str(os.listdir())    
        print('\n' + full_path_name + '\n ls: \n' + ls_string + '\n')
        file_name = input('Directory given. Enter the full name of the file to import\n')

    if verbose:   
        print('directory: ' + directory_name)
        print('importing data from ' + file_name )

    possible_encodings = ['utf8','iso8859_15']  
    #mpt files seem to be the latter encoding, even though they refer to themselves as ascii
    for encoding_type in possible_encodings:    
        try:
            with codecs.open(file_name, 'r', encoding = encoding_type) as file_object:
                file_lines = file_object.readlines()
            if verbose:
                print('Was able to readlines() with encoding ' + encoding_type)
            break
        except UnicodeDecodeError:
            if verbose:
                print('Shit, some encoding problem in readlines() for ' + encoding_type)
    else:
        print('couldn\'t read ' + file_name + '\n ... may by due to an encoding issue')
        
    os.chdir(original_directory)
    
    if verbose:
        print('\nfunction \'import_text\' finished!\n\n')    
    return file_lines    
    
    
def text_to_data(file_lines, title='get_from_file',
                 data_type='EC', N_blank=10, verbose=True,
                 header_string=None, timestamp=None, ):
    '''
    This method will organize data in the lines of text from an EC or MS file 
    into a dictionary as follows (plus a few more keys)
    {'title':title, 'header':header, 'colheader1':[data1], 'colheader2':[data2]...}
    '''    
    if verbose:
        print('\n\nfunction \'text_to_data\' at your service!\n')
    
    #disect header
    N_lines = len(file_lines)        #number of header lines
    N_head = N_lines                 #this will change when I find the line that tells me how ling the header is
    header_string = ''              
        
    dataset = {}
    commacols = []                   #will catch if data is recorded with commas as decimals.
    
    loop = False
    
    for nl, line in enumerate(file_lines):
        
        if nl < N_head - 1:            #we're in the header
        
            if data_type == 'EC':
                if title == 'get_from_file':
                    if re.search('File :',line):
                        title_object = re.search(r'[\S]*\Z',line.strip())
                        title = title_object.group()
                        if verbose:
                            print('name \'' + title + '\' found in line ' + str(nl))
                if re.search(r'[Number ]*header lines',line):
                    N_head_object = re.search(r'[0-9][0-9]*',line)
                    N_head = int(N_head_object.group())
                    if verbose:
                        print('N_head \'' + str(N_head) + '\' found in line ' + str(nl))
                elif re.search('Acquisition started',line):
                    timestamp_object = re.search(r'[\S]*\Z',line.strip())
                    timestamp = timestamp_object.group()
                    if verbose:
                        print('timestamp \'' + timestamp + '\' found in line ' + str(nl))        
                elif re.search('Number of loops', line): #Then I want to add a loop number variable to data_cols
                    loop = True
                    dataset['loop number'] = []
                elif re.search('Loop', line):
                    n = int(re.search(r'^Loop \d+', line).group()[5:])
                    start = int(re.search(r'number \d+', line).group()[7:])
                    finish = int(re.search(r'to \d+', line).group()[3:])
                    N = finish - start + 1
                    dataset['loop number'] += N * [n]
                    
                header_string = header_string + line
                
            
        elif nl == N_head - 1:      #then it is the column-header line
               #(EC-lab includes the column-header line in header lines)
            #col_header_line = line
            col_headers = line.strip().split('\t')
            dataset['N_col']=len(col_headers) 
            dataset['data_cols'] = col_headers.copy()  #will store names of columns containing data
            #DataDict['data_cols'] = col_headers.copy()
            for col in col_headers:
                dataset[col] = []              #data will go here    
            header_string = header_string+line #include this line in the header
            if verbose:
                print('Data starting on line ' + str(N_head) + '\n')
            

        else:                   # data, baby!
            line_data = line.strip().split('\t')
            if not len(line_data) == len(col_headers):
                if verbose:
                    print(list(zip(col_headers,line_data)))
                    print('Mismatch between col_headers and data on line ' + str(nl) + ' of ' + title)
                if nl == N_lines - 1:
                    print('mismatch due to an incomplete last line of ' + title + '. I will discard the last line.')
                    break
            for col, data in zip(col_headers, line_data):
                if col in dataset['data_cols']:
                    try:
                        data = float(data)
                    except ValueError:
                        if data == '':
                            continue        #added 17C22 to deal with data acquisition crashes.
                        try:
                            if verbose and not col in commacols:
                                print('ValueError on value ' + data + ' in column ' + col + ' line ' + str(nl) + 
                                      '\n Checking if you''re using commas as decimals in that column... ')
                            data = data.replace('.','')        #in case there's also '.' as thousands separator, just get rid of it.
                            data = data.replace(',','.')       #put '.' as decimals
                            data = float(data)
                            if not col in commacols:
                                if verbose:
                                    print('... and you were, dumbass. I''ll fix it.')
                                commacols += [col]
                        except ValueError:
                            if verbose :
                                print(list(zip(col_headers,line_data)))
                                print(title + ' in text_to_data: \nRemoved \'' + str(col) +'\' from data columns because of value \'' + 
                                    str(data) + '\' at line ' + str(nl) +'\n')
                            dataset['data_cols'].remove(col)
                            
                dataset[col].append(data)
                    
    if loop:
        dataset['data_cols'] += ['loop number']        
    dataset['title'] = title
    dataset['header'] = header_string
    dataset['timestamp'] = timestamp
    dataset['data_type'] = data_type
    
    if data_type == 'EC':           #so that synchronize can combine current data from different EC-lab techniques
        if '<I>/mA' in dataset['data_cols'] and 'I/mA' not in dataset['data_cols']:
            dataset['data_cols'].append('I/mA')
            dataset['I/mA'] = dataset['<I>/mA']

    if verbose:
        print('\nfunction \'text_to_data\' finished!\n\n')    
    return dataset
