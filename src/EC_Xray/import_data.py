#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:17:41 2017

@author: scott
"""
import os, platform, re, codecs
import time, datetime, pytz  #all three seem necessary for dealing with timezones.
import numpy as np

float_match = '[-]?\d+[\.]?\d*(e[-]?\d+)?'     #matches floats like '-3.54e4' or '7' or '245.13' or '1e-15'
#note, no white space included on the ends! Seems to work fine.
timestamp_match = '([0-9]{2}:){2}[0-9]{2}'     #matches timestamps like '14:23:01'
date_match = '([0-9]{2}[/-]){2}[0-9]{4}'          #matches dates like '01/15/2018' or '09-07-2016'
# older EC-Lab seems to have dashes in date, and newer has slashes. 
# Both seem to save month before day, regardless of where the data was taken or .mpt exported.


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

def parse_timezone(tz=None):
    '''
    Gets a timezone object from a timezone string. Includes some abbreviations
    useful for Scott. If the input is not a string, it is returned as is.
    '''
    abbreviations = {'CA':'US/Pacific', 'DK':'Europe/Copenhagen',}
    if tz in abbreviations:
        return pytz.timezone(abbreviations[tz])
    elif type(tz) is str:
        return pytz.timezone(tz)
    else:
        return tz

def timestamp_to_epoch_time(timestamp, date='today', tz=None, verbose=True):
    '''
    Possibly overly idiot-proof way to convert a number of timestamps read
    from my data into epoch (unix) time. 
    
    tz is the Timezone, which is only strictly necessary when synchronizing
    data taken at different places or accross dst at a place with different dst 
    implementation than the local (dst is stupid!). 
    If timezone is a number, it is interpreted as the offset from GMT of the
    data. 
    
    The epoch time is referred to here and elsewhere as tstamp.
    '''
    if tz is not None:
        tz = parse_timezone(tz)
        if verbose:
            print('getting epoch time given a timestamp local to ' + str(tz))
        epoch = pytz.utc.localize(datetime.datetime.utcfromtimestamp(0))
    if timestamp == 'now':
        return time.time()
    elif type(timestamp) is time.struct_time:
        if verbose:
            print('\'timestamp_to_unix_time\' revieved a time.struct_time object. ' +
                  'Returning the corresponding epoch time.')
    elif type(timestamp) is not str:
        if verbose:
            print('timestamp_to_unix_time\' didn\'t receive a string. Returning' + 
                  ' the argument.')
        return timestamp
    if len(timestamp) > 8 and date=='today':
        if verbose:
            print('\'timestamp_to_unix_time\' is assuming' + 
                  ' the date and timestamp are input in the same string.')
        try:
            if tz is None:
                struct = time.strptime(timestamp)
                tstamp = time.mktime(struct)
            else: 
                dt_naive = datetime.datetime.strptime(timestamp, '%a %b %d %H:%M:%S %Y')
                dt = tz.localize(dt_naive)
                tstamp = (dt - epoch).total_seconds()
            if verbose:
                print('timestamp went straight into time.strptime()! Returning based on that.')
            return tstamp
        except ValueError:
            if verbose:
                print('bit \'' + timestamp + '\' is not formatted like ' + 
                      'time.strptime() likes it. Checking another format')
            try:
                date = re.search(date_match, timestamp).group()
                if verbose:
                    print('Matched the date with \'' + date_match + '\'.')
            except AttributeError:
                if verbose:
                    print('Couldn\'t match with \'' + date_match +
                          '\'. Assuming you want today.')
            try:
                timestamp = re.search(timestamp_match, timestamp).group()
                if verbose:
                    print('Matched the time with \'' + timestamp_match + '\'.' )
            except AttributeError:
                if verbose:
                    print('I got no clue what you\'re talking about, dude, ' +
                          'when you say ' + timestamp + '. It didn\'t match \.' +
                          timestamp_match + '\'. Assuming you want 00:00:00')
                timestamp = '00:00:00'            
    #h, m, s = (int(n) for n in timestamp.split(':'))
    #D, M, Y = (int(n) for n in re.split('[-/]', date))
    if date == 'today':
        date = time.strftime('%m/%d/%Y')
    if tz is None:
        try:
            struct = time.strptime(date + ' ' + timestamp, '%m/%d/%Y %H:%M:%S')
        except ValueError:
            struct = time.strptime(date + ' ' + timestamp, '%m-%d-%Y %H:%M:%S')            
        tstamp = time.mktime(struct)
    else:
        try:
            dt_naive = datetime.datetime.strptime(date + ' ' + timestamp, '%m/%d/%Y %H:%M:%S')
        except ValueError:
            dt_naive = datetime.datetime.strptime(date + ' ' + timestamp, '%m-%d-%Y %H:%M:%S')            
        dt = tz.localize(dt_naive)
        tstamp = (dt - epoch).total_seconds()
        
    return tstamp
    
def epoch_time_to_timestamp(tstamp, tz=None, verbose=True):
    '''
    tz is the Timezone, which is only strictly necessary when synchronizing
    data taken at different places or accross dst at a place with different dst 
    implementation than the local (dst is stupid!). 
    If timezone is a number, it is interpreted as the offset from GMT of the
    data in hours (+1 for Denmark, -8 for California)
    '''
    if tz is None:
        struct = time.localtime(tstamp)
    else:
        tz = parse_timezone(tz)
        if verbose:
            print('getting the timestamp local to ' + str(tz) + ' from epoch time.')
        dt_utc = datetime.datetime.utcfromtimestamp(tstamp)
        dt_tz = tz.fromutc(dt_utc)
        struct = dt_tz.timetuple()
    hh = str(struct.tm_hour)
    if len(hh) == 1:
        hh = '0' + hh
    mm = str(struct.tm_min)
    if len(hh) == 1:
        mm = '0' + mm
    ss = str(struct.tm_sec)
    if len(hh) == 1:
        ss = '0' + ss   
    timestamp = hh + ':' + mm + ':' + ss
    return timestamp
        
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
    combining.syncrhonize reads.
    The timestamp is local time, not absolute time.
    We need to move to epoch time everywhere!!!
    '''
    t = get_creation_time(filepath)
    struct = time.localtime(t)
    print('creation time sctructure = ' + str(struct)) # debugging
    hh = str(struct.tm_hour)
    try:
        mm = str(struct.tm_minute)
    except AttributeError:  # it seems to have changed from tm_minute to tm_min
        mm = str(struct.tm_min)
    try:
        ss = str(struct.tm_second)
    except AttributeError: # it seems to have changed from tm_second to tm_sec
        ss = str(struct.tm_sec)
    return hh + ':' + mm + ':' + ss
    

def get_creation_time(filepath, verbose=True):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        tstamp = os.path.getctime(filepath)
        if verbose:
            print('In Windows. Using os.path.getctime(\'' + filepath + '\') as tstamp.')
    else:
        stat = os.stat(filepath)
        try:
            tstamp = stat.st_birthtime
            if verbose:
                print('In linux. Using os.stat(\'' + filepath + '\').st_birthtime as tstamp.')
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            tstamp = stat.st_mtime    
            if verbose:
                print('Couldn\'t get creation time! Returing modified time.\n' + 
                  'In linux. Using os.stat(\'' + filepath + '\').st_mtime as tstamp.')
    return tstamp

def timestamp_from_file(filepath, verbose=True):
    a = re.search('[0-9]{2}h[0-9]{2}', filepath)
    if a is None:
        if verbose:
            print('trying to read creation time')
        timestamp = get_creation_timestamp(filepath)
    else:
        if verbose:
            print('getting timestamp from filename ' + filepath)
        timestamp = timetag_to_timestamp(filepath)
    return timestamp

def load_from_csv(filepath, multiset=False, tstamp=None, verbose=True):
    '''
    This function is made a bit more complicated by the fact that some csvs
    seem to have multiple datasets appended, with a new col_header line as the
    only indication. If multiset=True, this will separate them and return them
    as a list.
    if timestamp = None, the timestamp will be the date created
    I hate that SPEC doesn't save absolute time in a useful way.
    '''
    if verbose:
        print('function \'load_from_csv\' at your service!')
    if tstamp is None:
        #a = re.search('[0-9]{2}h[0-9]{2}', filepath)
        #if a is None:
        print('trying to read creation time')
        tstamp = get_creation_time(filepath)
        #else:
        #    print('getting timestamp from filename ' + filepath)
        #    timestamp = timetag_to_timestamp(filepath)
        #    tstamp = timestamp_to_epoch_time(timestamp)
            
    with open(filepath,'r') as f: # read the file!
        lines = f.readlines()
    colheaders = [col.strip() for col in lines[0].split(',')]
    data = get_empty_set(colheaders, title=filepath, 
                         tstamp=tstamp, data_type='SPEC')
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
                data = get_empty_set(colheaders, title=filepath.split(os.sep)[-1],
                         tstamp=tstamp, data_type='SPEC')
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
    if verbose:
        print('function \'load_from_csv\' finished!')
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
                 timestamp=None, date='today', tstamp=None, tz=None,
                 data_type='EC', N_blank=10, sep=None,
                 header_string=None, verbose=True):
    '''
    This method will organize data in the lines of text from a file useful for
    electropy into a dictionary as follows (plus a few more keys)
    {'title':title, 'header':header, 'timestamp':timestamp,
     'data_cols':[colheader1, colheader2, ...],
     colheader1:[data1], colheader2:[data2]...}
     So far made to work with SPEC files (.csv), XAS files (.dat), 
     EC_Lab files (.mpt).
     If you need to import cinfdata files (.txt), you are in the wrong place.
     This is EC_Xray's import_data, not EC_MS!!!
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

    if data_type == 'SPEC':
        N_head = 1 #the column headers are the first line
        if sep is None:
            sep = ','
    elif data_type == 'XAS':
        datacollines = False # column headers are on multiple lines, this will be True for those lines
        col_headers = []   #this is the most natural place to initiate the vector
    
    if sep is None:  #EC, XAS, and MS data all work with '\t'
        sep = '\t'
    
    quitloop = False
    for nl, line in enumerate(file_lines):
        l = line.strip()        
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
                elif timestamp is None and re.search('Acquisition started',line):
                    timestamp_object = re.search(timestamp_match,l)
                    timestamp = timestamp_object.group()
                    date_object = re.search(date_match, l)
                    date = date_object.group()
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


            elif data_type == 'XAS':
                if datacollines:
                    if l == '': #then we're done geting the column headers
                        datacollines = False
                        #header = False  #and ready for data.
                        N_head = nl + 1 # so that the next line is data!
                        dataset['data_cols'] = col_headers.copy()
                        if verbose:
                            print('data col lines finish on line' + str(nl) + 
                                  '. the next line should be data')
                    else:
                        col_headers += [l]    
                        dataset[l] = []
                elif l == 'Data:':  #then we're ready to get the column headers
                    datacollines = True
                    if verbose:
                        print('data col lines start on line ' + str(nl+1))
                a = re.search(timestamp_match, l)
                if timestamp is None and a is not None:
                    timestamp = a.group()
                    d = re.search(date_match, l)
                    if d is not None:
                        date = d.group()
                    tstamp = timestamp_to_epoch_time(l, tz=tz, verbose=verbose) #the XAS data is saved with time.ctime()
                    if verbose:
                        print('timestamp \'' + timestamp + '\' found in line ' + str(nl)) 
                header_string = header_string + line                
            
        elif nl == N_head - 1:      #then it is the column-header line
               #(EC-lab includes the column-header line in header lines)
            #col_header_line = line
            col_headers = [col.strip() for col in l.split(sep=sep)]
            dataset['N_col']=len(col_headers) 
            dataset['data_cols'] = col_headers.copy()  #will store names of columns containing data
            #DataDict['data_cols'] = col_headers.copy()
            for col in col_headers:
                dataset[col] = []              #data will go here    
            header_string = header_string + line #include this line in the header
            if verbose:
                print('Data starting on line ' + str(N_head) + '\n')
            

        else:                   # data, baby!
            line_data = [dat.strip() for dat in l.split(sep=sep)]
            if not len(line_data) == len(col_headers):
                if verbose:
                    print(list(zip(col_headers,line_data)))
                    print('Mismatch between col_headers and data on line ' + str(nl) + ' of ' + title)
                if nl == N_lines - 1 and data_type=='MS':    # this is usually not useful!
                    print('mismatch due to an incomplete last line of ' + title + '. I will discard the last line.')
                    break
            data_cols = dataset['data_cols'].copy()
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
                            data = data.replace('.','')        
                            # ^ in case there's also '.' as thousands separator, just get rid of it.
                            data = data.replace(',','.')       #put '.' as decimals
                            data = float(data)
                            if not col in commacols:
                                if verbose:
                                    print('... and you were, dumbass. I''ll fix it.')
                                commacols += [col]
                        except ValueError:
                            if verbose :
                                #print(list(zip(col_headers,line_data))) # debugging
                                print(title + ' in text_to_data: \nRemoved \'' + str(col) +
                                      '\' from data columns because of value \'' + 
                                    str(data) + '\' at line ' + str(nl) +'\n')
                            dataset['data_cols'].remove(col)
                            #print(dataset['data_cols']) # debugging
                            if len(dataset['data_cols']) == 0:
                                if verbose:
                                    print('there\'s a full line of value errors. Must me something wrong. ' +
                                          'Terminating loop.')
                                    dataset['data_cols'] = data_cols
                                    quitloop = True
                dataset[col].append(data)
        if quitloop:
            print('removing last line from datacols and quitting loop')
            for col in dataset['data_cols']:
                dataset[col] = dataset[col][:-1]
            break
                    
    if loop:
        dataset['data_cols'] += ['loop number']        
    dataset['title'] = title
    dataset['header'] = header_string
    dataset['timestamp'] = timestamp
    dataset['date'] = date
    if tstamp is None:
        tstamp = timestamp_to_epoch_time(timestamp, date, tz=tz, verbose=verbose)
    dataset['timezone'] = tz
    dataset['tstamp'] = tstamp 
    #UNIX epoch time, for proper synchronization!
    dataset['data_type'] = data_type
    
    if data_type == 'EC':           #so that synchronize can combine current data from different EC-lab techniques
        if '<I>/mA' in dataset['data_cols'] and 'I/mA' not in dataset['data_cols']:
            dataset['data_cols'].append('I/mA')
            dataset['I/mA'] = dataset['<I>/mA']

    if verbose:
        print('\nfunction \'text_to_data\' finished!\n\n')    
    return dataset



def load_from_file(full_path_name='current', title='file', tstamp=None, timestamp=None,
                 data_type='EC', N_blank=10, tz=None, verbose=True):
    '''
    This method will organize the data in a file useful for
    electropy into a dictionary as follows (plus a few more keys)
    {'title':title, 'header':header, 'timestamp':timestamp,
    'data_cols':[colheader1, colheader2, ...],
    colheader1:[data1], colheader2:[data2]...}
    So far made to work with SPEC files (.csv), XAS files (.dat), 
    EC_Lab files (.mpt).
    If you need to import cinfdata files (.txt), you are in the wrong place.
    Use EC_MS's import_data instead!!!
    '''
    if verbose:
        print('\n\nfunction \'load_from_file\' at your service!\n')
    if title == 'file':
        folder, title = os.path.split(full_path_name)
    file_lines = import_text(full_path_name, verbose)
    dataset = text_to_data(file_lines=file_lines, title=title, data_type=data_type, 
                           timestamp=timestamp, N_blank=N_blank, tz=tz, tstamp=tstamp,
                           verbose=verbose)
    if tstamp is not None: #then it overrides whatever test_to_data came up with.
        dataset['tstamp'] = tstamp
    elif dataset['tstamp'] is None:
        dataset['tstamp'] = get_creation_time(full_path_name, verbose=verbose)
    numerize(dataset)

    if verbose:
        print('\nfunction \'load_from_file\' finished!\n\n')
    return dataset 
    
    
def load_EC_set(directory, EC_file=None, tag='01', 
                  verbose=True, tz=None, force_sort=False): 
    if verbose:
        print('\n\nfunction \'load_EC_set\' at your service!\n')
    from .combining import synchronize, sort_time
    
    lslist = os.listdir(directory)
    
    if EC_file is None:
        EC_file = [f for f in lslist if f[:2] == tag and f[-4:] == '.mpt']
    elif type(EC_file) is str:
        EC_file = [EC_file]
    EC_datas = []
    for f in EC_file:
        try:
            EC_datas += [load_from_file(directory + os.sep + f, data_type='EC', tz=tz, verbose=verbose)]
        except OSError:
            print('problem with ' + f + '. Continuing.')
    EC_data = synchronize(EC_datas, verbose=verbose, append=True, t_zero='first', tz=tz)
    if 'loop number' in EC_data['data_cols'] or force_sort:
        sort_time(EC_data, verbose=verbose) #note, sort_time no longer returns!
        
    if verbose:
         print('\nfunction \'load_EC_set\' finished!\n\n')       
    return EC_data
