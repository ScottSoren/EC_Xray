#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:17:41 2017

@author: scott
"""
import os, re


def numerize(data):
    for col in data['data cols']: #numerize!
        data[col] = np.array(data[col])    


def load_from_csv(filepath, multiset=False):
    
    f = open(filepath,'r') # read the file!
    lines = f.readlines()
    colheaders = [col.strip() for col in lines[0].split(',')]
    data = get_empty_set(colheaders, title=filepath)
    datasets = []
    
    for line in lines[1:]: #put data in lists!
        vals = [val.strip() for val in line.split(',')]
        not_data = []
        newline = {}
        for col, val in zip(colheaders, vals):
            if col in data['data cols']:
                try:
                    val = float(val)
                except ValueError:
                    print('value ' + val + ' of col ' + col + ' is not data.')
                    not_data += [col]   
            newline[col] = val
        if len(not_data) == len(data['data cols']):
            print('it looks like there is another data set appended!')
            if multiset:
                print('continuing to next set.')
                numerize(data)
                datasets += [data.copy()]
                colheaders = [val.strip() for val in vals]
                data = get_empty_set(colheaders)
                continue
            else:
                print('returning first set.')
                numerize(data)
                return data
        else:    
            for col in not_data:
                data['data cols'].remove(col)
                print('column ' + col + ' removed from \'data cols \'.')

        for col, val in zip(colheaders, vals):
            data[col] += [newline[col]]
    
    numerize(data)
    datasets += [data]
    if multiset:
        return datasets
    return data



'''
The following several functions are copied from EC_MS.Data_Importing on 17L09.
They could probably use a serious rewrite. 
'''


def import_text(full_path_name='current', verbose=1):   
    '''
    This method will import the full text of a file selected by user input as a 
    list of lines
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
            file_object = codecs.open(file_name, 'r', encoding = encoding_type)
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


def download_data(IDs='today', 
                  timestamps=None,
                  data_type='fullscan',
                  timestamp_interval=None,
                  comment=None,
                  connect={},
                  verbose=True, 
                  ):
        '''
        Returns data columns matching a certain set of ID's.
        So far
        '''
        
        connect_0 = dict(host='servcinf-sql',  # your host, usually localhost, servcinf would also work, but is slower (IPv6)
                               #    port=9995,  # your forwording port
                                   user='cinf_reader',  # your username
                                   passwd='cinf_reader',  # your password
                                   db='cinfdata')  # name of the data base
        
        for key, val in connect_0.items():
            if key not in connect:
                connect[key] = val
        
        if data_type == 'fullscan':
            data_string_template = 'SELECT x,y FROM xy_values_sniffer where measurement = {0} order by id'
            
        
 #       try:
        print('Connecting to CINF database...')
        cnxn =  MySQLdb.connect(**connect)
        cursor = cnxn.cursor()
        print('Connection successful!')
  #      except:
   #         print('Connection failed!')
        
        if type(IDs) is int:
            IDs = [IDs]
        
        datasets = {}
        for ID in IDs:
            data_string = data_string_template.format(str(ID))
            cursor.execute(data_string)
            raw_data = cursor.fetchall()
            list_data = np.array(raw_data)
            xy_data = np.swapaxes(list_data, 0, 1)
            datasets[ID] = xy_data
        
        return datasets
        
            
            

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
    n_blank = 0                      #header ends with a number of blank lines in MS
        
    DataDict = {}
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
                    DataDict['loop number'] = []
                elif re.search('Loop', line):
                    n = int(re.search(r'^Loop \d+', line).group()[5:])
                    start = int(re.search(r'number \d+', line).group()[7:])
                    finish = int(re.search(r'to \d+', line).group()[3:])
                    N = finish - start + 1
                    DataDict['loop number'] += N * [n]
                    
                header_string = header_string + line
            
            elif data_type == 'MS':
                if len(line.strip())==0:
                    n_blank += 1
                    if n_blank>N_blank and len(file_lines[nl+1].strip())>0:
                        N_head = nl+2
                else:
                    n_blank = 0    
                    if title == 'get_from_file':
                        object1 = re.search(r'"Comment"[\s]*"[^"]*',line)
                        if object1:
                            string1 = object1.group()
                            title_object = re.search(r'[\S]*\Z',string1.strip())
                            title = title_object.group()[1:]
                            if verbose:
                                print('name \'' + title + '\' found in line ' + str(nl))
                    object2 = re.search(r'"Recorded at"[\s]*"[^"]*',line)
                    if object2:
                        string2 = object2.group()
                        timestamp_object = re.search(r'[\S]*\Z',string2.strip()) 
                        timestamp = timestamp_object.group()
                        if verbose:
                            print('timestamp \'' + timestamp + '\' found in line ' + str(nl))                 
                header_string = header_string + line
                
            
        elif nl == N_head - 1:      #then it is the column-header line
               #(EC-lab includes the column-header line in header lines)
            #col_header_line = line
            col_headers = line.strip().split('\t')
            DataDict['N_col']=len(col_headers) 
            DataDict['data_cols'] = deepcopy(col_headers)   #do we need deepcopy? #will store names of columns containing data
            #DataDict['data_cols'] = col_headers.copy()
            for col in col_headers:
                DataDict[col] = []              #data will go here    
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
            for col, data in zip(col_headers,line_data):
                if col in DataDict['data_cols']:
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
                            DataDict['data_cols'].remove(col)
                            
                DataDict[col].append(data)
                    
    if loop:
        DataDict['data_cols'] += ['loop number']        
    DataDict['title'] = title
    DataDict['header'] = header_string
    DataDict['timestamp'] = timestamp
    DataDict['data_type'] = data_type
    
    if data_type == 'EC':           #so that synchronize can combine current data from different EC-lab techniques
        if '<I>/mA' in DataDict['data_cols'] and 'I/mA' not in DataDict['data_cols']:
            DataDict['data_cols'].append('I/mA')
            DataDict['I/mA'] = DataDict['<I>/mA']

    if verbose:
        print('\nfunction \'text_to_data\' finished!\n\n')    
    return DataDict
