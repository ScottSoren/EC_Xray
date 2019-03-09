#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 00:54:57 2017
@author: scott
"""
import os
import re
import numpy as np
#import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import time
import pickle

try:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
except ImportError:
    print('you need the package moviepy to be able to make movies!')

from .import_data import (load_from_file, read_macro,
                        epoch_time_to_timestamp, timestamp_to_epoch_time)
from .pilatus import Pilatus, calibration_0, shape_0
from .XRD import integrate_peak, get_background_line, get_peak_background
from .XRD import Peak

timestamp_matcher = '([0-9]{2}\:){2}[0-9]{2}'

def get_images(directory, tag, shape=shape_0, calibration=calibration_0, 
               slits=True, xslits=None, yslits=[60, 430], 
               pixelmax=None, verbose=True,  vverbose=False):
    if verbose:
        print('\n\nfunction \'get_images\' at your service!\n')
    try:
        lslist = os.listdir(directory)
    except FileNotFoundError:
        print('The directory doesn\'t exist. get_images is returning a blank dictionary.')
        return {}
    #print(tag) # debugging
    if verbose:
        print(str(len(lslist)) + ' items in ' + directory)
    imagenames = [f for f in lslist if f[-4:]=='.raw' and tag in f]
    if verbose:
        print(' of which ' + str(len(imagenames)) + ' are image files including \'' + tag + '\'')
    images = {}  
    for f in imagenames:
        n = int(f[-8:-4])  #this is the image number as SPEC saves them
        filepath = directory + os.sep + f
        images[n] = Pilatus(filepath, shape=shape, calibration=calibration, 
                            slits=slits, xslits=xslits, yslits=yslits, 
                            pixelmax=pixelmax, verbose=vverbose)
    if verbose:
        print('\nfunction \'get_images\' finished!\n\n')
    return images

def peak_colors(peak_list, colors=['k', 'b', 'r', 'g', 'c', 'm']):
    '''
    This is a fill-in function until I've got some kind of standard colors
    implemented. It takes a list of integral ranges and returns an identically
    indexed dictionary with each value of the form (integral_range, color)
    '''
    integrals = {}
    for i, integral in enumerate(peak_list):
        integrals[i] = (integral, colors[i])
    return integrals


def get_direction_mask(x, direction=True):
    '''
    Returns a mask selecting the values of x that are greater than (direction
    = True) or less than (direction = False) all previous values
    '''
    if type(direction) in [int, float]:
        direction = direction > 0
    mask = []
    X = x[0]
    for x_i in x:
        mask += [(x_i > X) == direction]
        if mask[-1]:
            X = x_i
    return np.array(mask)


# ----------------- here comes the CLASS -----------------


class ScanImages:

    #-------------- functions for defining the scan ----------    
    def __init__(self, name=None, csvfile=None, directory=None, 
                 pilatusfilebase='default', usecsv=True,
                 tag=None, scan_type='time', calibration=calibration_0, 
                 macro=None, tth=None, alpha=None, timestamp=None, tstamp=None, 
                 pixelmax=None, timecol=None, abstimecol=None, tz=None,
                 slits=True, xslits=None, yslits=[60, 430], 
                 scan=None, copy=False, load=False,
                 verbose=True, vverbose=False):
        '''
        give EITHER a csvfile name with full path, or a directory and a tag.
        pilatusfilebase can be constructed from this, and used to import the
        Pilatus image objects.
        The calibration is passed on to the Pilatus objects.
        The macro is read to get the (tth, alpha) values which aren't scanned,
        though they can also be put in manually.os.path.expanduser('~/o/FYSIK/list-SurfCat/setups/Synchrotron/May2018')
        timestamp can be either a str like 'hh:mm:ss' or a pointer.
        timestamp='abstimecol' uses the first value of the specified timecol in the csvfile
        timestamp=None tries to get it from the file
        '''

        # ------- load a pickle, to save computing time and space -------- #
        if load:
            try:
                with open(name) as f:
                    scan = pickle.load(f)
            except FileNotFoundError:
                print('Couldn\'t find ' + name)
                loadname = name + '.pckl'
                print('Trying ' + loadname + '.')
                with open(loadname, 'rb') as f:
                    scan = pickle.load(f)  
            print('Loaded ' + name)

        # ------ for development: new code with pre-loaded data -------#
        if copy or load: # take all data (images, csv_data, etc) from another scan
            for attr in dir(scan):
                if attr not in dir(self): # because we don't want to replace e.g. functions
                    setattr(self, attr, getattr(scan, attr))
            try:
                self.copied += 1
            except AttributeError: 
                self.copied = 1
            return
        
        # ---- parse inputs for name and, if csv used, csvname -----------------#
        csvname = None
        if usecsv:
            if csvfile is None:
                csv_directory = directory
                #print(load) # debugging
                if (tag is None or directory is None) and not load:
                    print('need a csv file name or a directory and a tag!')
                    return
                lslist = os.listdir(directory)
                try:
                    csvname = next(f for f in lslist if f[-4:]=='.csv' and '_scan' in f and tag in f)
                except StopIteration:
                    if load:
                        pass
                    else:
                        print(lslist)
                        print('Cound not find a csvname containing ' + tag + ' in ' + directory + '\n(ls above)')  
            else:
                csv_directory, csvname = os.path.split(csvfile)
                if len(csv_directory) == 0:
                    csv_directory = directory
            if csvname is not None:
                print('Loading Scan from directory = ' + directory +
                      '\n found csvname = ' + str(csvname))   

        if name is None:
            if tag is not None:
                name = tag
            elif csvname is not None:
                name = csvname[:-4] # to drop the .csv
            elif csvfile is not None:
                name = csvfile
        
        print('scan name = \'' + name + '\'')
        
        
        # --------------  install easy metadata ------------- #
        self.directory = directory
        self.name = name
        self.timecol = timecol
        self.abstimecol = abstimecol
        self.tz = tz        
        self.bg = False # stores whether background has been subtracted

        self.verbose = verbose
        self.vverbose = vverbose
        
        if scan_type in ['time', 't']:
            self.scan_type = 't'
        elif scan_type in ['tth', 'TwoTheta']:
            self.scan_type = 'tth'
        elif scan_type in ['alpha', 'a', 'th', 'Theta']:
            self.scan_type = 'alpha'
        
        if macro is not None:
            self.macro = macro
            self.settings = read_macro(macro)
            if tth is None:
                tth = self.settings['tth'][-1]
            if alpha is None:
                alpha = self.settings['alpha'][-1]
                
        self.tth = tth
        self.alpha = alpha

        #try to read stuff from file name
        if csvname is not None:
            for match in re.findall('_[A-Za-z]+[n?][0-9]+[p[0-9]+]?', csvname):
                attr = re.search('[A-Za-z]', match).group()
                value = re.search('[0-9]+[n]?[p[0-9]+]?', match).group()
                try:
                    value = float(value.replace('p','.').replace('n','-'))
                except ValueError:
                    print('not sure what ' + value + ' is.')
                if not hasattr(self, attr):
                    setattr(self, attr, value)
                elif getattr(self, value) is None:
                    setattr(self, attr, value)   


        # --------- import csv if requested
        if csvname is not None:
            csvfilepath = csv_directory + os.sep + csvname
            self.csv_data = load_from_file(csvfilepath, data_type='SPEC', 
                                           timestamp=timestamp, tstamp=tstamp, tz=tz)
            self.csvfilepath = csvfilepath  
   
        
        #-------------------- get images! ------------------------#
        if pilatusfilebase == 'default':
            for foldername in ['images', 'Pilatus']:
                pilatus_directory = directory + os.sep + foldername
                if os.path.isdir(pilatus_directory):
                    break
            else:
                print('could not find pilatus directory!')
            tag_pilatus = name
        else:
            pilatus_directory, tag_pilatus = os.path.split(pilatusfilebase)
        self.images = get_images(pilatus_directory, tag=tag_pilatus, 
                                 calibration=calibration, verbose=verbose,
                                 pixelmax=pixelmax,
                                 slits=slits, xslits=xslits, yslits=yslits, 
                                 vverbose=vverbose)
        
        if len(self.images) == 0:
            raise Warning('THIS SCAN IS EMPTY!!!!')
            self.empty = True
            return
        else:
            self.empty = False        


        # ------------------------- organize csvdata and metadata ---------- #        
        
        if hasattr(self, 'csv_data'):
            self.data = self.csv_data.copy()
            #self.csv_into_images()  # this causes problems now. csv is more likely to be currupt than images.
        else:
            self.data = {'title':name, 'data_type':'spec'}
            self.data['data_cols'] = []
        
        for col, attr in [('tth_scan', 'tth'), ('alpha','alpha'), ('t_abs', 'tstamp')]:
            try:
                self.data[col] = np.array([getattr(self.images[i], attr) for i in range(len(self))])
                self.data['data_cols'] += [col]
                if verbose:
                    print('got \'' + attr + '\' from Pilatus objects' + 
                          ' and saved it as self.data[\'' + col + '\']')
            except AttributeError:
                if verbose:
                    print('could not get ' + col + ', (' + attr + ') from images.')
                
        # this will conveniently store useful data, some from csv_data

        if timecol is None and abstimecol is not None:
            # put in the timecol!
            self.get_timecol_from_abstimecol()


        # ---------------------- get timestamp and timecol -----------------#
        if verbose:
            print('\nGetting tstamp and t according to inputs:\n\t' +
                  'timestamp = ' + str(timestamp) + ', tstamp = ' + str(tstamp))
        
        if timestamp in ['filename', 'csv','file']:
            tstamp = self.csv_data['tstamp']
            timestamp = epoch_time_to_timestamp(tstamp, tz=tz)
            if verbose:
                print('got self.tstamp from self.csv_data')
        elif timestamp in ['pdi']:
            tstamp = self.images[0].tstamp
            if verbose:
                print('got self.tstamp from self.images[0]')
        elif timestamp in ['abstimecol']:
            try:
                value = self.csv_data[abstimecol][0]
                try:
                    a = re.search(timestamp_matcher, value)
                except TypeError:
                    print('ERROR: You\'re trying to get the timestamp from an absolute' +
                          ' time column.\n Inputs:\ttimestamp=\'abstimecol\',\tabstimecol=\'' + 
                          str(abstimecol) + '\'\n but self.csv_data[abstimecol] = ' + str(value) + '.')
                    raise
                timestamp = a.group()
                tstamp = timestamp_to_epoch_time(value, tz=tz)
                #print('line 163: timestamp = ' + timestamp) # debugging
                if timecol is not None:
                    t = a.csv_data[timecol]
                    tstamp = tstamp - t[0]
                    #this is to correct for the fact that tstamp refers to the
                    timestamp = epoch_time_to_timestamp(tstamp, tz=tz)
                    #first datapoint
                if verbose:
                    print('got self.tstamp from self.csv_data[\'' + abstimecol + '\'], i.e., abstimecol.')
            except OSError: # a dummy error... I want to actually get the error messages at first
                pass
        elif 'tstamp' in self.csv_data:
            tstamp = self.csv_data['tstamp']
            print('got tstamp from self.csv_data')

        if 't' not in self.data:
            if timecol is not None:
                print('getting t from self.csv_data[\'' + timecol + '\'].')
                t = self.csv_data[timecol]
            elif 't_abs' in self.data:
                tstamp = self.data['t_abs'][0]
                t = self.data['t_abs'] - tstamp
                if verbose:
                    print('got self.tstamp and self.t from self.data[\'t_abs\']')
            else:
                try:
                    t = self.csv_data[self.timecol]
                    if verbose:
                        print('got self.t from self.csv_data[\'' + 
                               self.timecol + '\'], i.e. timecol.')
                except KeyError:
                    if self.timecol is not None:
                        print('self.timecol = ' + str(self.timecol) + 
                              ' is not in csv data. Check yo self.')
                    else:
                        print('This is a timescan but there\'s no time ' +
                              'variable specified. \nConsider using ' +
                              'EC_Xray.time_cal() to calibrate and specify one.')
                    return
            # we can only reach here if 't' has been successfully put into self.data_cols

        self.tstamp = tstamp
        self.timestamp = timestamp
        self.data['tstamp'] = tstamp
        self.data['timestamp'] = timestamp
        self.data['t'] = t
        if 't' not in self.data['data_cols']:
            self.data['data_cols'] += ['t']    
        #print('line 170: self.timestamp = ' + str(self.timestamp))
        # This code is a bit of a mess, and timestamp is here only for sanity-checking
        #purposes. All math will refer to tstamp
        
        #------- finished ---------- #
        if self.verbose:
            print('\nScanImages object with name ' + self.name + 
                  ' imported!\n\n')        

    
    def __len__(self):
        try:
            return len(self.images)
        except AttributeError:
            print('len(self) is tricky for scan named \'' + self.name + 
                  '\' which was loaded without images. Will try to use ' + 
                  'len(self.data[\'t\']) instead')
            try:
                return(len(self.data['t']))
            except AttributeError:
                print('There is no self.data')
            except KeyError:
                print('self.data has no t.')
        return None
    
    def __getitem__(self, indices):
        if type(indices) is int:
            if type(indices) is int and indices<0:
                indices = len(self) + indices
            return self.images[indices]
        elif type(indices) in [list, tuple]:
            return [self.images[i] for i in indices]
        print('indices must be an integer or sequence of integers')    
    
    def save(self, filename=None, with_images=False):
        savescan = ScanImages(copy=True, scan=self)
        if not with_images and hasattr(savescan, 'images'):
            del(savescan.images)
        if filename is None:
            filename = './' + self.name + '.pckl'
        with open(filename, 'wb') as f:
            pickle.dump(savescan, f)
        
    
    def append(self, scan):
        N = len(self)
        for n in range(len(scan)):
            self.images[N+n] = scan.images[n]
        for col, attr in [('tth_scan', 'tth'), ('alpha','alpha'), ('t_abs', 'tstamp')]:
            self.data[col] = np.array([getattr(self.images[i], attr) for i in range(len(self))])
            self.data['data_cols'] += [col]
        tstamp = self.data['t_abs'][0]
        t = self.data['t_abs'] - tstamp
        self.data['t'] = t
        
    
    def csv_into_images(self):
        if self.scan_type == 't':
            for i in range(len(self)):               
                # I don't like it, but that's where SPEC saves t.
                # data columns 'TTIMER' and 'Seconds' contain nothing.
                # If t is recorded, tth and alpha are constant, but...
                # The tth and alpha are not saved anywhere. The user must 
                # input them, or input a macro to read. Done in self.__init__
                #print('putting tth=' + str(self.tth) + ' into image!') #debugging
                if self.images[i].tth is None:
                    self.images[i].tth = self.tth
                if self.images[i].alpha is None:
                    self.images[i].alpha = self.alpha
        elif self.scan_type == 'tth':
            self.data['tth_scan'] = self.csv_data['TwoTheta']
            self.data['data_cols'] += ['tth_scan']
            for i in range(len(self)):
                #self.data['tth'] will be saved for when calculating the spectrum from the images
                self.images[i].tth = self.data['tth_scan'][i]
                self.images[i].alpha = self.alpha
        elif self.scan_type == 'alpha':
            for i in range(len(self)):
                self.images[i].tth = self.tth
    
    def set_tth(self, tth, update_images=True):
        self.tth = tth
        if update_images:
            for image in self.images.values():
                image.tth = tth
    
    def get_timecol_from_abstimecol(self):
        abstimecol = self.csv_data[self.abstimecol]
        t = []
        #print('line 228: self.timestamp = ' + str(self.timestamp))
        t0 = self.tstamp
        for timecol in abstimecol:
            t += [timestamp_to_epoch_time(time, tz=self.tz) - t0]
        self.data['t'] = t
        if 't' not in self.data['data_cols']:
            self.data['data_cols'] += ['t']

 
    
    #-------------- functions for calculating XRD spectra ----------
    
    def get_combined_spectrum(self, stepsize=0.05, override=False,
                              slits=True, xslits=None, yslits=None,
                              method='sum', min_pixels=10, tth=None,
                              scan_method='sum', out='spectrum',
                              weight=None,
                              recalculate=False, normalize=None):
        '''
        Calculates conventional tth spectrum (diffractogram) from the pixels
        of each Pilatus image. If the image spectra have already been
        calculated, they are used unless override is True.
        scan_method says whether to add ('sum') or average ('average') the
        contributions from each image.
        
        stepsize, method, min_pixels, xslits, yslits, and weight are
        all arguments which are passed on to Pilatus.tth_spectrum()
        '''
        
        if self.verbose:
            print('\n\nfunction \'get_combined_spectrum\' at your service!\n')
        
        if hasattr(self, 'spectrum') and not recalculate:
            return self.spectrum
            
        
        elif not hasattr(self, 'images'):
            print('scan \'' + self.name + '\' has no images! Can\'t calcualte spectrum')
            return
        
        if self.verbose:
            t0 = time.time()
            print('t = 0')
            print('calculating tth spectrum for each of ' + str(len(self)) + 
                  ' images, storing in Pilatus objects, and adding them all up.')

        if tth is not None:
            self.tth = tthTru
        if normalize:
            try:
                normalizer = self.data[normalize]
                if self.verbose:
                    print('normalizing spectra according to self.data[\'' +
                           normalize + '\'].')
            except KeyError:
                normalize = False
                raise Warning('normalize must be a key to self.data won\'t normalize')
        
        bins = {}
        contributors = {}
        raw_spectra = {}
        for i in range(len(self)):
            bins_i = self.images[i].tth_spectrum(out='bins', override=override,
                                                stepsize=stepsize, method=method,
                                                min_pixels=min_pixels, tth=tth,
                                                xslits=xslits, yslits=yslits,
                                                weight=weight,
                                                verbose=self.vverbose)
            raw_spectra[i] = self.images[i].spectrum
            if normalize:
                try:
                    norm = normalizer[i]
                    if type(norm) not in [int, float, np.float64]:
                        raise IndexError
                except IndexError:
                    print('encountered a problem in normalizer for image #' +
                          str(i) + '. Terminating.')
                    break
            else:
                norm = 1
            for n, counts in bins_i.items():
                if type(n) is not int:
                    continue
                if n in bins:
                    bins[n] += counts / norm
                    contributors[n] += [i]
                else:
                    bins[n] = counts / norm
                    contributors[n] = [i]
        if self.verbose:
            print('Counts per tth interval calculated locally and globally. ')
        tth_vec = []
        counts_vec = []
        n_min = min(bins.keys())
        n_max = max(bins.keys())
        for n in range(n_min, n_max+1):
            tth_vec += [(n + 0.5) * stepsize]
            if scan_method == 'average':
                counts_vec += [bins[n]/len(contributors[n])]
            else:
                counts_vec += [bins[n]]
                
        tth_vec = np.array(tth_vec)
        counts_vec = np.array(counts_vec)
        spectrum = np.stack([tth_vec, counts_vec], axis=0)
        N_contributors = np.array([len(contributors[i]) for i in bins.keys()])
        
        self.method = method
        self.scan_method = scan_method
        self.contributors = contributors
        self.N_contributors = N_contributors
        self.bins = bins
        self.raw_spectra = raw_spectra
        self.spectrum = spectrum
        self.data.update({'tth':tth_vec, 'counts':counts_vec})
        if 'counts' not in self.data['data_cols']:
            self.data['data_cols'] += ['counts']
        
        if self.verbose:
            print('Converted to global tth spectrum and stored in ScanImages oject.')
            print('t = ' + str(time.time() - t0) + ' seconds.')
            print('\nfunction \'get_combined_spectrum\' finished!\n\n')   

        if out == 'spectrum':
            return spectrum
        elif out == 'bins':
            return bins

    
    def get_stacked_spectra(self, stepsize=0.05, override=None,
                          slits=True, xslits=None, yslits=None, weight=None,
                          method='average', min_pixels=10, tth=None,
                          normalize=None):
        if self.verbose:
            print('\n\nfunction \'get_stacked_spectra\' at your service!\n')
            
        if override is False and hasattr(self, 'spectra') and self.spectra is not None:
            if hasattr(self, 'spectrab'):
                spectra = self.spectrab
            else:
                spectra = self.spectra
            if self.verbose:
                print('using the already-calculated image spectra')       
            return spectra
        
        if not override and hasattr(self, 'spectrum') and self.spectrum is not None:
            combined_spectrum = self.spectrum
            if self.verbose:
                print('using the already-calculated spectrum for each image')
        else:
            combined_spectrum = self.get_combined_spectrum(out='spectrum',
                                                           stepsize=stepsize, 
                                                           method=method, tth=tth,
                                                           min_pixels=min_pixels, 
                                                           normalize=normalize,
                                                           xslits=xslits, yslits=yslits,
                                                           weight=weight) 
        #this generates all the images' spectra, so they're saved when called later.
        tth_vec = combined_spectrum[0]       
        spectrums = []        # collection of the individual spectrum from each image
        if normalize:
            try:
                normalizer = self.data[normalize]
                if self.verbose:
                    print('normalizing spectra according to self.data[\'' +
                           normalize + '\'].')
            except KeyError:
                normalize = False
                raise Warning('normalize must be a key to self.data won\'t normalize')
                
        for i in range(len(self)):
            tth_i, counts_i = self.raw_spectra[i] 
            if normalize:
                norm = normalizer[i]
            else:
                norm = 1
            #spectra were generated during call to self.get_combined_spectrum
            spectrums += [np.interp(tth_vec, tth_i, counts_i, left=0, right=0) / norm]
            #print(norm) # debugging
            #spectra += [interp_with_zeros(tth_vec, tth_i, counts_i)] #may as well use numpy (above)       

        spectra = np.stack(spectrums, axis=0) #a 2-d spectrum space for the scan
        self.spectra = spectra
        if self.verbose:
            print('self.spectra.shape = ' + str(np.shape(spectra)))
            print('\nfunction \'get_stacked_spectra\' finished!\n\n')
        return spectra
    
    
    
    def slim(self):
        '''
        deletes image maps to save RAM space.
        '''
        try:
            for i, im in self.images.items():
                if hasattr(im, 'map_xyz'):
                    del(im.map_xyz)
                if hasattr(im, 'map_xyz_prime'):
                    del(im.map_xyz_prime)
                if hasattr(im, 'map_tth'):
                    del(im.map_tth)
                if hasattr(im, 'map_bin'):
                    del(im.map_bin)
        except AttributeError: # if there's no images, job is already done.
            pass
    
    
    def subtract_background(self, background='endpoint', background_type='local',
                            show=None, **kwargs):
        '''
        Generates background-subtracted tth spectrum and image tth spectra,
        to be saved as self.spectrumb and self.spectrab, respectively.
        
        background can be:
            'constant': subtracts the minimum non-zero value from
        nonzero values
            'linear': subtracts a linear interpolation. The interpolation 
        is a linear fit that is readjusted iteratively with outliers removed 
        until there are no outliers, defined by significance p. This will 
        not really work if peaks are large compared to the relevant tth range. 
            'endpoint': subtracts a simple line connecting endpoints. N_end
        points are included in each endpoint, and endpoitns are moved inwards
        if it looks (significance given by p) like an endpoint is on a peak.
            A 1D array with length equal to self.spectra.shape[0]: 
        background is simply subtracted from each image spectrum
        in spectra.
            A 2D array with shape[0]=2 or list or tuple of two 1D arrays: 
        interpreted as a tth spectrum. This spectrum is subtracted by 
        interpolation. background_type be specified as 'global' or 'local'.
            An integer or float: interpretation depends on background_type. if
        background_type is 'global' or 'local', a constant value equal to
        background is subtracted.
        
        background_type can be:
            'index': the spectra[background] is subtracted from all the spectra
            a string corresponding to a column in self.data: The interpolated
        spectrum corresponding to self.data[background_type] = background is
        subtracted from all the spectra. Used to subtract, for example, the
        spectruma at a given time (background_type='t' or electrochemical 
        potential (backgrouhd_type = 'U vs RHE / [V]').
            'global': background subtraction is done directly for spectrum,
        indirectly for spectra
            'local': background subtraction is done directly for spectra,
        indirectly for spectrum.
        
        Additional keward arguments are fed to get_background_line()
        
        
        For tth scans, a background-subtracted total spectrum should perhaps
        be calculated instead... this might not be implemented yet.
        '''
        
        # ---------- get spectra and spectrum ----------#
        if self.verbose:
            print('\n\nfunction \'subtract_background\' at your service!\n')
        
        from .combining import get_timecol
        #get spectra and spectrum
        try:
            spectrum = self.spectrum
        except AttributeError:
            print('spectrum not calculated. Call get_combined_spectrum() or' + 
                  ' get_stacked_spectra() before subtracting background.')
            return
        spectra = self.get_stacked_spectra()
            
        # alocate space for background-subtracted spectrum and spectra 
        spectrumb = spectrum.copy()
        spectrab = spectra.copy()
        tth_vec = spectrumb[0]
        
        # allocate space for actual backgrounds
        b0 = None
        bgs = None
        b1 = None
        
        # numerize
        if type(background) is list:
            background = np.array(background)
        
        #print('background = ' + str(background) + ', background_type = ' + str(background_type))
        #  ---- calculate, if appropriate, the constant spectrum to 
        # subtract from all spectra ------
        if type(background) is np.ndarray and background_type =='local':
            if background.shape == (spectra.shape[1],):
                if self.verbose:
                    print('will use the constant background to spectra as input.')
                b0 = background
            elif len(background.shape) == 1:
                print('local spectrum input does not have the right shape!')
                return
        elif type(background) is int and background_type == 'index':
            if self.verbose:
                print('will use the background of spectra[' + str(background) + '].')
            b0 = spectra[background]
        elif type(background) in [int, float, np.float64] and \
                background_type in self.data['data_cols']:
            if self.verbose:
                print('going to interpolate to ' + background_type + ' = ' + str(background))
            x = self.data[background_type]
            diff = np.diff(x)
            if not np.all(diff):
                print('WARNING! self.data[\'' + background_type + '\'] is not ' +
                      'monotonially increasing.\n I\'ll try to fix it but no guarantee...')
            try:
                interpolater = interp1d(x, spectra, axis=0, fill_value='extrapolate') 
                if self.verbose:
                    print('interpolater established with scipy.interpolate.interp1d.')
            except ValueError as e:
                print('got this error: ' + str(e) + 
                      '\n...gonna try interpolating to \'t\' first.')
                t_i = self.data[get_timecol(background_type)]
                t = self.data['t']
                print('t.shape = ' + str(t.shape) + ', spectra.shape = ' + str(spectra.shape))
                interpolater = interp1d(t, spectra, axis=0, fill_value='extrapolate') 
                try:
                    background > x[0]
                    if not np.all(diff):
                        print('using a direction mask to interpolate on a ' +
                              'monotonically increasing list.\nThis will get ' +
                              'the first time ' + background_type + ' passes '
                              + str(background))
                        direction = background > x[0]
                        mask = get_direction_mask(x, direction=direction)
                        x, t_i = x[mask], t_i[mask]
                        if not direction:
                            x, t_i = np.flipud(x), np.flipud(t_i)
                    background_type = get_timecol(background_type)
                    background = np.interp(background, x, t_i)
                except:
                    raise                    
            if self.verbose:
                print('will use the image spectrum corresponding to ' + 
                      background_type + ' = ' + str(background) + ' as background.')
            b0 = interpolater(background)
        elif type(background) in [int, float] and background_type == 'local':
            if self.verbose:
                print('will subtract the same constant background' + 
                      ' from each image spectrum')
            b0 = background
        elif self.verbose:
            print('not a combination giving b0. The background for each' + 
                  ' image will be calculated individually.')
        
        if b0 is not None and self.verbose:
            print('Inputs make sense: \nA constant background spectrum will ' + 
                  'be subtracted from each image spectrum in self.spectra')
        
        # ----- find background to global spectrum directly, if appropriate ---------
        if background_type == 'global':
            if type(background) is np.ndarray:
                if background.shape == (spectrum.shape[1], ):
                    b1 = background
                elif background.shape[0] == 2:
                    b1 = np.interp(spectrumb[0], background[0], background[1], left=0, right=0)
            elif background in ['linear', 'endpoint']:
                b1 = get_background_line(spectrum, method=background,
                                         name='global', out='values', 
                                         lincutoff=False,
                                         verbose=self.verbose, **kwargs)
            elif type(background) in [int, float, np.float64]:
                b1 = np.tile(background, np.size(tth_vec))
            if self.verbose and b1 is not None:
                print('Inputs make sense!\n' +  
                      'A global background spectrum will be subtracted.')
        
        
        # --------- subtract directly calculated background 
        #           from each image spectrum in spectra, if appropriate ----------
        
        if b0 is None:  #then the background to each spectrum must be found
            bg = {}
            for i, y_vec in enumerate(spectrab):
                bg[i] = np.zeros(np.shape(y_vec)) # has the length of the full tth vec
                bg_i = None #will only have the length of the image's tth vec
                mask = ~(y_vec==0)
                tth, y = tth_vec[mask], y_vec[mask]
                spec = np.array([tth, y])
                if background_type == 'global':        
                    #print('tth = ' + str(tth) + ', \n spectrumb[0] = ' + 
                    #      str(spectrumb[0]) + ', \b and b1 = ' + str(b1))    #debugging
                    bg_i = np.interp(tth, spectrumb[0], b1, left=0, right=0)                        
                    if self.scan_method == 'sum':
                        bg_i = bg_i / self.N_contributors[mask] #normalize background to one image
                elif background in ['linear', 'endpoint']:
                    #print(i) # for debugging
                    bg_i = get_background_line(spec, method=background, mode='match',
                                               name=' image number ' + str(i),
                                               floor=True, out='values', 
                                               verbose=self.vverbose, **kwargs)   
                if bg_i is not None:
                    bg[i][mask] = bg_i
                    spectrab[i] = y_vec - bg[i]
            if b1 is None:  #calculate it from the individual backgrouhds, bgs
                bgs = np.stack([bg[i] for i in range(len(self))], axis=0)
                if self.scan_method == 'sum':
                    b1 = np.sum(bgs, axis=0)
                else:
                    b1 = np.sum(bgs, axis=0) / self.N_contributors
        else:   # if there is a constant background, subtract it from them all!
            spectrab = spectrab - np.tile(b0, (len(self), 1))
            if b1 is None:  # calculated it from b0
                if self.scan_method == 'sum':
                    b1 = len(self) * b0
                else:
                    b1 = b0

        if show:
            i = show
            fig, ax = plt.subplots()
            x, y = spectrum[0], spectra[i]
            ax.plot(x, y, 'k')
            yb = bgs[i]
            ax.plot(x, yb, 'r')            
        
        # ---------- finalize and save background-subtracted spectra ------ 
        #print('b1.shape = ' + str(b1.shape) + ', and spectrumb.shape = ' + str(spectrumb.shape))
        spectrumb[1] -= b1
        
        self.b0 = b0
        self.bgs = bgs
        self.b1 = b1
        
        self.spectrab = spectrab
        self.spectrumb = spectrumb
        
        self.background = background
        self.background_type = background_type  
        self.bg = True
        
        if self.verbose:
            print('\nfunction \'subtract_background\' finished!\n\n')  
            
            
    def correct_for_refraction(self, delta_eff=None, beta_eff=None, alpha=None, delta_tth=None):
        from .XRD import refraction_correction


        try: 
            corrected = self.corrected
        except AttributeError:
            corrected = False
        
        if corrected:
            print('scan has already been corrected once for refraction.\n' + 
                  '... correcting from original angles')
            tth_0 = self.tth_0.copy()
        else:
            tth_0 = self.spectrum[0].copy()
            self.tth_0 = tth_0
        
        if alpha is None:
            alpha = self.alpha
            
        if delta_eff is None:
            try:
                delta_eff = self.delta_eff
            except AttributeError:
                delta_eff = 5.94e-6
        if beta_eff is None:
            try:
                beta_eff = self.beta_eff
            except AttributeError:
                beta_eff = 2.37e-7
        
        if delta_tth is None:
            delta_tth = refraction_correction(alpha=alpha, delta_eff=delta_eff, 
                                              beta_eff=beta_eff, alpha_c=None)
        else:
            print(f'SHIFTING TTH {-delta_tth} DEG!')
        
        tth = tth_0 - delta_tth
        
        self.data['tth_0'] = tth_0
        self.data['tth'] = tth
        
        self.spectrum[0] = tth
        try:
            self.spectrumb[0] = tth
        except AttributeError:
            pass
        self.corrected = True
        return delta_tth
        
        

    #-------------- functions for integrating and characterizing peaks ----------


    def integrate_spectrum(self, peaks={'Cu_111':([19.65, 20.65], 'brown'),
                                     'CuO_111':([17.55, 18.55], 'k')},
                        override_peaks=False, bg=None,
                        background='linear', background_type=None,
                        background_points=4):

        if bg is None:
            bg = self.bg
        try:
            if bg:
                spectrum = self.spectrumb
            else:
                spectrum = self.spectrum
        except AttributeError:
            print('spectrum not calculated. Call get_combined_spectrum() or' + 
                  ' get_stacked_spectra(). If you want background subtraction' + 
                  '(bg=True), also call subtract_background()')
            raise
        
        x, y = spectrum
        
        if 'peaks' in dir(self) and not override_peaks:
            self.peaks.update(peaks)
        else:
            self.peaks = peaks
            self.integrals = {}
        integrals = {}
        
        for name, props in peaks.items():
            xspan = props[0]
            I = integrate_peak(x, y, xspan, background=background,
                               background_points=background_points)
            if self.vverbose:
                print(name)
            integrals[name] = I
        self.integrals.update(integrals)
        
        if self.vverbose:
            print('Integrated peaks!')
        return integrals    


    def integrate_peaks(self, peaks={'Cu_111':([19.65, 20.65], 'brown'),
                                     'CuO_111':([17.55, 18.55], 'k')},
                        override_peaks=False, bg=None,
                        background='linear', background_type='global',
                        background_points=4, 
                        show=None, ax=None):
        print('\n\nfunction \'integrate_peaks\' at your service!\n')
        
        if self.scan_type == 'tth':
            return self.integrate_spectrum(peaks=peaks,
                        override_peaks=override_peaks, bg=bg,
                        background=background, background_points=background_points)
        
        if bg is None:
            bg = self.bg
        try:
            if bg:
                spectra = self.spectrab
            else:
                spectra = self.spectra
        except AttributeError:
            print('spectrum not calculated. Call get_combined_spectrum() or' + 
                  ' get_stacked_spectra(). If you want background subtraction' + 
                  '(bg=True), also call subtract_background()')
            raise
            
        if 'peaks' in dir(self) and not override_peaks:
            self.peaks.update(peaks)
        else:
            self.peaks = peaks
            self.integrals = {}
        peak_background = {}
        integrals = {}
        

        x = self.spectrum[0]
        if background == 'existing':
            peak_background = self.peak_background
        else:
            if self.verbose:
                print('defining background conditions for peaks')
            if type(background) is int:
                y = spectra[background]
                if self.verbose():
                    print('using image ' + str(background) + ' for background')
            elif background_type in ['average', 'global']:
                y = np.sum(spectra, axis=0) / len(self)
                if self.verbose:
                    print('using a global spectrum for background.')
            for name, props in peaks.items():
                if background is None or background is False:
                    bg = np.zeros(np.shape(x))
                    bg_type = None
                elif type(background) is int or background_type in ['average', 'global']:
                    xspan = props[0]
                    bg = get_peak_background(x, y, xspan, 
                                              background=background,
                                              background_points=background_points)
                    bg_type = 'global'
                else: 
                    bg = background
                    bg_type = background_type
                peak_background[name] = (bg, bg_type)

        self.peak_background = peak_background        
        
        for i in range(len(self)):
            if self.vverbose:
                print('working on image ' + str(i))
            y = spectra[i]
            plotit = show==i
            for name, props in peaks.items():
                xspan = props[0]
                if plotit:
                    axi = ax
                    color = props[1]
                else:
                    axi = None
                    color = None
                bg, bg_type = peak_background[name]
                I = integrate_peak(x, y, xspan, background=bg, background_type=bg_type,
                                   background_points=background_points,
                                   ax=axi, color=color, returnax=False)
                if self.vverbose:
                    print(name)
                if name not in integrals:
                    integrals[name] = []
                integrals[name] += [I]   
            if self.vverbose:
                print('Integrated peaks!')
            
        for key, value in integrals.items():  #numerize and save
            value = np.array(value)
            integrals[key] = value
            self.integrals[key] = value
            self.data[key] = value
            if key not in self.data['data_cols']:
                self.data['data_cols'] += [key]
        
        print('\nfunction \'integrate_peaks\' finished!\n\n')
            
        return integrals                     
    
    
    def get_peaks(self, peaks):
        x, y = self.get_combined_spectrum()
        P = {}
        for name, peak in peaks.items():
            xspan, color = peaks[name]
            try:
                xspan[-1] - xspan[0]
            except TypeError:
                xspan = [xspan, color]
                color = 'k'
            P[name] = Peak(x, y, name=name, xspan=xspan, color=color)
        self.P = P
        return P
    
    def track_peaks(self, peaks):
        spectra = self.get_stacked_spectra(override=False)
        x = self.get_combined_spectrum(override=False)[0]
        N = len(self)
        P = {}
        for name, (xspan, color) in peaks.items():
            P[name] = []
            for n in range(N):
                y = spectra[n]
                P[name] += [Peak(x, y, xspan=xspan, color=color, name=name)]
        self.P = P
        return P
        
    #-------------- functions for plots and videos ----------    
    
    def plot_spectrum(self, bg=None, ax='new', fig=None, color='k', 
                      show_integrals=None, tthspan=None, **kwargs):
        if ax == 'new':
            fig, ax = plt.subplots()
        if bg is None:
            bg = self.bg
        try:
            if bg:
                tth, counts = self.spectrumb
            else:
                tth, counts = self.spectrum
        except AttributeError:
            print('spectrum not calculated. Call get_combined_spectrum() or' + 
                  ' get_stacked_spectra(). If you want background subtraction' + 
                  '(bg=True), also call subtract_background()')
            raise        
        if not tthspan is None:
            mask = np.logical_and(tthspan[0]<tth, tth<tthspan[-1])
            tth = tth[mask]
            counts = counts[mask]
        ax.plot(tth, counts, color=color, **kwargs)
        ax.set_ylabel('counts: ' + self.scan_method + '-' + self.method)
        ax.set_xlabel('tth / deg')
        
        if show_integrals:
            for (name, (xspan, color)) in self.peaks.items():
                integrate_peak(tth, counts, xspan, ax=ax, 
                               color=None, fill_color=color)        
        if fig is None:
            fig = ax.get_figure()
        return fig, ax
    
    
    
    def plot_integrals(self, peaks='existing', fig=None,
                          ax='new', legend=True, **kwargs):
        if ax == 'new':
            fig, ax = plt.subplots()
        
        if peaks == 'existing':
            peaks = self.peaks
        if self.scan_type == 't':
            x = self.data['t']
            x_str = 'time / [s]'
        if 'integrals' not in dir(self):
            self.integrals = {}
        for (name, (xspan, color)) in peaks.items():
            print(name)
            if name not in self.integrals.keys():
                self.integrate_peaks(peaks={name: (xspan, color)}, **kwargs)
            I = self.integrals[name]
            ax.plot(x, I, color=color, label=name)
        ax.set_ylabel('counts')
        ax.set_xlabel(x_str)
        if legend:
            ax.legend()
        if fig is None:
            fig = ax.get_figure()
        return fig, ax
    
    
    def heat_plot(self, stepsize=0.05, override=False, tthspan=None,
                          slits=True, xslits=None, yslits=None, tth=None,
                          method='average', bg=None, min_pixels=10, N_x=300, 
                          ax='new', orientation='xy', logscale=False, zrange=None,
                          aspect='auto', colormap='inferno', 
                          split_tth=None, splitspec={'color':'g', 'linestyle':'-'},
                          tspan='all'):
        #get the raw spectra
        if bg is None:
            bg = self.bg
        try:
            if bg:
                spectra_raw = self.spectrab
            else:
                spectra_raw = self.spectra
        except AttributeError:
            print('spectrum not calculated. Call get_combined_spectrum() or' + 
                  ' get_stacked_spectra(). If you want background subtraction' + 
                  '(bg=True), also call subtract_background()')
            raise
        #print('spectra_raw = \n' + str(spectra_raw)) # debugging
        # Whatever we're scanning against is called x now.    
        if self.scan_type == 't':
            if self.timecol is None:
                timecol = 't'
            else:
                timecol = self.timecol
            x_i = self.data[timecol]
            x_str = 'time / [s]'
        if self.scan_type == 'tth':
            x_i = self.data['tth_scan']
            x_str = 'center tth / deg'
            
        # we want the scan axis to vary linearly, but the input might not.
        f = interp1d(x_i, spectra_raw, axis=0, fill_value='extrapolate') 
        if tspan == 'all':
            x = np.linspace(x_i[0], x_i[-1], num=N_x)
        else:
            x = np.linspace(tspan[0], tspan[-1], num=N_x)
        #print('interpolating to time vector x = ' + str(x)) # debugging
        spectra = f(x)


        # and of course the other dimension, which is tth:
        tth_vec = self.spectrum[0]  #I know this is linear, because it's defined here and in pilatus.py
        
        if tthspan is not None:
            mask = np.logical_and(tthspan[0]<tth_vec, tth_vec<tthspan[-1])
            spectra = spectra[:,mask]
            #print(spectra.shape) # debugging
            tth_vec = tth_vec[mask]
              
        if logscale:
            spectra = np.log(spectra)
        if zrange is None:
            good = np.logical_and(~np.isnan(spectra), ~np.isinf(spectra))
            #print('spectra = \n' + str(spectra)) # debugging
            low = np.min(spectra[good])
            high = np.max(spectra[good])
        else:
            low = zrange[0]
            high = zrange[1]
        spectra[spectra<low] = low
        spectra[spectra>high] = high
        spectra[np.isnan(spectra)] = low
        spectra[np.isinf(spectra)] = low
 
        if orientation == 'xy':
            spectra = np.swapaxes(spectra, 0, 1)
            extent=[x[0], x[-1], tth_vec[0], tth_vec[-1]]
        elif orientation == 'yx':
            extent=[tth_vec[0], tth_vec[-1], x[0], x[-1]]
            
            
        if ax == 'new':
            fig, ax = plt.subplots()
            
        if split_tth:
            I_split = np.argmax(tth_vec>split_tth)
            spectra1 = spectra[:I_split, :]
            extent1 = [x[0], x[-1], tth_vec[0], tth_vec[I_split]]
            
            spectra2 = spectra[I_split:, :]
            extent2 = [x[0], x[-1], tth_vec[I_split], tth_vec[-1]]
            
            ax.imshow(spectra1, extent=extent1,
                      aspect=aspect, origin='lower', cmap=colormap)
            ax.imshow(spectra2, extent=extent2,
                      aspect=aspect, origin='lower', cmap=colormap)
            ax.plot([x[0], x[-1]], [tth_vec[I_split], tth_vec[I_split]], **splitspec)
            
        else:
            ax.imshow(spectra, extent=extent,
                      aspect=aspect, origin='lower', cmap=colormap)
            
        if orientation == 'xy':
            ax.set_xlabel(x_str)
            ax.set_ylabel('TwoTheta / deg')            
        elif orientation == 'yx':
            ax.set_ylabel(x_str)
            ax.set_xlabel('TwoTheta / deg')
        return ax
    
    
    def plot_experiment(self, *args, **kwargs):
        from .plotting import plot_experiment
        if 'plot_type' not in kwargs:
            kwargs['plot_type'] = 'heat'
        return plot_experiment(self, *args, **kwargs)
    
    def make_spectrum_movie(self, duration=20, fps=24,
                   title='default', peaks='existing', bg=None,
                   slits=True, xslits=None, yslits=[60, 430], 
                   xlims=None, ylims=None, tspan=None, full=False,
                   spectrum_specs={},):
        '''
            # tspan is the time/tth/index interval for which the movie is made
        '''    
        if self.scan_type == 't':
            t_vec = self.data['t']
        elif self.scan_type == 'tth':
            t_vec = self.csv_data['TwoTheta']
        else:
            t_vec = np.arange(len(self))        
        
        if tspan is None:  # then use the whole interval
            tspan = [t_vec[0], t_vec[-1]]
        
        if peaks == 'existing':
            try:
                peaks = self.peaks
            except AttributeError:
                peaks = None
        elif type(peaks) is list:
            peaks = peak_colors(peaks)

        if bg is None:
            bg = self.bg
        try:
            if bg:
                spectra = self.spectrab
            else:
                spectra = self.spectra
        except AttributeError:
            print('spectrum not calculated. Call get_combined_spectrum() or' + 
                  ' get_stacked_spectra(). If you want background subtraction' + 
                  '(bg=True), also call subtract_background()')

        
        def make_frame(T):
            t = tspan[0] + T / duration * (tspan[-1] - tspan[0])
            try:
                n = next(i for i, t_i in enumerate(t_vec) if t_i > t) - 1
            except StopIteration:
                n = len(self) - 1
            n = max(n, 0)
            
            if full:
                x, y = self.spectrum[0], spectra[n]
            else:
                y_vec = spectra[n]
                mask = ~(y_vec==0)
                x, y = self.spectrum[0][mask], y_vec[mask]
            
            fig, ax = plt.subplots()
            ax.plot(x, y, )
            
            if peaks is not None:
                for (name, (xspan, color)) in peaks.items():
                    integrate_peak(x, y, xspan, ax=ax, color=None, fill_color=color)
            
            if xlims is not None:
                ax.set_xlim(xlims)
            if ylims is not None:
                ax.set_ylim(ylims)
            if self.scan_type == 't':
                ax.text(0, 0.95, 't = ' + str(np.round(t_vec[n], 2)) + ' s', 
                        bbox={'facecolor':'white'}, transform=ax.transAxes)
            return mplfig_to_npimage(fig)  
        
        if title == 'default':
            title = self.name + '_spectrum.mp4'
        #mpl.use('Agg')  # So that it doesn't print the figures #doesn't seem to work
        #imp.reload(plt)
        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile(title, fps=fps)  
        
    
    def make_movie(self, title='default', duration=20, fps=24,
                   norm='default', tspan=None,
                   slits=True, xslits=None, yslits=[60, 430]):
        '''
            # tspan is the time/tth/index interval for which the movie is made
        '''
        if self.scan_type == 't':
            t_vec = self.data['t']
        elif self.scan_type == 'tth':
            t_vec = self.csv_data['TwoTheta']
        else:
            t_vec = np.arange(len(self))        
        
        if tspan is None:  
            tspan = [t_vec[0], t_vec[-1]]
        
        if norm == 'default':
            if slits:
                immin = None
                immax = None
                for image in self.images.values():
                    image.apply_slits(xslits=xslits, yslits=yslits)
                    if immin is None:
                        immin = np.min(image.im1)
                    else:
                        immin = min(immin, np.min(image.im1))
                    if immax is None:
                        immax = np.max(image.im1)
                    else:
                        immax = max(immax, np.max(image.im1))            
            else:
                immin = min([np.min(image.im) for image in self.images.values()])
                immax = max([np.max(image.im) for image in self.images.values()])                
            norm = [immin, immax]
        if title == 'default':
            title = self.name + '.mp4'
            
        def make_frame(T):
            t = tspan[0] + T / duration * (tspan[-1] - tspan[0])
            try:
                n = next(i for i, t_i in enumerate(t_vec) if t_i > t) - 1
            except StopIteration:
                n = len(self) - 1
            n = max(n, 0)
            fig, ax = plt.subplots()
            ax = self.images[n].show_image(norm=norm, slits=slits, ax=ax)
            if self.scan_type == 't':
                ax.text(0, 0.95, 't = ' + str(np.round(t_vec[n], 2)) + ' s', 
                        bbox={'facecolor':'white'}, transform=ax.transAxes)
            return mplfig_to_npimage(fig)  
        

        #mpl.use('Agg')  # So that it doesn't print the figures #doesn't seem to work
        #imp.reload(plt)
        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile(title, fps=fps)

