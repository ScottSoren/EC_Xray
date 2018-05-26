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

try:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
except ImportError:
    print('you need the package moviepy to be able to make movies!')

from .import_data import (load_from_file, read_macro,
                        epoch_time_to_timestamp, timestamp_to_epoch_time)
from .pilatus import Pilatus, calibration_0, shape_0
from .XRD import integrate_peak

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
    print(tag) # debugging
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

def get_background_line(spectrum, method='endpoint', floor=True, N_end=3, 
                        steps0=2, lincutoff=True, p1=0.1, p2=0.4,
                        name='\'name\'', out='values', verbose=False):
    '''
    A couple cool algorithms for finding a linear background to data with
    peaks, knowing that there's a risk that the data may start or end on 
    a peak. spectrum is a 2xN numpy array containing x and y.
    
    ---------- method = 'endpoint' ---------------
    Draws a line of best fit between the center of N_end left end points and 
    N_end right end points.
    If floor=True: If the portion of the points an amount below the line
    corresponding to improbability p2 given the standard deviation of the 
    residuals of the endpoitns has itself a probability less than p1, 
    the endpoints are moved inwards at steps of N_end, starting with the left
    and minimizing the total number of steps in. The initial number of steps
    in, steps0, is 2 on each side by default to avoid cutoff effects. 
    The background is assumed to drop linearly in the cutoff region if 
    cutoff is True, and otherwise the cutoff region is assumed to be entirely
    background up to the endpoint level.
    
    --------- method = 'filter' ------------
    Iteratively fit a line of best fit and remove outliers until there are
    none.
    A point is considered an outlier if its square error from the fit line is
    greater than threshhold. threshhold is set such that the probability
    given a gaussian distribution of residual of having one outlier point
    given the measured standard deviation is less than p1.
    If floor=True, no outlying point lying below the line can be ignored.
    
    ---------
    The function returns either
    out = 'line' : the [slope, intercept] of the background line
    out = 'values' : the linear background values corresponding to x
    '''
    
    from scipy.stats import norm
    x, y = spectrum
    N = len(x)
    
    if method == 'endpoint':
        again = True
        if type(steps0) is int:
            steps0 = [steps0, steps0]
        N_left, N_right = N_end*steps0[0], N_end*steps0[-1] # cutoff region
        z = norm.ppf(0.5 + p1) # max acceptable standard deviation from mean
        sigma = np.sqrt(N * p2 * (1-p2)) # standard deviation of number below
        max_below = N * p2 + sigma * z
        steps = steps0
        while again:
            left = np.arange(steps[0]*N_end, (steps[0] + 1)*N_end)
            right = np.arange(N - (steps[1] + 1)*N_end, N - steps[1]*N_end)
            both = np.append(left, right)
            try:
                x_ends, y_ends = x[both], y[both]
            except IndexError:
                print('Couldn\'t find background endpoints meeting your ' + 
                      ' demands for ' + name + '. steps = ' + str(steps))
                raise
            poly = np.polyfit(x_ends, y_ends, 1)
            bg = poly[0] * x + poly[1]
            if lincutoff: # bg drops linearly to zero in cutoff regions:
                bg[:N_left] = np.linspace(0, bg[N_left], num=N_left)
                bg[-N_right:] = np.linspace(bg[-N_right], 0, num=N_right)
            else: # everything in cutoff up to linear bg is bg
                bg[:N_left] = np.min(np.stack([y[:N_left],  \
                                 np.tile(bg[N_left], (N_left,))]), axis=0)
                bg[-N_right:] = np.min(np.stack([y[-N_right:],  \
                                 np.tile(bg[-N_right], (N_right,))]), axis=0)
            if floor:
                res_ends = y_ends - bg[both]
                std = np.std(res_ends)
                N_below = np.sum(y < bg - std * z)
                again = (N_below > max_below)
                if steps[0] == steps0[0]:
                    if verbose:
                        print('moving left endpoint way in to find good background')
                    steps = [steps[1] + 1, steps0[1]]
                else:
                    steps = [steps[0] - 1, steps[1] + 1]
                    if verbose:
                        print('moving right endpoint in and left out to find good background')
            else:
                again = False
    else:
        print('get_background_line(method=\'' + method + '\'...) ' + 
              'not implemented! Using method = \'endpoint\'')
        return get_background_line(spectrum, method='endpoint', floor=floor, 
                                   N_end=N_end, p1=p1, p2=p2, out=out,
                                   verbose=verbose)
    if out=='line':
        return poly
    else:
        return bg


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
    def __init__(self, csvfile=None, directory=None, name=None,
                 pilatusfilebase='default', 
                 tag=None, scan_type='time', calibration=calibration_0, 
                 macro=None, tth=None, alpha=None, timestamp=None, tstamp=None, 
                 pixelmax=None, timecol=None, abstimecol=None, tz=None,
                 slits=True, xslits=None, yslits=[60, 430], 
                 scan=None, copy=False,
                 verbose=True, vverbose=False):
        '''
        give EITHER a csvfile name with full path, or a directory and a tag.
        pilatusfilebase can be constructed from this, and used to import the
        Pilatus image objects.
        The calibration is passed on to the Pilatus objects.
        The macro is read to get the (tth, alpha) values which aren't scanned,
        though they can also be put in manually.
        timestamp can be either a str like 'hh:mm:ss' or a pointer.
        timestamp='abstimecol' uses the first value of the specified timecol in the csvfile
        timestamp=None tries to get it from the file
        '''
        # ------ for development: new code with pre-loaded data -------#
        if copy: # take all data (images, csv_data, etc) from another scan
            for attr in dir(scan):
                if attr not in dir(self): # because we don't want to replace e.g. functions
                    setattr(self, attr, getattr(scan, attr))
            try:
                self.copied += 1
            except AttributeError: 
                self.copied = 1
            return
        
        # ------------------- get csv and name ---------------------#
        if csvfile is None:
            if tag is None or directory is None:
                print('need a csv file name or a directory and a tag!')
                return
            lslist = os.listdir(directory)
            try:
                csvname = next(f for f in lslist if f[-4:]=='.csv' and 'scan' in f and tag in f)
            except StopIteration:
                print(lslist)
                print('Cound not find a csvname containing ' + tag + ' in ' + directory + '\n(ls above)')
                csvname = None
          
        else:
            directory, csvname = os.path.split(csvfile)
            
        print('Loading Scan from directory = ' + directory +
              '\n found csvname = ' + str(csvname))   
        
        if name is None:
            if csvname is not None:
                name = csvname[:-4] # to drop the .csv
            else:
                name = tag
        
        if csvname is not None:
            csvfilepath = directory + os.sep + csvname
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
            print('THIS SCAN IS EMPTY!!!!')
            self.empty = True
            return
        else:
            self.empty = False
        
        
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
                
        

        # ------------------------- organize csvdata and metadata ---------- #        
        
        if hasattr(self, 'csv_data'):
            self.data = self.csv_data.copy()
            self.csv_into_images()
        else:
            self.data = {'title':name, 'data_type':'spec'}
            self.data['data_cols'] = []
        
        for col, attr in [('TwoTheta', 'tth'), ('alpha','alpha'), ('t_abs', 'tstamp')]:
            self.data[col] = np.array([getattr(self.images[i], attr) for i in range(len(self))])
            self.data['data_cols'] += [col]
                
        # this will conveniently store useful data, some from csv_data

        if timecol is None and abstimecol is not None:
            # put in the timecol!
            self.get_timecol_from_abstimecol()


        # ---------------------- get timestamp and timecol -----------------#
        if timestamp in ['filename', 'csv','file']:
            tstamp = self.csv_data['tstamp']
            timestamp = epoch_time_to_timestamp(tstamp, tz=tz)
        elif timestamp in ['pdi']:
            tstamp = self.images[0].tstamp
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
                print('line 163: timestamp = ' + timestamp)
                if timecol is not None:
                    t1 = a.csv_data[timecol][0]
                    timestamp = epoch_time_to_timestamp(tstamp - t1, tz=tz)
                    #this is to correct for the fact that tstamp refers to the
                    #first datapoint
            except OSError: # a dummy error... I want to actually get the error messages at first
                pass
        
        if self.scan_type == 't':
            if 't' not in self.data:
                if 't_abs' in self.data:
                    tstamp = self.data['t_abs'][0]
                    t = self.data['t_abs'] - tstamp
                else:
                    try:
                        t = self.csv_data[self.timecol]
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
                self.data['t'] = t
            if 't' not in self.data['data_cols']:
                self.data['data_cols'] += ['t']            

        self.tstamp = tstamp
        self.timestamp = timestamp
        
        if 'tstamp' not in self.data or self.data['tstamp'] is None:
            # and the tstamp!
            self.data['tstamp'] = tstamp
        
        #print('line 170: self.timestamp = ' + str(self.timestamp))
        # This code is a bit of a mess, and timestamp is here only for sanity-checking
        #purposes. All math will refer to tstamp
        
        #------- finished ---------- #
        if self.verbose:
            print('ScanImages object with name ' + self.name + 
                  ' imported!\n\n')        
    
    
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, indices):
        if type(indices) is int:
            if type(indices) is int and indices<0:
                indices = len(self) + indices
            return self.images[indices]
        elif type(indices) in [list, tuple]:
            return [self.image[i] for i in indices]
        print('indices must be an integer or sequence of integers')    
    
    
    
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
            for i in range(len(self)):               
                self.images[i].t = self.data['t'][i] #even though this is never read
                
            print('line 235: self.timestamp = ' + str(self.timestamp) + 
                  ',\tlen(self) = ' + str(len(self)) + 
                  ',\tlen(self.data[\'t\'])) = ' + str(len(self.data['t']))) 
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

 
    
    #-------------- functions for calculating spectra and derived quantities ----------
    
    def get_combined_spectrum(self, stepsize=0.05, override=False,
                              slits=True, xslits=None, yslits=None,
                              method='sum', min_pixels=10, tth=None,
                              scan_method='sum', out='spectrum',):
        '''
        spectrum specs are arguments to Pilatus.tth_spectrum, in module 
        pilatus.py
        '''
        
        if self.verbose:
            print('\n\nfunction \'get_combined_spectrum\' at your service!\n')
            t0 = time.time()
            print('t = 0')
            print('calculating tth spectrum for each of ' + str(len(self)) + 
                  ' images, storing in Pilatus objects, and adding them all up.')

        if tth is not None:
            self.tth = tth
        bins = {}
        contributors = {}
        for i in range(len(self)):
            bins_i = self.images[i].tth_spectrum(out='bins', override=override,
                                                stepsize=stepsize, method=method,
                                                min_pixels=min_pixels, tth=tth,
                                                xslits=xslits, yslits=yslits,
                                                verbose=self.vverbose)
            for n, counts in bins_i.items():
                if type(n) is not int:
                    continue
                if n in bins:
                    bins[n] += counts
                    contributors[n] += [i]
                else:
                    bins[n] = counts
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

    
    def get_stacked_spectra(self, stepsize=0.05, override=False,
                          slits=True, xslits=None, yslits=None,
                          method='average', min_pixels=10, tth=None):
        if self.verbose:
            print('\n\nfunction \'get_stacked_spectra\' at your service!\n')
        if not override and hasattr(self, 'spectrum') and self.spectrum is not None:
            combined_spectrum = self.spectrum
            if self.verbose:
                print('using the already-calculated image spectra')
        else:
            combined_spectrum = self.get_combined_spectrum(out='spectrum',
                                                           stepsize=stepsize, 
                                                           method=method, tth=tth,
                                                           min_pixels=min_pixels,
                                                           xslits=xslits, yslits=yslits) 
        #this generates all the images' spectra, so they're saved when called later.
        tth_vec = combined_spectrum[0]       
        spectrums = []        # collection of the individual spectrum from each image
        for i in range(len(self)):
            tth_i, counts_i = self.images[i].spectrum  
            #spectra were generated during call to self.get_combined_spectrum
            spectrums += [np.interp(tth_vec, tth_i, counts_i, left=0, right=0)]
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
        for i, im in self.images.items():
            if hasattr(im, 'map_xyz'):
                del(im.map_xyz)
            if hasattr(im, 'map_xyz_prime'):
                del(im.map_xyz_prime)
            if hasattr(im, 'map_tth'):
                del(im.map_tth)
            if hasattr(im, 'map_bin'):
                del(im.map_bin)
            
    
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
                    bg_i = get_background_line(spec, method=background, 
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
        
    
    def integrate_peaks(self, peaks={'Cu_111':([19.65, 20.65], 'brown'),
                                     'CuO_111':([17.55, 18.55], 'k')},
                        spectrum_specs={}, override_peaks=False,
                        background='linear', background_points=4,
                        stepsize=0.05, override=False,
                        slits=True, xslits=None, yslits=None, tth=None,
                        method='average', min_pixels=10, bg=None):
        print('\n\nfunction \'integrate_peaks\' at your service!\n')
        
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
        integrals = {}
        
        x = self.spectrum[0]
        for i in range(len(self)):
            if self.vverbose:
                print('working on image ' + str(i))
            y = spectra[i]
            for name, props in peaks.items():
                xspan = props[0]
                I = integrate_peak(x, y, xspan, background=background,
                                   background_points=background_points)
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
            
        return(integrals)                     


    
    
    #-------------- functions for plots and videos ----------    
    
    def plot_spectrum(self, bg=None, ax='new', fig=None, color='k'):
        if ax == 'new':
            fig, ax = plt.subplots()
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
        ax.plot(spectrum[0], spectrum[1], color=color)
        ax.set_ylabel('counts: ' + self.scan_method + '-' + self.method)
        ax.set_xlabel('tth / deg')
        if fig is None:
            fig = ax.get_figure()
        return fig, ax
        

    def plot_integrals(self, peaks='existing',
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
        return ax
    
    
    def heat_plot(self, stepsize=0.05, override=False, tthspan=None,
                          slits=True, xslits=None, yslits=None, tth=None,
                          method='average', bg=None, min_pixels=10, N_x=300, 
                          ax='new', orientation='xy', logscale=False, zrange=None,
                          aspect='auto', colormap='inferno', 
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
        spectra = f(x)


        # and of course the other dimension, which is tth:
        tth_vec = self.spectrum[0]  #I know this is linear, because it's defined here and in pilatus.py
        
        if tthspan is not None:
            mask = np.logical_and(tthspan[0]<tth_vec, tth_vec<tthspan[-1])
            spectra = spectra[:,mask]
            print(spectra.shape)
            tth_vec = tth_vec[mask]
        
        if logscale:
            spectra = np.log(spectra)
        if zrange is None:
            good = np.logical_and(~np.isnan(spectra), ~np.isinf(spectra))
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
            [tth_vec[0], tth_vec[-1], x[0], x[-1]]

        if ax == 'new':
            fig, ax = plt.subplots()
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

