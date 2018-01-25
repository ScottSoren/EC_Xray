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
try:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
except ImportError:
    print('you need the package moviepy to be able to make movies!')

from .import_data import load_from_file, read_macro
from .combining import timestamp_to_seconds, seconds_to_timestamp
from .pilatus import Pilatus, calibration_0, shape_0
from .XRD import integrate_peak

timestamp_matcher = '([0-9]{2}\:){2}[0-9]{2}'

def get_images(directory, tag, shape=shape_0, calibration=calibration_0, 
               pixelmax=None, verbose=True,  vverbose=False):
    if verbose:
        print('\n\nfunction \'get_images\' at your service!\n')
    try:
        lslist = os.listdir(directory)
    except FileNotFoundError:
        print('The directory doesn\'t exist. get_images is returning a blank dictionary.')
        return {}
    if verbose:
        print(str(len(lslist)) + ' items in ' + directory)
    imagenames = [f for f in lslist if f[-4:]=='.raw' and tag in f]
    if verbose:
        print(' of which ' + str(len(imagenames)) + ' are image files including \'' + tag + '\'')
    images = {}  
    for f in imagenames:
        n = int(f[-8:-4])  #this is the image number as SPEC saves them
        filepath = directory + os.sep + f
        images[n] = Pilatus(filepath, shape=shape, calibration=calibration_0, 
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



class ScanImages:
    def __init__(self, csvfile=None, directory=None, pilatusfilebase='default', 
                 tag=None, scan_type='time', calibration=calibration_0, 
                 macro=None, tth=None, alpha=None, timestamp='abstimecol', 
                 pixelmax=None, timecol=None, abstimecol='pd16', 
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
        timestamp='filename' tries to get it from the file name
        '''
        # ------------------- get csv ---------------------#
        if csvfile is None:
            if tag is None or directory is None:
                print('need a csv file name or a directory and a tag!')
                return
            lslist = os.listdir(directory)
            try:
                csvname = next(f for f in lslist if f[-4:]=='.csv' and tag in f)
            except StopIteration:
                print(lslist)
                print('Cound not find a csvname containing ' + tag + ' in ' + directory + '\n(ls above)')
            print('Loading Scan from directory = ' + directory +
                  '\n found csvname = ' + csvname)            
        else:
            directory, csvname = os.path.split(csvfile)
        
        csvfilepath = directory + os.sep + csvname
        #self.csv_data = load_from_csv(csvfilepath, timestamp=timestamp)
        self.csv_data = load_from_file(csvfilepath, data_type='SPEC', timestamp=timestamp)
        
        # --------------  install easy metadata ------------- #
        name = csvname[:-4] # to drop the .csv
        print('Loading Scan: directory = ' + directory + ',\n\tname = ' + name)
        self.directory = directory
        self.name = name
        self.timecol = timecol
        self.abstimecol = abstimecol        
        
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

        #-------------------- get images! ------------------------#
        if pilatusfilebase == 'default':
            pilatus_directory = directory + os.sep + 'Pilatus'
            tag_pilatus = name
        else:
            pilatus_directory, tag_pilatus = os.path.split(pilatusfilebase)
        self.images = get_images(pilatus_directory, tag=tag_pilatus, 
                                 calibration=calibration_0, verbose=verbose,
                                 pixelmax=pixelmax,
                                 vverbose=vverbose)
        
        if len(self.images) == 0:
            print('THIS SCAN IS EMPTY!!!!')
            self.empty = True
            return
        else:
            self.empty = False

        # ---------------------- get timestamp and timecol -----------------#
        if timestamp in ['filename']:
            timestamp = self.csv_data['timestamp']
        elif timestamp in ['abstimecol']:
            try:
                value = self.csv_data[abstimecol][0]
                a = re.search(timestamp_matcher, value)
                timestamp = a.group()
                print('line 163: timestamp = ' + timestamp)
                if timecol is not None:
                    t1 = a.csv_data[timecol][0]
                    timestamp = seconds_to_timestamp(timestamp_to_seconds(timestamp) - t1)
            except OSError: # a dummy error... I want to actually get the error messages at first
                pass
        self.timestamp = timestamp
        print('line 170: self.timestamp = ' + self.timestamp)
        

        # ------------------------- organize csvdata and metadata ---------- #        
  
        self.data = self.csv_data.copy()
        # this will conveniently store useful data, some from csv_data

        if timecol is None and abstimecol is not None:
            self.get_timecol_from_abstimecol()
        # put in the timecol!
            
        self.csv_into_images()

        self.verbose = verbose
        self.vverbose = vverbose
    
    
    def csv_into_images(self):
        if self.scan_type == 't':
            if 't' not in self.data:
                try:
                    self.data['t'] = self.csv_data[self.timecol]
                except KeyError:
                    print('self.timecol = ' + str(self.timecol) + 
                           ' is not in csv data. Check yo self.')
                    return
                if 't' not in self.data['data_cols']:
                    self.data['data_cols'] += ['t']
            print('line 200: self.timestamp = ' + self.timestamp + 
                  ',\tlen(self) = ' + str(len(self)) + 
                  ',\tlen(self.data[\'t\'])) = ' + str(len(self.data['t']))) 
            for i in range(len(self)):               
                self.images[i].t = self.data['t'][i]
                # I don't like it, but that's where SPEC saves t.
                # data columns 'TTIMER' and 'Seconds' contain nothing.
                # If t is recorded, tth and alpha are constant, but...
                # The tth and alpha are not saved anywhere. The user must 
                # input them, or input a macro to read. Done in self.__init__
                self.images[i].tth = self.tth
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
        abstime = self.csv_data[self.abstimecol]
        t = []
        print('line 228: self.timestamp = ' + self.timestamp)
        t0 = timestamp_to_seconds(self.timestamp)
        for time in abstime:
            value = re.search(timestamp_matcher, time).group()
            t += [timestamp_to_seconds(value) - t0]
        self.data['t'] = t
        if 't' not in self.data['data_cols']:
            self.data['data_cols'] += ['t']
            
    
    def integrate_peaks(self, peaks={'Cu_111':([19.65, 20.65], 'brown'),
                                     'CuO_111':([17.55, 18.55], 'k')},
                        spectrum_specs={}, override_peaks=False,
                        background='linear', background_points=4,
                        stepsize=0.05, override=False,
                        slits=True, xslits=None, yslits=None, tth=None,
                        method='average', min_pixels=10):
        if 'peaks' in dir(self) and not override_peaks:
            self.peaks.update(peaks)
        else:
            self.peaks = peaks
            self.integrals = {}
        integrals = {}
        for i in range(len(self)):
            image = self.images[i]
            print('working on image ' + str(i))
            x, y = image.tth_spectrum(override=override, stepsize=stepsize, 
                                      method=method, min_pixels=min_pixels, 
                                      tth=tth, xslits=xslits, yslits=yslits)
            print('... calculated spectrum!')
            for name, props in peaks.items():
                xspan = props[0]
                I = integrate_peak(x, y, xspan, background=background,
                                   background_points=background_points)
                print(name)
                if name not in integrals:
                    integrals[name] = []
                integrals[name] += [I]      
            print('... and integrated peaks!')
            
        for key, value in integrals.items():  #numerize and save
            value = np.array(value)
            integrals[key] = value
            self.integrals[key] = value
            self.data[key] = value
            if key not in self.data['data_cols']:
                self.data['data_col'] += [key]
            
        return(integrals)                     
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, indices):
        if type(indices) is int:
            return self.images[indices]
        elif type(indices) in [list, tuple]:
            return [self.image[i] for i in indices]
        print('indices must be an integer or sequence of integers')

    
    def plot_integrals(self, peaks='existing', stepsize=0.05, override=False,
                          slits=True, xslits=None, yslits=None, tth=None,
                          method='average', min_pixels=10,
                          ax='new', legend=True):
        if ax == 'new':
            fig, ax = plt.subplots()
        
        if peaks == 'existing':
            peaks = self.peaks
        if self.scan_type == 't':
            x = self.data['t']
            x_str = 'time / s'
        if 'integrals' not in dir(self):
            self.integrals = {}
        for (name, (xspan, color)) in peaks.items():
            print(name)
            if name not in self.integrals.keys():
                self.integrate_peaks(peaks={name: (xspan, color)})
            I = self.integrals[name]
            ax.plot(x, I, color=color, label=name)
        ax.set_ylabel('counts')
        ax.set_xlabel(x_str)
        if legend:
            ax.legend()
        return ax
        
        
    
    def get_combined_spectrum(self, stepsize=0.05, override=False,
                              slits=True, xslits=None, yslits=None,
                              method='average', min_pixels=10, tth=None,
                              scan_method='sum', out='spectrum',):
        '''
        spectrum specs are arguments to Pilatus.tth_spectrum, in module 
        pilatus.py
        '''
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
                    contributors[n] += 1
                else:
                    bins[n] = counts
                    contributors[n] = 1

        tth_vec = []
        counts_vec = []
        n_min = min(bins.keys())
        n_max = max(bins.keys())
        for n in range(n_min, n_max+1):
            tth_vec += [(n + 0.5) * stepsize]
            if scan_method == 'average':
                counts_vec += [bins[n]/contributors[n]]
            else:
                counts_vec += [bins[n]]
                
        tth_vec = np.array(tth_vec)
        counts_vec = np.array(counts_vec)
        spectrum = (tth_vec, counts_vec)
        
        self.bins = bins
        self.spectrum = spectrum
        self.data.update({'tth':tth_vec, 'counts':counts_vec})
        if 'counts' not in self.data['data_cols']:
            self.data['data_cols'] += ['counts']
        
        if out == 'spectrum':
            return spectrum
        elif out == 'bins':
            return bins
        
    
    def get_stacked_spectra(self, stepsize=0.05, override=False,
                          slits=True, xslits=None, yslits=None,
                          method='average', min_pixels=10, tth=None):
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
        print('spectra shape : ' + str(np.shape(spectra)))
        self.spectra = spectra
        return spectra
    
    def heat_plot(self, stepsize=0.05, override=False,
                          slits=True, xslits=None, yslits=None, tth=None,
                          method='average', min_pixels=10, N_x=300, 
                          ax='new', orientation='xy',
                          aspect='auto', colormap='inferno', 
                          tspan='all'):
        #get the raw spectra
        spectra_raw = self.get_stacked_spectra(stepsize=stepsize, tth=tth,
                                               method=method, override=override,
                                               min_pixels=min_pixels,
                                               xslits=xslits, yslits=yslits) 
        # Whatever we're scanning against is called x now.
        if self.scan_type == 't':
            x_i = self.data['t']
            x_str = 'time / s'
        if self.scan_type == 'tth':
            x_i = self.data['tth_scan']
            x_str = 'center tth / deg'
            
        # we want the scan axis to vary linearly, but the input might not.
        f = interp1d(x_i, spectra_raw, axis=0) 
        if tspan == 'all':
            x = np.linspace(x_i[0], x_i[-1], num=N_x)
        else:
            x = np.linspace(tspan[0], tspan[-1], num=N_x)
        spectra = f(x)
 
        # and of course the other dimension, which is tth:
        tth_vec = self.spectrum[0]  #I know this is linear, because it's defined here and in pilatus.py

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
        

    
    def make_spectrum_movie(self, duration=20, fps=24,
                   title='default', peaks='existing',
                   slits=True, xslits=None, yslits=[60, 430], 
                   xlims=None, ylims=None,
                   spectrum_specs={},):
    
        if self.scan_type in ['t', 'tth']:
            t_total = self.csv_data['TwoTheta'][-1]
            t_vec = self.csv_data['TwoTheta']
        else:
            t_total = len(self)
            t_vec = np.arange(t_total)        
        
        if peaks == 'existing':
            peaks = self.peaks
        elif type(peaks) is list:
            peaks = peak_colors(peaks)
        
        def make_frame(T):
            t = T / duration * t_total
            try:
                n = next(i for i, t_i in enumerate(t_vec) if t_i > t) - 1
            except StopIteration:
                n = len(self) - 1
            n = max(n, 0)

            fig, ax = plt.subplots()
            x, y = self.images[n].tth_spectrum(**spectrum_specs)
            ax = self.images[n].plot_spectrum(ax=ax, specs={'color':'k'},
                                             **spectrum_specs)
            
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
        
    
    def make_movie(self, duration=20, fps=24,
                   title='default', norm='default', 
                   slits=True, xslits=None, yslits=[60, 430]):
        '''
        '''
        if self.scan_type in ['t', 'tth']:
            t_total = self.csv_data['TwoTheta'][-1]
            t_vec = self.csv_data['TwoTheta']
        else:
            t_total = len(self)
            t_vec = np.arange(t_total)
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
            t = T / duration * t_total
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


    def slim(self):
        '''
        deletes maps to save RAM space.
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
