# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:07:45 2016
Most recently edited: 17B21

@author: scott

copied from EC_MS on 17L09 as last commited to EC_MS with code c1c6efa, 
and modified from there

"""

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
#from mpl_toolkits.axes_grid1 import make_axes_locatable

    

standard_colors = {'XAS':'b','XAS1':'r','XAS2':'g'}


def plot_vs_potential(scan, 
                      peaks=None,
                      tspan=0, RE_vs_RHE=None, A_el=None, cycles='all',
                      ax1='new', ax2='new', ax=None, #spec='k-',
                      overlay=0, logplot = [1,0], leg=False,
                      verbose=True, removebackground=None,
                      #masses=None, mols=None, unit=None,
                      fig=None, spec={}):
    '''
    This will plot current and select MS signals vs E_we, as is the 
    convention for cyclic voltammagrams. added 16I29
    
    #there's a lot of code here that's identical to plot_experiment. Consider
    #having another function for e.g. processing these inputs.
    '''
    if verbose:
        print('\n\nfunction \'plot_vs_potential\' at your service!\n')
    from .EC import sync_metadata, select_cycles
    
    if type(logplot) is not list:
        logplot = [logplot, False]
    if removebackground is None:
        removebackground = not logplot[0]

    scanobject = False
    if type(scan) is dict:
        data = scan.copy()
    else:
        scanobject = True
        data = scan.data.copy()
        
    if not cycles == 'all':
        data = select_cycles(data, cycles, verbose=verbose)
        
    #prepare axes. This is ridiculous, by the way.
    if ax == 'new':
        ax1 = 'new'
        ax2 = 'new'
    elif ax is not None:
        ax1 = ax[0]
        ax2 = ax[1]
    if ax1 != 'new':
        figure1 = ax1.figure
    elif ax2 != 'new':
        figure1 = ax2.figure
    else:
        if fig is None:
            figure1 = plt.figure()
        else:
            figure1 = fig
    if overlay:
        if ax1 == 'new':
            ax1 = figure1.add_subplot(111)
        if ax2 == 'new':
            ax2 = ax1.twinx()
    else:
        if ax1=='new':
            gs = gridspec.GridSpec(3, 1)
            #gs.update(hspace=0.025)
            ax1 = plt.subplot(gs[0:2, 0])
            ax2 = plt.subplot(gs[2, 0])
    if type(logplot) is int:
        logplot = [logplot,logplot]
    if logplot[0]:
        ax1.set_yscale('log')
    if logplot[1]:
        ax2.set_yscale('log')
            
    # get EC data
    V_str, J_str = sync_metadata(data, RE_vs_RHE=RE_vs_RHE, A_el=A_el)
    V = data[V_str]
    J = data[J_str]

    #get time variable and plotting indexes
    t = data['time/s']
    if tspan == 0:                  #then use the whole range of overlap
        tspan = data['tspan']
    mask_plot = np.logical_and(tspan[0]<t, t<tspan[1])
    
    if ax2 is not None:
        #plot EC-lab data
        ec_spec = spec.copy()
        if 'color' not in ec_spec.keys():
            ec_spec['color'] = 'k'
        ax2.plot(V[mask_plot], J[mask_plot], **ec_spec)      
            #maybe I should use EC.plot_cycles to have different cycles be different colors. Or rewrite that code here.
        ax2.set_xlabel(V_str)
        ax2.set_ylabel(J_str)
    
#    print(masses)
    if ax1 is not None: #option of skipping an axis added 17C01
        if peaks == 'existing':
            if scanobject:
                peaks = scan.peaks
            else:
                print('need a scan object to read existing peaks!')   
        elif peaks is None:
            peaks = {}
        try:
            colors = dict([(name, color) for (name, (interval, color)) in peaks.items()])
        except ValueError:
            colors = peaks
    
        for (key, color) in colors.items():
            x = data['t']
            y = data[key]
            try:
                Y = np.interp(t, x, y)  #obs! np.interp has a has a different argument order than Matlab's interp1
            except ValueError as e:
                print('can\'t interpolate onto time for plotting vs potential!')
                print('x ' + str(x) + '\ny ' + str(y) + '\nt ' + str(t))
                print(e)
            xray_spec = spec.copy()
            if 'color' not in xray_spec.keys():
                xray_spec['color'] = color
            ax1.plot(V[mask_plot], Y[mask_plot], label=key, **xray_spec)
        
        ax1.xaxis.tick_top()
        ax1.set_ylabel('counts / [a.u.]')
        if leg:
            ax1.legend()
    
    if verbose:
        print('\nfunction \'plot_vs_potential\' finished!\n\n')
    
    for ax in [ax1, ax2]: 
        if ax is not None:
            ax.tick_params(axis='both', direction='in') #17K28  
        
        #parameter order of np.interp is different than Matlab's interp1
    return [ax1, ax2]    
    

def smooth_data(data_0, points=3, cols=None, verbose=True):
    '''
    Does a moving-average smoothing of data. I don't like it, but
    experencing problems 17G26
    '''
    data = data_0.copy()
    if cols is None:
        cols = data['data_cols']
    for col in cols:
        if verbose:
            print('smoothening \'' + col + '\' with a ' + str(points) + '-point moving average')
        x = data[col]
        c = np.array([1] * points) / points
        #print(str(len(c)))
        #data[col] = 0 #somehow this doesn't come through, 17G25
        data[col] = np.convolve(x, c, mode='same')
        #print('len = ' + str(len(x)))
        #x = None

    data['test'] = None #This does make it through. 
    #data['U vs RHE / [V]'] = None #this doesn't make it through either.
    return data    



def plot_simple(data, peaks={'XAS':'b'},
                tspan=None, ax='new', unit='a.u.',
                logplot=False, saveit=False, leg=False, 
                override=False, verbose=True):
    '''
    plots selected masses for a selected time range from MS data or EC_MS data
    Could probably be simplified a lot, to be the same length as plot_fluxes
    '''
    if verbose:
        print('\n\nfunction \'plot_simple\' at your service! \n Plotting from: ' + 
              data['title'])

    from .combining import get_timecol
    if tspan is None:
        tspan = data['tspan']
    if ax == 'new':
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)    
    lines = {}
    #note, tspan is processed in get_signal, and not here!
    if type(lines) is str:
        lines = [lines]
    if type(lines) is list:
        c = lines
        lines = {}
        for m in c:
            color = standard_colors[m]
            peaks[m] = color

    for peak, color in peaks.items():
        if verbose:
            print('plotting: ' + peak)
        try:
            timecol = get_timecol(peak)
            x = data[timecol]
            y = data[peak]
            mask = np.logical_and(tspan[0]<x, x<tspan[-1])
            x = x[mask]
            y = y[mask]
        except KeyError:
            print('Can\'t get signal for ' + str(peak))
            continue
        lines[peak] = ax.plot(x, y, color, label = peak) 
        #as it is, lines is not actually used for anything         
    if leg:
        if type(leg) is not str:
            leg = 'lower right'
        ax.legend(loc=leg)
    ax.set_xlabel('time / [s]')
    ax.set_ylabel('signal / [' + unit + ']')           
    if logplot: 
        ax.set_yscale('log') 
    ax.tick_params(axis='both', direction='in') #17K28  
    if not 'fig' in locals():
        fig = ax.get_figure()
    if verbose:
        print('function \'plot_simple\' finsihed! \n\n')
    return fig, ax


    
def plot_experiment(scan,
                    plot_type='timescan', peaks='existing',
                    tspan=None, overlay=False, logplot=[False,False], verbose=True,   
                    plotpotential=True, plotcurrent=True, ax='new',
                    RE_vs_RHE=None, A_el=None, removebackground=True,
                    saveit=False, title=None, leg=False, tthspan=None,
                    V_color='k', J_color='r', V_label=None, J_label=None,
                    fig=None, t_str=None, J_str=None, V_str=None, bg=None,
                    ): 
    '''
    this plots signals or fluxes on one axis and current and potential on other axesaxis
    '''
    if verbose:
        print('\n\nfunction \'plot_experiment\' at your service!\n')
    from .EC import sync_metadata
    
    if ax == 'new':
        if fig is None:
            figure1 = plt.figure()
        else:
            figure1 = fig
            plt.figure(figure1.number)
            print('plot_expeiriment using ' + str(fig))
        if overlay:
            ax = [figure1.add_subplot(111)]
            ax += [ax[0].twinx()]                     
        else:
            gs = gridspec.GridSpec(3, 1)
            #gs.update(hspace=0.025)
            #gs.update(hspace=0.05)
            ax = [plt.subplot(gs[0:2, 0])]
            ax += [plt.subplot(gs[2, 0])]
            if plotcurrent and plotpotential:
                ax += [ax[1].twinx()]
    
    object_file = False
    if type(scan) is dict:
        data = scan.copy()
    else:
        data = scan.data.copy()
        object_file = True
    
    
    if tspan is None:                  #then use the range of overlap                            
        tspan = data['tspan'] 
    if type(tspan) is str and not tspan=='all':
        tspan = data[tspan]
    if type(logplot) is not list:
        logplot = [logplot, False]
    
    if t_str is None:
        t_str = 'time/s'
    if V_str is None or J_str is None or RE_vs_RHE is not None or A_el is not None: 
        V_str_0, J_str_0 = sync_metadata(data, RE_vs_RHE=RE_vs_RHE, A_el=A_el, verbose=verbose) 
        #added 16J27... problem caught 17G26, fixed in sync_metadata
    if V_str is None: #this way I can get it to plot something other than V and J.
        V_str = V_str_0
    if J_str is None:
        J_str = J_str_0
    
    A_el = data['A_el']
    
    if object_file and plot_type == 'heat':
            scan.heat_plot(ax=ax[0], tspan=tspan, bg=bg, tthspan=tthspan)
    elif plot_type in ['timescan', 'peaks', 'integrals']:
        if object_file and peaks == 'existing':
            peaks = dict([(n, p[1]) for n, p in scan.peaks.items()])
        plot_simple(data, ax=ax[0], peaks=peaks, tspan=tspan, logplot=logplot[0])
    
            
    ax[0].set_xlim(tspan)
    ax[0].xaxis.tick_top()
    ax[0].xaxis.set_label_position('top')  
    ax[0].tick_params(axis='both', direction='in')    

    if title is not None:
            plt.title(title)
    
    try:
        t = data[t_str]       
    except KeyError:
        print('data doesn\'t contain \'' + str(t_str) + '\', i.e. t_str. Can\'t plot EC data.')
        plotpotential = False
        plotcurrent = False
    try:
        V = data[V_str]
    except KeyError:
        print('data doesn\'t contain \'' + str(V_str) + '\', i.e. V_str. Can\'t plot that data.')
        plotpotential = False      
    try:
        J = data[J_str]      
    except KeyError:
        print('data doesn\'t contain \'' + str(J_str) + '\', i.e. J_str. Can\'t plot that data.')
        plotcurrent = False     
        
        # to check if I have problems in my dataset
#    print('len(t) = ' + str(len(t)) + 
#          '\nlen(V) = ' + str(len(V)) + 
#          '\nlen(J) = ' + str(len(J)))
    
    if tspan is not 'all' and plotcurrent or plotpotential:
        mask = np.logical_and(tspan[0]<t, t<tspan[1])
        t = t[mask]
        if plotpotential:
            V = V[mask]
        if plotcurrent:
            J = J[mask]

    i_ax = 1
    if plotpotential:
        ax[i_ax].plot(t, V, color=V_color, label=V_label)
        ax[i_ax].set_ylabel(V_str)
        if len(logplot) >2:
            if logplot[2]:
                ax[i_ax].set_yscale('log')
        xlim = ax[i_ax-1].get_xlim()
        ax[i_ax].set_xlim(xlim)
        ax[i_ax].yaxis.label.set_color(V_color)
        ax[i_ax].tick_params(axis='y', colors=V_color)
        ax[i_ax].spines['left'].set_color(V_color)
        ax[i_ax].tick_params(axis='both', direction='in') #17K28  
        i_ax += 1
        
    if plotcurrent:
        ax[i_ax].plot(t, J, color=J_color, label=J_label)
        ax[i_ax].set_ylabel(J_str)
        ax[i_ax].set_xlabel('time / [s]')
        xlim = ax[i_ax-1].get_xlim()
        ax[i_ax].set_xlim(xlim)
        if logplot[1]: 
            ax[i_ax].set_yscale('log')
        ax[i_ax].yaxis.label.set_color(J_color)
        ax[i_ax].tick_params(axis='y', colors=J_color)
        if i_ax == 2:
            ax[i_ax].spines['right'].set_color(J_color)
        else:
            ax[i_ax].spines['left'].set_color(J_color)
        ax[i_ax].tick_params(axis='both', direction='in')
        
    if plotcurrent or plotpotential:
        ax[1].set_xlabel('time / [s]')
        ax[1].set_xlim(tspan)
    
    if saveit:
        if title == 'default':
            title == data['title'] + '.png'
        figure1.savefig(title)
    
    if fig is None:
        fig = ax[0].get_figure()
    
    if verbose:
        print('function \'plot_experiment\' finished!\n\n')
    
    return fig, ax


def plot_datapoints(integrals, colors, ax='new', label='', X=None, X_str='V',
                    logplot=True, specs={}, Xrange=None):
    '''
    integrals will most often come from functino 'get_datapoitns' in module
    Integrate_Signals
    '''
    if ax == 'new':
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
    if X is None:
        X = integrals[X_str]
        
    for (quantity, color) in colors.items(): 
        # Here I just assme they've organized the stuff right to start with.
        # I could alternately use the more intricate checks demonstrated in
        # DataPoints.plot_errorbars_y
        value = integrals[quantity]
        if type(Xrange) is dict:
            Xrange_val = Xrange[quantity]
        else:
            Xrange_val = Xrange 
        if type(color) is dict:
            plot_datapoints(value, color, ax=ax, logplot=logplot,
                            label=label+quantity+'_', X=X, Xrange=Xrange_val, specs=specs)
        else:
            if type(color) is tuple: #note a list can be a color in rbg
                spec = color[0]
                color = color[1]
                if 'markersize' not in specs:
                    specs['markersize'] = 5
            else:
                spec = '.'
                if 'markersize' not in specs:
                    specs['markersize'] = 15
            #print(quantity + '\n\tvalue=' + str(value) + 
            #        '\n\tcolor=' + str(color) + '\n\tV=' + str(V))
            #print(quantity + ' ' + str(color))
            if Xrange is not None:
                I_keep = np.array([I for (I, X_I) in enumerate(X) if 
                          Xrange_val[0] <= float(np.round(X_I,2)) <= Xrange_val[1]])
                X_plot = np.array(X)[I_keep]
                value_plot = np.array(value)[I_keep]
                #there was a mindnumbing case of linking here.
                #tried fix it with .copy(), but new variable names needed.
            else:
                X_plot = X
                value_plot = value
            ax.plot(X_plot, value_plot, spec, 
                    color=color, label=label+quantity, **specs, )
    if logplot:
        ax.set_yscale('log')
    return ax




def plot_operation(cc=None, t=None, j=None, z=None, tspan=None, results=None,
                   plot_type='heat', ax='new', colormap='inferno', aspect='auto', 
                   unit='pmol/s', dimensions=None, verbose=True):
    if verbose:
        print('\n\nfunction \'plot_operation\' at your service!\n')
    # and plot! 
    if type(cc) is dict and results is None:
        results = cc
        cc = None
    if results is None:
        results = {} #just so I don't get an error later
    if cc is None:
        cc = results['cc']
    if t is None:
        if 't' in results:
            t = results['t']
        elif 'x' in results:
            t = results['x']
        else:
            t = np.linspace(0, 1, np.size(cc, axis=0))
    if z is None:
        if 'z' in results:
            z = results['z']
        elif 'y' in results:
            z = results['y']
        else:
            z = np.linspace(0, 1, np.size(cc, axis=1))
    if j is None:
        if j in results:
            j = results['j']
        else:
            j = cc[0, :]   
    if dimensions is None:
        if 'dimensions' in results:
            dimensions = results['dimensions']
        else:
            dimensions = 'tz'
    
    
    if tspan is None:
        tspan = [t[0], t[-1]]
    if plot_type == 'flux':   
        if ax == 'new':
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
        else:
            ax1 = ax #making a heat map will only work with a new axis.
        ax1.plot(t, j, label='simulated flux')
        ax1.set_xlabel('time / [s]')
        ax1.set_ylabel('flux / [' + unit + ']')
        axes = ax1
        
    elif plot_type == 'heat' or plot_type == 'both':
        if ax == 'new':
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
        elif type(ax) is list: 
            ax1 = ax[0]
        else:
            ax1 = ax
        
        #t_mesh, x_mesh = np.meshgrid(t,x)
        #img = ax1.contourf(t_mesh, x_mesh*1e6, np.transpose(cc,[1,0]), cmap='Spectral', 
        #                   levels=np.linspace(np.min(cc),np.max(cc),100))
        
        # imshow objects seem more versatile than contourf for some reason.
        
        trange = [min(t), max(t)]
        if dimensions[0] == 'x':
            trange = [t*1e3 for t in trange] # m to mm
        zrange = [min(z*1e6), max(z*1e6)]
        
        img = ax1.imshow(np.transpose(cc,[1,0]), 
                         extent=trange[:] + zrange[:],  #have to be lists here!
                         aspect=aspect, origin='lower',
                         cmap = colormap)        
        
#        divider = make_axes_locatable(ax1)
#        cax = divider.append_axes("right", size="5%", pad=0.05)
#https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

        cbar = plt.colorbar(img, ax=ax1)
        cbar.set_label('concentration / [mM]')
        if dimensions[0] == 't':
            ax1.set_xlabel('time / [s]')
        elif dimensions[0] == 'x':
            ax1.set_xlabel('position / [mm]')
        ax1.set_ylabel('position / [um]')
        
#        print('plot_type = ' + plot_type)
        if plot_type == 'both':
            if type(ax) is list:
                ax2 = ax[1]
            else:
                ax2 = ax1.twinx()
            ax2.set_ylabel('flux / [' + unit + ']')
            ax2.plot(t, j, 'k-')
            cbar.remove()
            ax3 = img.figure.add_axes([0.85, 0.1, 0.03, 0.8])
            cbar = plt.colorbar(img, cax=ax3)
            cbar.set_label('concentration / [mM]')
            ax1.set_xlim(tspan)
            print('returning three axes!')
            axes = [ax1, ax2, ax3]
        else:
            axes = [ax1, cbar]
            
    if verbose:
        print('\nfunction \'plot_operation\' finished!\n\n')
    return axes

def set_figparams(figwidth=8,aspect=4/3,fontsize=7,figpad=0.15,figgap=0.08):
    import matplotlib as mpl
    
    #figwidth=8  #figwidth in cm width 20.32cm = 8inches being standard and thesis textwidth being 12.
    #aspect=4/3  #standard is 4/3

    #fontsize=7  #standard is 12.0pt, thesis is 10.0pt and footnotesize is 8.0pt and lower case seems to be 2/3 of full font size, which makes 7pt "nice" for thesis plotting
    
    realfigwidth=20*(fontsize/12)*1.2  #a factor 1.2 makes all lines "equaly thick" - make this 1 for figwidth=4cm (sniff2fig6)
    #figsize=[20.32,15.24]
    figsize=[realfigwidth,realfigwidth/aspect]
    
    mpl.rc('font', size=fontsize*(realfigwidth/figwidth))
    
    mpl.rc('mathtext', fontset='custom',
                       rm='Helvetica',
                       #it='Helvetica:italic',
                       #bf='Helvetica:bold',
                       )
    
    mpl.rc('figure', figsize=[figsize[0]/2.54,figsize[1]/2.54],
                     dpi=100*2.54*figwidth/realfigwidth
                     )
    
    #figpad=0.14  #fraction of figure size
    #figgap=0.08  #fraction of figure size
    mpl.rc('figure.subplot', left=figpad,
                             right=1-figpad,
                             bottom=figpad,
                             top=1-figpad,
                             hspace=figgap,
                             )
    mpl.rc('xtick', labelsize='small') 
    mpl.rc('ytick', labelsize='small')
    
    #mpl.rc('axes', labelweight='medium')
    
    mpl.rc('savefig', dpi=250*2.54*figwidth/realfigwidth)
    
    
    
    
    
