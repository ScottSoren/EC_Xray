#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 05:07:10 2017

@author: scott
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def extended_call(Obj, fun, **kwargs):
    if type(Obj) is dict:
        results = {}
        for key, value in Obj.items():
            results[key] = extended_call(value, fun, **kwargs)
        return results
    elif type(Obj) is list:
        return [getattr(o, fun)(**kwargs) for o in Obj]
    else:
        return getattr(Obj, fun)(**kwargs)


def gauss(x, center, sigma, height):
    y = height * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return y


def interp_with_zeros(x, x_sub, y_sub):
    """
    Turns out this handy little function is covered by numpy. It is equivalent
    to np.interp(x, x_sub, y_sub, left=0, right=0)
    """
    y = np.zeros(np.size(x))
    mask = np.logical_and(x_sub[0] < x and x < x_sub[-1])
    y[mask] = np.interp(x[mask], x_sub, y_sub)
    return y


def integrate_peak(
    x,
    y,
    xspan,
    background="linear",
    background_type="local",
    background_points=4,
    ax=None,
    color="k",
    fill_color="g",
    returnax=True,
):
    if ax == "new":
        fig, ax = plt.subplots()

    # select data in range of peak
    #    print('x = ' + str(x) + '\nxspan = ' + str(xspan)) # debugging
    try:
        i_start = next(i for i, x_i in enumerate(x) if x_i > xspan[0])
        i_finish = next(i for i, x_i in enumerate(x) if x_i > xspan[-1])
    except StopIteration:
        print("couldn't integrate peak with xspan=" + str(xspan) + " : out of range.")
        return 0
    peak_x = np.array(x[i_start:i_finish])
    peak_y = np.array(y[i_start:i_finish])

    # parse inputs and get background
    if background is None or background is False:
        background = np.zeros(np.shape(peak_y))
    elif background_type == "local":
        background = get_peak_background(
            x,
            y,
            xspan,
            background=background,
            background_points=background_points,
            mode="peak",
        )
    elif type(background) is np.ndarray:
        if background.shape == y.shape:  # we only want the background where the peak is
            background = background[i_start:i_finish]

    # integrate background
    I = np.trapz(peak_y - background, peak_x)

    if ax is not None:
        if color is not None:
            ax.plot(peak_x, peak_y, color=color)
            ax.plot(peak_x, background, "--", color=color)
        if fill_color is not None:
            ax.fill_between(
                peak_x,
                peak_y,
                background,
                where=peak_y > background,
                facecolor=fill_color,
                interpolate=True,
            )
        if returnax:
            return I, ax
    return I


def get_peak_background(
    x,
    y,
    xspan=None,
    background="linear",
    background_points=3,
    mode="full",
    ax=None,
    color="k",
    fill_color="g",
):
    # print('calculating background') # debugging
    if xspan is None:
        i_start, i_finish = background_points, len(x) - background_points
        xspan = x
    else:
        i_start = next(i for i, x_i in enumerate(x) if x_i > xspan[0])
        i_finish = next(i for i, x_i in enumerate(x) if x_i > xspan[-1])

    y_prepeak = np.mean(y[i_start - background_points : i_start])
    y_postpeak = np.mean(y[i_finish : i_finish + background_points])

    # print(f'xspan = {xspan}, y_prepeak={y_prepeak}, y_postpeak={y_postpeak}') # debugging
    peak_x = np.array(x[i_start:i_finish])
    background = (
        (peak_x - xspan[0]) * y_postpeak + (xspan[-1] - peak_x) * y_prepeak
    ) / (xspan[-1] - xspan[0])
    if (
        mode == "match"
    ):  # return a background equal to the input outside the peak interval
        background = np.append(np.append(y[0:i_start], background), y[i_finish:])
    elif mode == "full":
        background = ((x - xspan[0]) * y_postpeak + (xspan[-1] - x) * y_prepeak) / (
            xspan[-1] - xspan[0]
        )

    elif mode == "peak":  # return a background only for the peak interval
        background = background
    return background


def get_background_line(
    spectrum,
    method="endpoint",
    floor=True,
    N_end=3,
    steps0=2,
    lincutoff=True,
    p1=0.1,
    p2=0.4,
    name="'name'",
    out="values",
    verbose=False,
):
    """
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
    given a gaussian distribution with the measured standard deviation of
    having one outlier is less than p1.
    If floor=True, no outlying point lying below the line can be removed.
    
    ---------
    The function returns either
    out = 'line' : the [slope, intercept] of the background line
    out = 'values' : the linear background values corresponding to x
    """

    from scipy.stats import norm

    x, y = spectrum
    N = len(x)

    if method == "endpoint":
        again = True
        if type(steps0) is int:
            steps0 = [steps0, steps0]
        N_left, N_right = N_end * steps0[0], N_end * steps0[-1]  # cutoff region
        z = norm.ppf(0.5 + p1)  # max acceptable standard deviation from mean
        sigma = np.sqrt(N * p2 * (1 - p2))  # standard deviation of number below
        max_below = N * p2 + sigma * z
        steps = steps0
        while again:
            left = np.arange(steps[0] * N_end, (steps[0] + 1) * N_end)
            right = np.arange(N - (steps[1] + 1) * N_end, N - steps[1] * N_end)
            both = np.append(left, right)
            try:
                x_ends, y_ends = x[both], y[both]
            except IndexError:
                print(
                    "Couldn't find background endpoints meeting your "
                    + " demands for "
                    + name
                    + ". steps = "
                    + str(steps)
                )
                raise
            poly = np.polyfit(x_ends, y_ends, 1)
            bg = poly[0] * x + poly[1]
            if lincutoff:  # bg drops linearly to zero in cutoff regions:
                bg[:N_left] = np.linspace(0, bg[N_left], num=N_left)
                bg[-N_right:] = np.linspace(bg[-N_right], 0, num=N_right)
            else:  # everything in cutoff up to linear bg is bg
                bg[:N_left] = np.min(
                    np.stack([y[:N_left], np.tile(bg[N_left], (N_left,))]), axis=0
                )
                bg[-N_right:] = np.min(
                    np.stack([y[-N_right:], np.tile(bg[-N_right], (N_right,))]), axis=0
                )
            if floor:
                res_ends = y_ends - bg[both]
                std = np.std(res_ends)
                N_below = np.sum(y < bg - std * z)
                again = N_below > max_below
                if steps[0] == steps0[0]:
                    if verbose:
                        print("moving left endpoint way in to find good background")
                    steps = [steps[1] + 1, steps0[1]]
                else:
                    steps = [steps[0] - 1, steps[1] + 1]
                    if verbose:
                        print(
                            "moving right endpoint in and left out to find good background"
                        )
            else:
                again = False
    else:
        print(
            "get_background_line(method='"
            + method
            + "'...) "
            + "not implemented! Using method = 'endpoint'"
        )
        return get_background_line(
            spectrum,
            method="endpoint",
            floor=floor,
            N_end=N_end,
            p1=p1,
            p2=p2,
            out=out,
            verbose=verbose,
        )
    if out == "line":
        return poly
    else:
        return bg


class Peak:
    def __init__(self, x, y, xspan=None, name=None, color="k"):
        self.name = name
        self.xspan = xspan
        self.color = color
        if xspan is not None:
            mask = np.logical_and(xspan[0] < x, x < xspan[-1])
            x, y = x[mask], y[mask]
        self.x = x
        self.y = y
        self.background = np.zeros(x.shape)
        self.bg = False

    def set_background(self, background):
        if not len(background) == len(self.x):
            print("background wrong length.")
            raise ValueError
        self.background = background
        self.bg = True

    def get_background(self, *args, **kwargs):
        x, y = self.x, self.y
        self.background = get_peak_background(x, y, **kwargs)
        self.bg = True
        return self.background

    def get_integral(self, *args, **kwargs):
        if "mode" in kwargs and kwargs["mode"] in ["gauss", "fit"]:
            if "ax" in kwargs:
                self.fit_gauss(ax=kwargs["ax"])
            return self.integral_f
        x, y, background = self.x, self.y, self.background
        integral = np.trapz(y - background, x)
        self.integral = integral
        if "ax" in kwargs:
            ax = kwargs["ax"]
            if ax == "new":
                fig, ax = plt.subplots()
            if ax is not None:
                ax.plot(x, y, "k.")
                ax.plot(x, background, "b--")
                ax.fill_between(x, background, y, where=y > background, color="g")
        return integral

    def fit_gauss(self, center=None, sigma=None, ax=None):
        x, y, background = self.x, self.y, self.background
        y = y - background

        guess_c = (x[-1] + x[0]) / 2
        guess_s = (x[-1] - x[0]) / 2
        guess_h = max(y)

        if center is not None and sigma is not None:

            def gauss_i(x, height):
                return gauss(x, center=center, sigma=sigma, height=height)

            guess = guess_h
            popt, pcov = curve_fit(gauss_i, x, y, p0=guess)
            height = popt[0]
        elif center is not None:

            def gauss_i(x, sigma, height):
                return gauss(x, center=center, sigma=sigma, height=height)

            guess = [guess_s, guess_h]
            popt, pcov = curve_fit(gauss_i, x, y, p0=guess)
            sigma, height = popt[0], popt[1]
        elif sigma is not None:

            def gauss_i(x, center, height):
                return gauss(x, center=center, sigma=sigma, height=height)

            guess = [guess_c, guess_h]
            popt, pcov = curve_fit(gauss_i, x, y, p0=guess)
            center, height = popt[0], popt[1]
        else:

            def gauss_i(x, center, sigma, height):
                return gauss(x, center=center, sigma=sigma, height=height)

            guess = [guess_c, guess_s, guess_h]
            try:
                popt, pcov = curve_fit(gauss_i, x, y, p0=guess)
                center, sigma, height = popt[0], popt[1], popt[2]
            except RuntimeError:
                center, sigma, height = guess
        sigma = abs(sigma)
        # print(f'center={center}, sigma={sigma}, height={height}') # debugging
        fit = gauss(x, center, sigma, height)
        integral_f = np.sqrt(2 * np.pi) * height * sigma
        self.center, self.sigma, self.height = center, sigma, height
        self.fit, self.integral_f = fit, integral_f

        if ax is not None:
            if ax == "new":
                fig, ax = plt.subplots()
            ax.plot(x, background, "b--")
            ax.plot(x, y + background, "k.")
            ax.plot(x, fit + background, "r--")

        return center, sigma, height

    def get_dQ(self, wavelength=None, E=None):
        if wavelength is None:
            if E is None:
                E = self.E  # photon energy in eV
            h = 6.6260693e-34  # planks constant / (J*s)
            c = 299792458  # speed of light / (m/s)
            qe = 1.602176e-19  # elementary charge / C
            wavelength = h * c / (E * qe)  # wavelength / (m)
        k = 2 * np.pi / wavelength

        theta = self.center / 2
        Theta = theta * np.pi / 180
        dtheta = self.sigma / 2
        dTheta = dtheta * np.pi / 180

        dQ = 2 * k * np.cos(Theta) * dTheta
        return dQ

    def get_grain_size(self, wavelength=None, E=None, shape_factor=0.9):
        if wavelength is None:
            if E is None:
                E = self.E  # photon energy in eV
            h = 6.6260693e-34  # planks constant / (J*s)
            c = 299792458  # speed of light / (m/s)
            qe = 1.602176e-19  # elementary charge / C
            wavelength = h * c / (E * qe)  # wavelength / (m)

        theta = self.center / 2
        Theta = theta * np.pi / 180
        fwhm = 2 * np.sqrt(2 * np.log(2)) * self.sigma
        FWHM = fwhm * np.pi / 180
        r0 = shape_factor * wavelength / (FWHM * np.cos(Theta))
        return r0


def grain_size(fwhm, theta, wavelength=7.293e-11, shape_factor=0.9):
    Theta = theta * np.pi / 180
    FWHM = fwhm * np.pi / 180
    tau = shape_factor * wavelength / (FWHM * np.cos(Theta))
    return tau


def refraction_correction(
    alpha=0.15, delta_eff=5.94e-6, beta_eff=2.37e-7, alpha_c=0.197, tth=None
):

    if alpha_c is None:
        Alpha_c = np.sqrt(2 * delta_eff)
    else:
        Alpha_c = np.pi / 180 * alpha_c
    if delta_eff is None:
        delta_eff = Alpha_c ** 2 / 2

    Alpha = np.pi / 180 * alpha
    Beta = beta_eff
    delta_tTh = Alpha - 1 / np.sqrt(2) * np.sqrt(
        np.sqrt(
            (Alpha ** 2 - Alpha_c ** 2 + Alpha ** 2 * Alpha_c ** 2 / 2) ** 2
            + (-2 * Beta + Alpha ** 2 * Beta) ** 2
        )
        - Alpha_c ** 2
        + Alpha ** 2 * Alpha_c ** 2 / 2
        + Alpha ** 2
    )

    delta_tth = 180 / np.pi * delta_tTh

    if tth is None:
        return delta_tth
    else:
        return tth - delta_tth
