"""
Module for the analysis of Markov Chains.
"""
import os

from math import *
import numpy as np
import numpy as n
import pickle
import gzip
import scipy
import string
import glob

from .Objects_MCMC import Chain
from .priors import prior_constructor, compute_priors


def confidence_intervals(C, q=0.6827, hdi=None, nbins=50, burnin=0.0,
                         percentile=True, cumulative=False, difftol=0.3,
                         pmsym='+/-', report='mean', output=False):
    """
    Computes confidence intervals for all parameters in a given chain, and
    print it to the standard output.

    Parameters
    ----------
    C, Chain instance, VDchain instance or dictionary
       The chain for which the confidence intervals are to be computed. An
       actual Chain instance, or a dictionary containing all the traces for
       the chain can be given.

    Other parameters
    ----------------
    q, float
        Fraction of entire distribution to use for the computation of the
        vals (default: 0.6827, i.e. 68.27% confidence interval, 1-sigma).

    hdi, float or list
        The fraction of the mass of the distribution to include in the
        Highest Denstity Interval. It can be a float or an iterable object
        with a series of values. If None, no hdi is computed
    
    nbins, int
        The number of bins used for the computation. If negative, the
        computation is done using the absolute value of nbins for the mode,
        but using the cumulative distribution for the confidence intervals.

    burnin, float
        The fraction to be discarded from the beginning of each trace. Useful
        if the trace contain the burn-in period.

    percentile, boolean
        Defines whether confidence intervals are obtained using the percentile
        function instead of interpolating histogram bins.

    cumulative, boolean
        Defines if instead of returning median and confidence limits, return
        q*100% cumulative value.
        
    difftol, float
        Maximum fractional difference between upper and lower limits above which
        both values are printed. If the difference is below this value, print
        only the largest of both values.

    pmsym, string
        Symbol to use for +/-. It can be used to include latex code

    report, string
        Controls which statistics is reported. It can be 'mean', 'median',
        'mode' or 'map'.

    output, boolean
        Decides whether the value and confidence intervals are returned in a
        dictionary or not.
        
    """
    if isinstance(C, Chain):
        vd = C.get_value_dict()
    elif isinstance(C, dict):
        vd = C
    else:
        print('Warning! Instance not recognized. Assuming it is VDchain '
              'instance.')
        try:
            vd = C.get_value_dict()
        except NameError:
            raise NameError('Failed to load posterior sample. Check input.')

    # Try to compute index of MAP
    try:
        mapindex = n.argmax(vd['posterior'])
    except KeyError:
        mapindex = n.nan
        if report is 'map':
            print('Posterior not given, will return posterior mode.')
            report = 'mode'

    if cumulative:
        # Minimum and maximum prob
        qmin = 0
        qmax = q

    else:
        # Minimum and maximum prob
        qmin = (1 - q) * 0.5
        qmax = (1 + q) * 0.5

    # Initialise output dictionary.
    outputdict = {}

    # Find index after burn in.
    istart = int(len(list(vd.values())[0]) * burnin)

    for param in n.sort(list(vd.keys())):

        # Get median and mode, and dispersion
        statdict = {'median': n.median(vd[param][istart:]),
                    'mean': n.mean(vd[param][istart:]),
                    'sigma': n.std(vd[param][istart:]),
                    'min': n.min(vd[param][istart:]),
                    'max': n.max(vd[param][istart:])
                    }

        # Try to add MAP value
        try:
            statdict['map'] = vd[param][mapindex]
        except IndexError:
            statdict['map'] = n.nan

        # Compute histogram for all cases that require it.
        if report is 'mode' or not percentile or hdi is not None:
            # Compute histogram
            m, bins = n.histogram(vd[param][istart:], nbins, normed=True)
            x = bins[:-1] + 0.5 * n.diff(bins)

            modevalue = x[n.argmax(m)]

            statdict['mode'] = modevalue

        ###
        # Find confidence intervals
        ###
        if percentile:
            # Method independent of bin size.
            lower_limit = n.percentile(vd[param][istart:], 100 * qmin)
            upper_limit = n.percentile(vd[param][istart:], 100 * qmax)

        else:
            # Original method
            ci = m.astype(float).cumsum() / n.sum(m)

            if not cumulative:
                imin1 = n.argwhere(n.less(ci, qmin))

                if len(imin1) >= 1:
                    imin1 = float(imin1.max())
                else:
                    imin1 = 0.0

                imin2 = imin1 + 1.0
                lower_limit = scipy.interp(qmin, [ci[imin1], ci[imin2]],
                                           [x[imin1], x[imin2]])
            else:
                lower_limit = 0.0

            imax1 = float(n.argwhere(n.less(ci, qmax)).max())
            imax2 = imax1 + 1.0
            upper_limit = scipy.interp(qmax, [ci[imax1], ci[imax2]],
                                       [x[imax1], x[imax2]])

        try:
            reportvalue = statdict[report]
        except KeyError:
            raise NameError('report statistics not recognised.')

        hc = upper_limit - reportvalue
        lc = reportvalue - lower_limit

        statdict['{:.2f}-th percentile'.format(qmin * 100)] = lower_limit
        statdict['loweruncertainty'] = lc
        statdict['{:.2f}-th percentile'.format(qmax * 100)] = upper_limit
        statdict['upperuncertainty'] = hc

        # ## Trick to show good number of decimal places.
        """
        # Express hc and lc in scientific notation.
        s = '{:e}'.format(min(hc, lc))
        # Find the exponent in scientific notation
        a = s[s.find('e') + 1:]
        """
        # Express hc and lc in scientific notation.
        s = '{:e}'.format(statdict['sigma'])
        # Find the exponent in scientific notation
        a = s[s.find('e') + 1:]  # if negative, the number of decimals is equal to the exponent + 1
        if a[0] == '-':
            ndecimal = int(a[1:]) + 1
        # Else, just one decimal place.
        else:
            ndecimal = 1
        # print('sigma {}: {}, {}'.format(param, statdict['sigma'], ndecimal))

        statdict['ndec'] = int(ndecimal)
        # If upper and lower errors do not differ by more than difftol*100%
        # only report one of them
        maxerr = max(hc, lc)

        if cumulative:

            formatdict = {'param': param, 'value': upper_limit,
                          'ndec': int(ndecimal), 'pmsym': pmsym,
                          'err': maxerr, 'q': qmax * 100}

            formatdict.update(statdict)

            print('{param} ({q}-th percentile): {value:.{ndec}f} '
                  '(mean = {mean:.{ndec}f}, '
                  'median = {median:.{ndec}f}, '
                  'map = {map:.{ndec}f}, '
                  'min = {min:.{ndec}f}, '
                  'max = {max:.{ndec}f}, '
                  'sigma = {sigma:.1e})'.format(**formatdict))

        else:
            if abs(hc - lc) / maxerr < difftol:

                formatdict = {'param': param, 'value': reportvalue,
                              'ndec': int(ndecimal), 'pmsym': pmsym,
                              'err': maxerr}

                formatdict.update(statdict)
                print('{param}: {value:.{ndec}f} {pmsym} {err:.1e} '
                      '(mean = {mean:.{ndec}f}, '
                      'median = {median:.{ndec}f}, '
                      'map = {map:.{ndec}f}, '
                      'min = {min:.{ndec}f}, '
                      'max = {max:.{ndec}f}, '
                      'sigma = {sigma:.1e})'.format(**formatdict))

            else:
                hcr = n.round(hc, ndecimal)
                lcr = n.round(lc, ndecimal)
                formatdict = {'param': param, 'value': reportvalue,
                              'ndec': int(ndecimal), 'pmsym': pmsym,
                              'err+': hcr, 'err-': lcr}

                formatdict.update(statdict)

                print('{param}: {value:.{ndec}f} +{err+:.1e} -{err-:.1e} '
                      '(mean = {mean:.{ndec}f}, '
                      'median = {median:.{ndec}f}, '
                      'map = {map:.{ndec}f}, '
                      'min = {min:.{ndec}f}, '
                      'max = {max:.{ndec}f}, '
                      'sigma = {sigma:.1e})'.format(**formatdict))

        # Compute HDI
        if hdi is not None:
            hdi = n.atleast_1d(hdi)

            for hdii in hdi:
                hdints = compute_hdi(bins, m, q=hdii)

                statdict['{}%-HDI'.format(hdii*100)] = hdints

                hdistr = '{0:.1f}% HDI: '.format(hdii * 1e2)
                for jj, interval in enumerate(hdints):
                    if jj > 0:
                        hdistr += ' U '

                    if n.isnan(interval[0]):
                        continue
                    hdistr += (
                        '[{0:.' + str(ndecimal + 1) + 'f}, {1:.' + str(
                            ndecimal + 1) + 'f}]').format(*interval)
                print(hdistr)

        if output:
            outputdict[param] = statdict

    if output:
        return outputdict
    else:
        return


def autocorr(x, lags):
    """
    Compute autocorrelation of x at elements of lag.
    
    :param array-like x: vector for which to compute autocorrelation
    :param iterable lags: lags for which to compute auto-correlation.
    """
    lags = np.atleast_1d(lags)
    x = np.array(x)
    
    # Compute means that are used at each lag
    xmean = x.mean()
    x2mean = (x**2).mean()

    corr = np.zeros_like(lags, dtype='float')

    for i, lag in enumerate(lags):
        # corr[i] = corr_onelag(x, lag, xmean2=xmean**2, x2mean=x2mean)
        corr[i] = corr_onelag(x, lag, xmean=xmean)
    return corr

    
def corr_onelag_tegmark(x, lag, x2mean=None, xmean2=None):
    """
    Compute autocorrelation of x at a single lag.
    Assumes elements in x are equally spaced.
    Give x2mean and xmean to reduce computation time in an interative frame.

    :param array-like x: vector for which to compute autocorrelation
    :param int lag: lag for which to compute auto-correlation.

    Other params
    ------------
    :param float xmean2: square of mean of x over entire vector.
    :param float x2mean: mean of x**2 over entire vector.
    """
    x = np.array(x)
    
    if xmean2 is None:
        xmean2 = x.mean()**2

    if x2mean is None:
        x2mean = (x**2).mean()

    # Shift array, use circular contour conditions.
    xs = cshift(x, lag)

    # Compute mean over shifted version of array
    return ((xs * x).mean() - xmean2) / (x2mean - xmean2)
    

def corr_onelag(x, lag, xmean=None):
    x = np.array(x).copy()

    if xmean is None:
        xmean = x.mean()

    # Remove mean from x
    x -= xmean
    return np.sum(x[:-lag]*x[lag:]) / np.sum(x**2)


def corrlength(x, step=1, BI=0.2, BO=1.0, widget=False, verbose=True,
               plot=False, **kwargs):
    """
    Computes the correlation length of a given trace of a Parameter

    The value of the shift for which the correlation reaches 1/e is printed.

    Parameters
    ----------
    x: ndarray
        An array containing the elements of the trace.

    step: int
        The number of elements to shift x at each step of the computation.

    BI: float
        The fraction of the chain to be discarded for the computation of the
        correlation length. This is used if the traces
        contain the burn-in period.

    BO: float
        The fraction up to which the chain is considered. To use in combination
        of BI. E.g.: a BI of 0.2 and a BO of 0.6 imply that the correlation is
        computed over 40 % of the chain.
    
    widget: boolean
        To display or not the printed information

    plot: boolean
        To perform a plot of the correlation length

    Other parameters
    ----------------
    xlabel: str
        Label for x axis of plot (default: 'Step')

    ylabel: str
        Label for y axis of plot (default: 'Correlation')

    title: str
        Title for plot (default: '')
        
    circular: bool
        Assumes the time series have simmetry, so that when shifted, the last
        part goes to the beginning. This saves time.

    stop: bool
        Defines whether the computation stops when the correlation has fallen
        below 1/e or if it continues to the end.
        
    Returns
    -------
    shifts: ndarray
        The shift values for which the computation was done.
	
    corr: ndarray
        The correlation value for each step.

    corrlength: int
        The number of steps after which the correlation has fallen below 1/e.

    Notes
    -----
    To reduce computation of useless values, the algorithm is stopped when the
    last 500 values of the correlation are below 0.2.
    """

    xlabel = kwargs.pop('xlabel', 'Step')
    ylabel = kwargs.pop('ylabel', 'Correlation')
    title = kwargs.pop('title', '')
    circular = kwargs.pop('circular', True)
    stop = kwargs.pop('stop', True)

    if widget == True:
        verbose = False

    step = int(step)
    
    indstart = n.int(len(x) * BI)
    indend = n.int(len(x) * BO)
    x = x[indstart: indend]

    ## Reduce mean
    x = x - x.mean()

    if circular:
        ## Compute values of unchanged chain only once
        xmean = x.mean()
        x2mean = (x ** 2).mean()
        xx2 = xmean ** 2.0
        den = x2mean - xmean ** 2.0
    #
    shifts = n.zeros(len(x)/step)
    corr = n.zeros(len(x)/step)
    for j in range(len(corr)):
        #
        if circular:
            xs = cshift(x, j * step)
            corr[j] = ((x * xs).mean() - xx2) / den
        else:
            ## Compute for each iteration
            if j == 0:
                corr[j] = 1
            else:
                xs = x[:-j * step]
                xmean = x[j * step:].mean()
                x2mean = (x[j * step:] ** 2).mean()
                xx2 = xmean ** 2.0
                den = x2mean - xmean ** 2.0

                corr[j] = ((x[j * step:] * xs).mean() - xx2) / den

        shifts[j] = j * step

        #
        if (j + 1) % 100 == 0 and verbose:
            print('Step {} out of a maximum of {}'.format(j+1, len(corr)))
            os.sys.stdout.flush()

        if j > 500.0 and stop:
            # Stop iteration if last 500 points are below 1/e
            if n.alltrue(n.less(corr[j + 1 - 500: j + 1], 1.0 / e)):
                break

    shifts, corr = shifts[:j + 1], corr[:j + 1]
    try:
        corrlength = n.min(n.compress(corr < 1.0 / e, shifts))
    except ValueError:
        corrlength = len(x)
        print('Error! Correlation length not found.')
    else:
        if verbose: print('Correlation drops to 1/e after %d steps' %
                          corrlength
                          )

    if plot:
        import pylab as p

        fig1 = p.figure()
        ax = fig1.add_subplot(111)
        ax.plot(shifts, corr)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=16)
        ax.axhline(1 / e, ls=':', color='0.5')
        ax.axvline(corrlength, ls=':', color='0.5')
        p.draw()

    return shifts, corr, corrlength


def corrlength2(x, step=1, BI=0.2, widget=False):
    """
    Computes the correlation length of a given trace of a Parameter

    The value of the shift for which the correlation reaches 1/e is printed.

    Parameters
    ----------
    x: ndarray
        An array containing the elements of the trace.

    step: int
        The number of elements to shift x at each step of the computation.

    BI: float
        The fraction of the chain to be discarded for the computation of the
        correlation length. This is used if the traces
        contain the burn-in period.
    
    widget: boolean
        To display or not the printed information

    Returns
    -------
    shifts: ndarray
        The shift values for which the computation was done.
	
    corr: ndarray
        The correlation value for each step.

    corrlength: int
        The number of steps after which the correlation has fallen below 1/e.

    Notes
    -----
    To reduce computation of useless values, the algorithm is stopped when the
    last 500 values of the correlation are below 0.2.
    """
    x = x - x.mean()

    indstart = n.round(len(x) * BI)

    x = x[indstart:]

    xmean = x.mean()
    x2mean = (x ** 2).mean()
    xx2 = xmean ** 2.0
    den = x2mean - xmean ** 2.0
    #
    shifts = []  # n.zeros(len(x)/float(step))
    corr = []  # n.zeros(len(x)/float(step))
    j = 0
    while j < len(x) / float(step):
        # for j in range(len(corr)):
        #
        xs = cshift(x, j * step)

        shifts.append(j * step)
        corr.append(((x * xs).mean() - xx2) / den)
        #
        if (j + 1) % 100 == 0 and not widget:
            print('Step {} out of a maximum of {}'.format(j+1, len(corr)))
            os.sys.stdout.flush()

        if j > 500.0:
            # Stop iteration if last 500 points are below 0.2
            if n.alltrue(n.less(corr[j + 1 - 500: j + 1], 0.2)): break

        j += 1

    shifts = n.array(shifts)
    corr = n.array(corr)

    # shifts, corr = shifts[:j + 1], corr[:j + 1]   # solve the problem with append ?
    corrlength = n.min(n.compress(corr < 1.0 / e, shifts))
    if not widget: print('Correlation drops to 1/e after %d steps' %
                         n.min(n.compress(corr < 1.0 / e, shifts))
                         )

    return shifts, corr, corrlength


def cshift(x, j):
    j %= len(x)
    return n.concatenate((x[j:], x[:j]))


def corrlength_multichain(vds, step=1, BI=0.2, plot=False, widget=False,
                          verbose=True, plotCL=False, **kwargs):
    """
    Compute correlation lenght for all parameters of a multichain

    Parameters
    ----------
    vds: list
        a list of the value dictionaries of each chain. See Chain.get_value_dict()
    
    step: int
        The number of elements to shift x at each step of the computation.

    BI: float or list
        The fraction of the chain to be discarded for the computation of the
        correlation length.
	An iterable object containing the fraction to be discarded for each
	individual chain to be merged can also be given. In this case, the
	number of elements in BI must equal that in vds.
        
    plot: bool
        plot the chain correlation lenght for each parameter
        
    plotCL: bool
        plot the chain correlation curve  for each parameter of each chain.

    widget, boolean
        option that displays a widget status bar

    Other parameters
    ----------------
    The remaining keyword arguments are passed to corrlength function
    
    Returns
    -------
    corrlen: dict
        A dictionary containing the correlation lenght for all chains for each 
        parameter (key)
 
    """
    if widget:
        verbose = False

    import Tkinter
    import MeterBar
    ## Check if it is a list of VDchain instances or of dictionaries
    if isinstance(vds[0], dict):
        vdd = vds[0].copy()
    else:
        # Assume its a VDchain from before reloading the module....
        print('Warning! Class VDchain have changed.')
        vdd = vds[0].get_value_dict()

    ## Create list of starting indexes
    try:
        iter(BI)
    except TypeError:
        # BI is a float; convert to a list
        BI = [BI] * len(vds)

    # Check if BI contains the same number of elements as vds
    if len(BI) != len(vds):
        raise TypeError('BI must be either a float or contain the same number\
of elements as the input list')

    # Create output dictionary
    corrlen = {}
    if widget:
        meterbar = Tkinter.Toplevel()
        m = MeterBar.Meter(meterbar, relief='ridge', bd=3)
        m.pack(fill='x')
        m.set(0.0, 'Computing correlation lenght ...')

    for i, parameter in enumerate(vdd.keys()):  # parameter boucle
        if parameter in ['logL', 'posterior']: continue

        if not widget:
            print('Computing corrlength of ' + parameter)

        corrlengthi = []
        for chain in range(len(vds)):  # chain boucle

            if isinstance(vds[chain], dict):
                vdd = vds[chain].copy()
            else:
                # Assume its a VDchain from before reloading the module....
                raise TypeError('Warning! Input must be dictionary.')

            shifts, corr, corrlengthc = corrlength(vdd[parameter],
                                                   step=step,
                                                   BI=BI[chain],
                                                   widget=widget,
                                                   verbose=verbose,
                                                   plot=plotCL,
                                                   title=parameter,
                                                   **kwargs)

            corrlengthi.append(corrlengthc)
            if widget: m.set(
                float(i) / len(vdd.keys()) + float(chain + 1) / len(
                    vdd.keys()) / len(vds), parameter)

        if widget: m.set(float(i + 1) / len(vdd.keys()), parameter)

        corrlen[(parameter)] = corrlengthi

    if widget:
        m.set(1., 'Correlation lenght computed')
        meterbar.destroy()

    if plot:
        import pylab as p

        for parameter in corrlen.keys():
            p.figure()
            p.plot(corrlen[(parameter)], 'o')
            p.xlabel('chain index')
            p.ylabel(parameter)

    if plotCL:
        import pylab as p

        p.show()

    return corrlen


def corrlength_multichain2(vds, step=1, BI=0.2, plot=False, widget=False):
    """
    Compute correlation lenght for all parameters of a multichain

    Parameters
    ----------
    vds: list
        a list of the value dictionaries of each chain. See Chain.get_value_dict()
    
    step: int
        The number of elements to shift x at each step of the computation.

    BI: float
        The fraction of the chain to be discarded for the computation of the
        correlation length. This is used if the traces
	contain the burn-in period.
        
    plot: bool
        plot the chain correlation lenght for each parameter
        
    widget, boolean
        option that displays a widget status bar

    Returns
    -------
    corrlen: dict
        A dictionary containing the correlation lenght for all chains for each 
        parameter (key)
 
    """
    import Tkinter
    import MeterBar
    ## Check if it is a list of VDchain instances or of dictionaries
    if isinstance(vds[0], dict):
        vdd = vds[0].copy()
    else:
        print(type(vds[0]))

    ## Create list of starting indexes
    try:
        iter(BI)
    except TypeError:
        # BI is a float; convert to a list
        BI = [BI] * len(vds)

    # Create output dictionary
    corrlen = {}
    if widget:
        meterbar = Tkinter.Toplevel()
        m = MeterBar.Meter(meterbar, relief='ridge', bd=3)
        m.pack(fill='x')
        m.set(0.0, 'Computing correlation lenght ...')

    for i, parameter in enumerate(vdd.keys()):  # parameter boucle
        if parameter == 'logL': continue
        if not widget:
            print('--------------------')
            print('    ' + parameter)
            print('--------------------')
        corrlengthi = []
        for chain in range(len(vds)):  # chain boucle

            if isinstance(vds[chain], dict):
                vdd = vds[chain].copy()
            else:
                if not widget: print(type(vds[chain]))

            shifts, corr, corrlengthc = corrlength2(vdd[parameter],
                                                    step=step, BI=BI[chain],
                                                    widget=widget)
            corrlengthi.append(corrlengthc)
            if widget: m.set(
                float(i) / len(vdd.keys()) + float(chain + 1) / len(
                    vdd.keys()) / len(vds), parameter)

        if widget: m.set(float(i + 1) / len(vdd.keys()), parameter)

        corrlen[(parameter)] = corrlengthi

    if widget:
        m.set(1., 'Correlation lenght computed')
        meterbar.destroy()

    if plot:
        import pylab as p

        for parameter in corrlen.keys():
            p.figure()
            p.plot(corrlen[(parameter)], 'o')
            p.xlabel('chain index')
            p.ylabel(parameter)

    return corrlen


def corrlenchain(corrlen):
    """
    Compute the maximum correlation length among all parameters of a chain

    Parameters
    ----------
    corrlen: dict
        A dictionary containing the correlation lenght for all chains for each 
        parameter (key)

    Returns
    -------
    cl: list
        Maximum correlation length among all parameters of a chain.
        To use as CL keyword in merge_chains.
    """

    # get correlation length values and put into an array
    acorr = n.array(corrlen.values())

    # reshape (all correlation length values for each chain instead of for each parameter)
    acorr2 = acorr.reshape(len(corrlen.keys()), len(corrlen[corrlen.keys()[0]]))

    # get maximum correlation length in each chain
    cl = n.max(acorr2, axis=0)

    return cl


def checkpriors(vdc, pastisfile, **kargs):
    """
    Plot histogram of each parameter of the merged chain together with 
    its prior taken from a .pastis configuration file 

    Parameter
    ---------
    vdc: dict
        The merged chain, output of the merge_chains function.

    pastisfile: string
        The name of the .pastis configuration file 
        
    **kargs
        Parameters for the hist function

    """

    f = open(pastisfile, 'r')
    dd = pickle.load(f)
    f.close()

    priordict = prior_constructor(dd[1], dd[3])

    if isinstance(vdc, dict):
        vdd = vdc.copy()
    else:
        print(type(vdc))

    ### PLOTS
    import pylab as p

    for parameter in vdd.keys():
        if parameter != 'logL' and parameter != 'posterior':
            p.figure()
            pdf, bins, patches = p.hist(vdd[parameter], normed=True, **kargs)
            p.xlabel(parameter)
            xmin = n.min(vdd[parameter])
            xmax = n.max(vdd[parameter])
            x = n.arange(xmin, xmax, (xmax - xmin) / 1000.)
            y = priordict[parameter].pdf(x)
            p.plot(x, y, 'r')

    return


def get_multichain(vdchain, beta=None):
    """
    Gets multichain dictionary from list of VDchain objects
    
    Parameters
    ----------
    vdchain: VDchain
        list of VDchain objects

    Returns
    -------
    vds: dict
        multichain dictionary
    """

    vds = []

    for chain in n.arange(len(vdchain)):
        if vdchain[chain].beta == beta or beta == None:
            vds.append(vdchain[chain]._value_dict)
        else:
            continue

    return vds


def print_param(vds, kk, BI=0.5):
    """
    Print the median and std of parameter kk for all chains in vds.
    Also print name of chains and median logL
    """

    for i in range(len(vds)):
        vd = vds[i].get_value_dict()
        fname = os.path.split(vds[i].filename)[-1]
        N = len(vd['logL'])

        error = n.std(vd[kk][BI * N:])
        ## Trick to show good number of decimal places
        s = '%e' % error
        a = s[s.find(
            'e') + 1:]  # Contains power of range of axis in scientific notation
        if a[0] == '-':
            ndecimal = int(a[1:]) + 1
        else:
            ndecimal = 1

        if i == 0:
            print('Filename\tlog(L)\t%s\tsigma(%s)' % (kk, kk))

        fmtstr = '%s\t%.1f\t%.' + str(ndecimal) + 'f\t%.' + str(ndecimal) + 'f'
        print(fmtstr % (fname, n.median(vd['logL'][BI * N:]),
                        n.median(vd[kk][BI * N:]), error)
              )
    return


###
# CONVERGENCE DIAGNOSTICS
###

def gelmanrubin(vds, BI=0.2, BO=1.0, thinning=1, qs=[0.9, 0.95, 0.99]):
    ## From http://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introbayes_sect008.htm#statug.introbayes.bayesgelm

    if isinstance(vds[0], dict):
        vdi = vds[0].copy()
    else:
        print(vds[0].__class__)

    start_index = n.round(BI * len(vdi[vdi.keys()[0]]))
    end_index = n.round(BO * len(vdi[vdi.keys()[0]]))

    Ws = {}
    Bs = {}
    Vs = {}
    MUs = {}
    DFs = {}
    PSRF = {}
    Rcs = {}

    for kk in vdi.keys():

        values = []
        for i, vd in enumerate(vds):

            if isinstance(vd, dict):
                pass

            values.append(vd[kk][start_index: end_index: thinning])

        values = n.array(values)
        nn = len(values[0])
        m = len(values[:, 0])

        # Compute within-chain variance
        sm2 = n.var(values, axis=1, ddof=1)  # Variance for each chain
        W = n.mean(sm2)  # Mean variance over all chains
        Ws[kk] = W

        # Compute between-chain variance
        thetamean = n.mean(values, axis=1)  # Mean for each chain
        B = nn * n.var(thetamean, ddof=1)  # Variance of means, multiplied by nn
        Bs[kk] = B

        # Estimate mean (mu) using all chains
        mu = n.mean(values)
        MUs[kk] = mu

        # Estimate variance by weighted average of B and W (eq.3 Gelman & Rubin)
        sig = (nn - 1.0) / nn * W + B / nn

        # The parameter distribution can be approximated by a Student's t
        # distribution (Gelman & Rubin) with scale sqrt(V):
        # print kk, sig, B/(nn*m)
        V = sig + B / (nn * m)
        Vs[kk] = V

        # and degrees of freedom df = 2*V**2/var(V) (see eq. 4 G & R)
        varV = ((nn - 1.0) / nn) ** 2.0 * (1.0 / m) * n.var(sm2) + \
               ((m + 1.0) / (nn * m)) ** 2.0 * (2.0 / (m - 1.0)) * B ** 2.0 + \
               2 * (m + 1.0) * (nn - 1.0) / (m * nn ** 2.0) * \
               (1.0 * nn / m) * (n.cov(sm2, thetamean ** 2.0)[1, 0] - \
                                 2 * mu * n.cov(sm2, thetamean)[1, 0])

        df = 2 * V ** 2.0 / varV
        DFs[kk] = df

        psr = n.sqrt((V / W) * df / (df - 2.0))
        PSRF[kk] = psr

        ## Compute degrees of freedom for F distribution for PSRF
        # see sect 3.5 and 3.7 of G & R
        dfn = m - 1.0
        dfd = 2.0 * W ** 2 / (n.var(sm2) / m)

        ## Compute 90%, 95% and 99% percentiles for this distribution
        qq = []
        for q in qs:
            qq.append(scipy.stats.f.ppf(q, dfn, dfd))

        qq = n.array(qq)

        lims = n.sqrt(
            ((nn - 1.) / nn + (m + 1.) / (nn * m) * qq) * df / (df - 2.0))

        print('%s\t%.5f\t%.5f\t%.5f\t%.5f' % (kk, psr, lims[0],
                                              lims[1], lims[2]))

    return


def geweke(x,  bi=0, thin=1, first=0.1, size=0.1, forward=1):
    """
    Compute the Geweke diagnostic of covergence.

    X is the trace of a given parameter.
    """

    # remove burn-in and thin
    xx = x[bi::thin]
    n = len(xx)

    # select reference part of chain
    xref = xx[: int(first*n)]

    # compute mean and variance of reference
    mean_ref = np.mean(xref)
    var_ref = np.var(xref, ddof=1)

    # Iterate over chain and compare to reference
    fractions = np.arange(first, 1+size, size)[::forward]

    results = np.empty((len(fractions) - 1, 2))
    
    for i in range(len(fractions)-1):

        istart = int(fractions[i]*n)
        iend = int(fractions[i+1]*n)
        
        xcomp = xx[istart : iend]
        mean_comp = np.mean(xcomp)
        var_comp = np.var(xcomp, ddof=1)
        
        # Compute Geweke statistics
        z = (mean_comp - mean_ref)/sqrt(var_ref + var_comp)
        #z = (mean_comp - mean_ref)/sqrt(2 * var_ref)
        
        # The variance must also be approximately the same
        # If normal, these variables should be Chi2(N-1), 
        # so their variance is 2*(N - 1), where N is the size of the 
        # sample.
        zz = (var_ref - var_comp) / sqrt(
            2 * (len(xcomp) - 1) + 2 * (len(xref) - 1))

        results[i] = [zz, z]

    return results


# To find the BI of a given chain
def find_BI(vds, samplesize=0.05, endsample=0.1, backwards=True,
            tolerance=2.0, checkvariance=True, correlen=1, param='logL',
            sigmaclip=False, nsigma=5,
            nitersigmaclip=3, verbose=False):
    """
    Find the BurnIn of a given chain or a group of chains by comparing the
    mean and, possibly, also the variance of the trace of parameter param
    throughtout the chain to that at the end of the chain.
    
    Parameters
    ----------
    vds: list, dict, or VDchain instance.
        The list of chains, or individual chain for which the BI will be
        computed. The list can contain dict instances with the traces of the
        parameters, or VDchain instances. All chains must contain param  as a 
        parameter.

    Other parameters
    ----------------
    samplesize: float.
       Size of the sample used to compare to the end sample, at the end of
       the chain. It must be expressed as a fraction of the total length.

    endsample: float.
        Size of the end part of the logL trace used to compare with the rest 
        of the chain

    tolerance: float.
        Number of standard deviations to use as tolerance.

    checkvariance: boolean.
        Determines if a test on the variance of the samples is also performed.

    correlen: float or list
        Correlation length: the number of samples to thin the chain.
        An iterable object containing the correlation length for each
        individual chain for which the BI will be measured can also be given.
        In this case, the number of elements in correlen must equal that in vds.

    param: string.
        Parameter key used to compute burn-in. Default: log(Likelihood)

    sigmaclip: boolean.
        Determines if samples throughout the chain are sigma-clipped before
        computing its variance and mean.

    nsigma: int.
        Number of sigmas to use for sigma clipping algorithm. Default: 5.

    nitersigmaclip: int.
        Number of times the sigma-clipping algorithm is iterated.
    """

    try:
        vds = list(vds)
    except TypeError:
        pass

    # Create list of correlation lengths
    try:
        iter(correlen)
    except TypeError:
        # correlen is a float; convert to a list
        correlen = [correlen] * len(vds)

    # Check if single chain is given.
    if not isinstance(vds, list):
        vds = [vds, ]

    # Define list to contain results
    zlist = []
    zzlist = []
    BIfrac = []
    for j, vd in enumerate(vds):

        if isinstance(vd, dict):
            y = vd[param][::correlen[j]]
        else:
            print('Warning! VDchain class might have changed.')
            y = vd.get_value_dict()[param][::correlen[j]]

        N = len(y)

        # Select end sample
        yf = y[-N * endsample:]

        if sigmaclip:
            # Sigma clip end sample
            yf, ccs = tools.sigma_clipping(yf, nsigma,
                                           niter=nitersigmaclip
                                           )

        mean_yf = n.mean(yf)
        var_yf = n.var(yf, ddof=1)

        # Explore the chain
        Nsample = N * samplesize

        # Define list to contain results
        zlisti = []
        zzlisti = []
        if not backwards:
            ei = 0
            ef = samplesize
            # Forward
            while ef < 1 - endsample:

                # Select sample to compare to end sample
                ys = y[ei * N: ef * N]

                if sigmaclip:
                    # Sigma clips sample
                    ys, ccs = tools.sigma_clipping(ys, nsigma,
                                                   niter=nitersigmaclip
                                                   )

                var_ys = n.var(ys, ddof=1)

                # Compute Geweke statistics (modified: instead of using the
                # variance of sample, which can be extremely large and hinder
                # a correct estimation of BI, we use twice the variance of the
                # end sample.
                z = (n.mean(ys) - mean_yf) / n.sqrt(2 * var_yf)

                # The variance must also be approximately the same
                # If normal, these variables should be Chi2(N-1), 
                # so their variance is 2*(N - 1), where N is the size of the 
                # sample.
                zz = (var_ys - var_yf) / n.sqrt(
                    2 * (len(ys) - 1) + 2 * (len(yf) - 1))

                # Ad hoc condition on maximum of sample
                """
                print n.max(ys), mean_yf, tolerance*n.sqrt(var_yf)
                condAH = n.max(ys) >= (mean_yf - tolerance*n.sqrt(var_yf))
                """
                zlisti.append(z)
                zzlisti.append(n.abs(zz))

                if checkvariance:
                    accept = n.abs(z) < tolerance and n.abs(zz) < tolerance
                else:
                    accept = n.abs(z) < tolerance
                if accept:
                    print('Chain %d: BI set to '
                          '%.2f' % (j, ef + 0.5 * samplesize))
                    BIfrac.append(min(ef + 0.5 * samplesize, 1.0))
                    zlist.append(zlisti)
                    zzlist.append(zzlisti)
                    break

                ei = ef
                ef = ei + samplesize

            if ef >= 1 - endsample:
                # If BI was not found, print warning and set BI = 1
                print('Burn In fraction not found. Has the chain converged?')
                BIfrac.append(1)
                zlist.append(zlisti)
                zzlist.append(zzlisti)

        else:
            ef = 1.0 - endsample
            ei = ef - samplesize
            # Backwards
            while ei >= -1e-8:
                # Select sample to compare to end sample
                ys = y[ei * N: ef * N]

                if sigmaclip:
                    # Sigma clips sample
                    ys, ccs = tools.sigma_clipping(ys, nsigma,
                                                   niter=nitersigmaclip
                                                   )

                var_ys = n.var(ys, ddof=1)

                # Compute Geweke statistics
                z = (n.mean(ys) - mean_yf) / n.sqrt(2 * var_yf)

                # The variance must also be approximately the same
                # If normal, these variables should be Chi2(N-1), 
                # so their variance is 2*(N - 1), where N is the size of the 
                # sample.
                zz = (var_ys - var_yf) / n.sqrt(
                    2 * (len(ys) - 1) + 2 * (len(yf) - 1))

                # Ad hoc condition on maximum of sample
                condAH = n.max(ys) >= (mean_yf - tolerance * n.sqrt(var_yf))

                """
                ## Compute KS test on the two samples
                zz, probKS = scipy.stats.ks_2samp(yf, ys)
                """

                if verbose:
                    # print z, zz, probKS
                    print(ei, ef, mean_yf, var_yf, n.mean(ys), var_ys, condAH)
                zlisti.append(n.abs(z))
                zzlisti.append(n.abs(zz))

                if checkvariance:
                    accept = n.abs(z) > tolerance or n.abs(
                        zz) > tolerance or -condAH
                else:
                    accept = n.abs(z) > tolerance or -condAH

                if accept:
                    print('Chain %d: BI set to'
                          ' %.2f' % (j, ef + 0.5 * samplesize))
                    BIfrac.append(min(ef + 1.5 * samplesize, 1.0))
                    zlist.append(zlisti)
                    zzlist.append(zzlisti)
                    break

                ef = ei
                ei = ef - samplesize

            if ei < -1e-8:
                # If BI was not found, means that BI = 0!
                print('Burn In fraction not found. Setting BI = 0')
                BIfrac.append(0)
                zlist.append(zlisti)
                zzlist.append(zzlisti)

    return zlist, zzlist, BIfrac


def select_best_chains(vds, BI, CL, nmin=100, tolerance=2.0,
                       KStolerance=0.1 / 1e2,
                       param='posterior', param2=None, fnames=None):
    """
    Identify the chains having a significant worse logL than the best one.
    
    Parameters
    ----------
    vds: list, dict, or VDchain instance.
        The list of chains to compare. The list can contain dict instances
        with the traces of the parameters, or VDchain instances.
        All chains must contain param  as a parameter.

    BI: float or list
        The fraction of the chains to be discarded before comparing
	An iterable object containing the fraction to be discarded for each
	individual chain to be can be given. In this case, the
	number of elements in BI must equal that in vds. Otherwise a single
        BI for all chains is used

    CL: float or list
        The correlation length of the chains to be compared. Chains will be
        thinned by this number before comparing.
	An iterable object containing the fraction to be discarded for each
	individual chain to be can be given. In this case, the
	number of elements in CL must equal that in vds. Otherwise a single
        CL for all chains is used
        
    Other parameters
    ----------------
    tolerance: float.
        Number of standard deviations to use as tolerance.

    KStolerance: float.
        Tolerance for KS test.

    param: string.
        Parameter key used to compute burn-in. Default: log(Likelihood)

    fnames: list.
        List of names of chains to compare.
    """
    try:
        vds = list(vds)
    except TypeError:
        pass

    # Check if single chain is given.
    if not isinstance(vds, list):
        vds = [vds, ]

    # Define list to contain results
    ys = []
    mediany = []
    sigmay = []
    pars = []
    ireject = []
    iaccept = []
    ## Create list of starting indexes
    try:
        iter(BI)
    except TypeError:
        print('Warning!')
        # BI is a float; convert to a list
        BI = [BI] * len(vds)

    ## Create list of correlation lengths
    try:
        iter(CL)
    except TypeError:
        # CL is a float; convert to a list
        CL = [CL] * len(vds)

    # Check if BI and CL contain the same number of elements as vds
    for v, namev in zip((BI, CL), ('BI', 'CL')):
        if len(v) != len(vds):
            raise TypeError(
                '%s must be either a float or contain the same number of elements as the input list' % namev)

    ## Prepare lists
    accepted_chains = []
    rejected_chains = []
    accepted_ind = []

    if fnames is not None:
        accepted_fnames = []
        rejected_fnames = []

    ## Iterate over all chains
    for j, vd in enumerate(vds):

        if isinstance(vd, dict):
            y = vd[param]
            if param2 is not None:
                y2 = vd[param2]
        else:
            print('Warning! VDchain class might have changed.')
            y = vd._value_dict[param]
            if param2 is not None:
                y2 = vd._value_dict[param2]


                ## Compute median and standard deviation
        yc = y[len(y) * BI[j]:: CL[j]]
        if param2 is not None:
            y2c = y2[len(y2) * BI[j]:: CL[j]]

        if len(yc) <= nmin:
            print('Chain %d: Less than %d steps' % (j, nmin))
            ireject.append(j)
            rejected_chains.append(vd)
            if fnames is not None:
                rejected_fnames.append(fnames[j])
            continue
        else:
            iaccept.append(j)

        median_y = n.median(yc)
        sigma_y = n.std(yc, ddof=1)

        ## Add array to list
        ys.append(yc)
        if param2 != None:
            pars.append(y2c)
        ##
        mediany.append(median_y)
        sigmay.append(sigma_y)

    y0 = ys[n.argmax(mediany)]
    my0 = mediany[n.argmax(mediany)]
    sy0 = sigmay[n.argmax(mediany)]

    probsKS = []
    for i in range(len(mediany)):

        if fnames is not None:
            fname = fnames[iaccept[i]]
        else:
            fname = ''

        ## Compute difference between chain i and best
        z = (mediany[i] - my0) / n.sqrt(sy0 ** 2 + sigmay[i] ** 2)

        ## Compute KS test on the two samples
        zz, probKS = scipy.stats.ks_2samp(y0, ys[i])
        probsKS.append(probKS)

        rejectstr = ''
        if abs(z) > tolerance or probKS < KStolerance:
            rejectstr = '*'
            rejected_chains.append(vds[iaccept[i]])
            if fnames is not None:
                rejected_fnames.append(fname)

        else:
            accepted_chains.append(vds[iaccept[i]])
            accepted_ind.append(iaccept[i])
            if fnames is not None:
                accepted_fnames.append(fname)

        if z == 0:
            rejectstr = '!'

        if param2 != None:

            print('%s%s : %.3f\t%.3f\t%.3f\t%.3e\t%.3f\t%d' % (rejectstr,
                                                               fname,
                                                               mediany[i],
                                                               sigmay[i], z,
                                                               probKS,
                                                               n.median(
                                                                   pars[i]),
                                                               len(pars[i])
                                                               )
                  )
        else:
            print('%s%s : %.3f\t%.3f\t%.3f\t%.3e' % (rejectstr,
                                                     fname, mediany[i],
                                                     sigmay[i], z,
                                                     probKS,
                                                     )
                  )

    if fnames is not None:
        return accepted_chains, accepted_fnames, rejected_chains, rejected_fnames, accepted_ind
    else:
        return accepted_chains, rejected_chains, accepted_ind


def get_priors_from_value_dict(vds, pastisfile):
    """
    Get the priors for all steps of a chain represented by vd.
    Adds a key to vd dictionary called 'prior'

    Parameters
    ----------
    vds: list
        List containing dictionary, or VDchain instances of the chains for
        which to compute the priors.

    pastisfile: str, or file instance
        The file containing the configuration for the run for which to compute
        the priors.        
    """

    # Read configuration file
    if isinstance(pastisfile, file):
        f = pastisfile
    else:
        f = open(pastisfile, 'r')
    dd = pickle.load(f)
    f.close()

    # Construct priordict from configuration file
    priordict = prior_constructor(dd[1], dd[3])

    newvds = []
    for vd in vds:
        print(vd)

        # Get value dict
        if isinstance(vd, dict):
            vdd = vd.copy()
        else:
            raise TypeError('Warning! Input must be dictionary')

        N = len(vdd[vdd.keys()[0]])
        vdd['prior'] = n.zeros(N)

        # Iterate over all elements in chain.
        for i in range(N):

            if i % 1e4 == 0:
                print('Step %d out of %d' % (i, N))
            # Based on input_dict from pastis file, construct a new dictionary
            # with values
            # step i
            ddi = dd[1].copy()

            for kk in vdd.keys():
                try:
                    k1, k2 = kk.split('_')
                except ValueError:
                    continue

                try:
                    ddi[k1][k2][0] = vdd[kk][i]
                except KeyError:
                    raise KeyError('Parameter {} of object {} not present in '
                                   'configuration dict.'.format(k2, k1))

            # With modified input_dict, compute state
            Xi, labeldict = state_constructor(ddi)

            # Compute priors on new state
            prior, priorprob = compute_priors(priordict, labeldict)

            # Add prior to dictionary.
            vdd['prior'][i] = prior

        newvds.append(vdd)

    return newvds


def compute_hdi_dev(x, y, q=0.95):
    """
    Compute the 100*q% Highest Density Region, given samples from the posterior
    distribution and the posterior (up to a constant) evaluated in those
    positions.

    :param np.array x: posterior samples. This could be a multidimensional
    array as long as the samples run along the first axis.
    :param np.array y: posterior density evaluated in x.
    :param float q: fraction of mass contained in HDI.
    """

    # Sort posterior density array
    ind = np.argsort(y)

    # Keep only first q*n samples
    n = len(x)

    # How to report the result ? The question is on mode detection
    return x[ind[-n*q:]]


def compute_hdi(binedges, pdf, q=0.95):
    """
    Compute the 100*q% Highest Density Interval, given a normalised distribution
    pdf (len N), sampled in bins m (m has N+1 elements)
    """

    lower_bin_edges = binedges[: - 1]
    upper_bin_edges = binedges[1:]
    bincentre = 0.5 * (binedges[1:] + binedges[:-1])
    binsize = n.diff(binedges)

    # Sort elements from pdf
    isort = n.argsort(pdf)[::-1]

    cumulq = 0
    # Start adding bins until the requested fraction of the mass (q) is
    # reached
    for i, binnumber in enumerate(isort):
        cumulq += pdf[binnumber] * binsize[binnumber]

        if cumulq >= q:
            break

    # Keep only bins in 100*q% HDI
    bins_in_HDI = isort[: i + 1]

    # Sort binindex to find for non-contiguous bins
    sorted_binnumber = n.sort(bins_in_HDI)
    jumpind = n.argwhere(n.diff(sorted_binnumber) > 1)

    # Construct intervals
    HDI = []

    if len(jumpind) == 0:
        # HDI.append([binedges[sorted_binnumber[0]], binedges[sorted_binnumber[-1]+1]])
        HDI.append([lower_bin_edges[bins_in_HDI].min(),
                    upper_bin_edges[bins_in_HDI].max()])

    else:
        ji = 0
        for jf in jumpind[:, 0]:
            HDI.append([lower_bin_edges[sorted_binnumber[ji]],
                        upper_bin_edges[sorted_binnumber[jf]]])
            ji = jf + 1

        HDI.append([lower_bin_edges[sorted_binnumber[ji]],
                    upper_bin_edges[sorted_binnumber[-1]]])

    return HDI
