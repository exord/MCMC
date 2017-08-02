# Functions to transform input dictionary to list of parameters
# and to update this dictionary to feed it back to the Object Builder
import os
import numpy as n
import numpy as np
import pickle
from math import log, pi

import bayev.perrakis as perr

# Intra-package imports
from . import Objects_MCMC as objMCMC


def state_constructor(input_dict):
    """
    Construct the state of a Markov Chain starting from a input dictionary
    with a given structure (see PASTIS documentation?).
    """

    theta = []
    global labeldict
    labeldict = {}

    # Iteration over all objects
    for objkey in input_dict.keys():

        # Iteration over all parameters of a given object
        for parkey in input_dict[objkey]:

            if parkey == 'object':
                continue

            parlist = input_dict[objkey][parkey]

            if not isinstance(parlist, list) or len(parlist) < 2:
                continue

            # If parameter has None value, continue
            if parlist[0] is None:
                continue

            # Search jump size proposed by user
            try:
                if parkey != 'ebmv':
                    jumpsize = parlist[8]
                else:
                    raise IndexError
            except IndexError:
                # If not present, get prior information to do a reasonable
                # guess on the jump size
                if parlist[2] in ('Uniform', 'Jeffreys', 'Sine'):
                    priorsize = 0.683*(parlist[4] - parlist[3])

                elif parlist[2] in ('Normal', 'LogNormal', 'TruncatedUNormal'):
                    priorsize = parlist[4]*2.0

                elif parlist[2] == 'Binormal':
                    priorsize = 2.0*max(parlist[4], parlist[6])

                elif parlist[2] == 'AssymetricNormal':
                    priorsize = 2.0*max(parlist[4], parlist[5])

                elif parlist[2] == 'PowerLaw':
                    priorsize = 0.683*(parlist[5] - parlist[4])

                elif parlist[0] != 0:
                    priorsize = 2.0*abs(parlist[0]*0.1)

                else:
                    priorsize = 1.0

                jumpsize = priorsize*0.5

            ##
            # Construct parameter instance with information on dictionary
            # par = Parameter(parlist[0], input_dict[objkey]['object'], jump =
            #                parlist[1], label = objkey+'_'+parkey)
            par = objMCMC.Parameter(parlist[0], None, jump=parlist[1],
                                    label=objkey + '_' + parkey,
                                    proposal_scale=jumpsize)

            theta.append(par)
            labeldict[par.label] = par

    return n.array(theta), labeldict


def state_deconstructor(state, input_dict):
    output_dict = {}
    for key in input_dict.keys():
        output_dict[key] = input_dict[key].copy()

    # Iterate over all parameters
    for par in state:

        objkey = par.label.split('_')[0]
        parkey = par.label.split('_')[1]

        if not (objkey in output_dict):
            print('Warning! Dictionary does not have key \"{}\"'.format(objkey))
            continue

        if not (parkey in output_dict[objkey]):
            raise KeyError('Error! Dictionary of object {0} does not have key '
                           '\"{1}\"'.format(objkey, parkey))

        output_dict[objkey][parkey] = []
        output_dict[objkey][parkey].append(par.get_value())
        for i in range(1, len(input_dict[objkey][parkey])):
            output_dict[objkey][parkey].append(input_dict[objkey][parkey][i])

        # output_dict[objkey][parkey][0] = par.get_value()

    return output_dict

# WARNING! CHECK WHICH VERSION OF state_deconstructor IS BETTER IN TERMS
# OF OVERWRITING THE INPUT STATE
"""
def state_deconstructor(state, output_dict):
    #
    # Iterate over all parameters
    for par in state:

        objkey = par.label.split('_')[0]
        parkey = par.label.split('_')[1]

        if not (objkey in output_dict):
            print('Warning! Dictionary does not have key \"{}\"'.format(objkey))
            continue

        if not (parkey in output_dict[objkey]):
            raise KeyError('Error! Dictionary of object {0} does not have key '
                           '\"{1}\"'.format(objkey, parkey))

        output_dict[objkey][parkey][0] = par.get_value()


    return output_dict
"""


def get_jitter(data, instrument, paramdict, observable=None):
    """
    Compute jitter given an element of a datadict (data).
    :return:
    """

    # # Construct jitter key (to match new keys in RV diagnostics).
    if observable is None:
        jitterkey = instrument + '_jitter'
    else:
        jitterkey = instrument + '_' + observable + 'jitter'

    if 'jittermodel' not in data or data['jittermodel'] == 'constant':
        try:
            return paramdict[jitterkey].get_value()
        except KeyError:
            return 0.0

    elif data['jittermodel'] == 'linear_rhk':
        minjitterkey = jitterkey.replace('jitter', 'minjitter')
        alphajitterkey = jitterkey.replace('jitter', 'alphajitter')

        alpha = paramdict[alphajitterkey].get_value()
        minjitter = paramdict[minjitterkey].get_value()

        return alpha * (data['data']['rhk'] + 5.0) + minjitter

    elif data['jittermodel'] == 'linear_halpha':
        minjitterkey = jitterkey.replace('jitter', 'minjitter')
        alphajitterkey = jitterkey.replace('jitter', 'alphajitter')

        try:
            alpha = paramdict[alphajitterkey].get_value()
        except KeyError:
            # Use same alpha for SOPHIEm and SOPHIEp
            alpha = paramdict[alphajitterkey.replace('SOPHIEm',
                                                     'SOPHIEp')].get_value()
        minjitter = paramdict[minjitterkey].get_value()

        return alpha * (data['data']['halpha'] -
                        data['data']['halpha'].min()) + minjitter


def chain2inputdict(vddict, index=None):
    """
    Convert a chain dictionary to an input_dict appropiate to construct a model.

    The function returns an input dict with nonesense prior information that
    can be passed to the object builder to construct objects

    :param dict vddict: a dictionary instance with the parameter names and the
     chain traces.

    :param int index: if not None, use this element of chain to build input dict.
    """

    vddict.pop('logL', None)
    vddict.pop('posterior', None)

    # Prepare input dict skeleton
    import string
    input_dict = dict((string.split(s, '_')[0], {}) for s in vddict)

    for i, p in enumerate(vddict.keys()):
        # split object and param name
        obj, par = p.split('_')
        input_dict[obj][par] = [vddict[p][index], 0, 'Uniform', 0.0, 0.0,
                                0.0, 0.0, '']

    return input_dict

def emcee_flatten(sampler, bi=None, chainindexes=None):
    """
    chainindexes must be boolean
    """

    if bi is None:
        bi = 0
    else:
        bi = int(bi)

    nwalkers, nsteps, dim = sampler.chain.shape

    if chainindexes is None:
        chainind = [True] * nwalkers
    else:
        chainind = list(chainindexes)
        assert len(chainind) == sampler.chain.shape[0]
    
    fc = sampler.chain[chainind, bi:, :].reshape(sum(chainind) * (nsteps - bi),
                                                 dim)
    
    return fc

def read_samplers(samplerfiles, rootdir):
    f = open(samplerfiles)
    samplerlist = f.readlines()
    samplers = []
    for sam in samplerlist:
        print('Reading sampler from file {}'.format(sam.rstrip()))
        f = open(os.path.join(rootdir, sam.rstrip()))
        samplers.append(pickle.load(f))
        
    return samplers

def emcee_vd(sampler, parnames, bi=None, chainindexes=None):
    """
    Produce PASTIS-like value dict from emcee sampler.
    """

    assert sampler.chain.shape[-1] == len(parnames)

    # First flatten chain
    fc = emcee_flatten(sampler, bi, chainindexes)

    # Value dict
    vd = dict((p, fc[:, parnames.index(p)]) for p in parnames)
    return vd

def get_map_values(sampler):

    ind = np.unravel_index(np.argmax(sampler.lnprobability),
                           sampler.lnprobability.shape)
    return sampler.chain[ind[0], ind[1]]

def emcee_mapdict(sampler):
    mapvalues = get_map_values(sampler)
    return dict((sampler.args[0][i], mapvalues[i])
                for i in range(len(mapvalues)))

def emcee_perrakis(sampler, nsamples=5000, bi=0, cind=None):
    """
    Compute the Perrakis estimate of ln(Z) for a given sampler.
    """

    # Flatten chain first
    fc = emcee_flatten(sampler, bi=bi, chainindexes=cind)

    # Construct marginal samples
    marginal = perr.make_marginal_samples(fc, nsamples)

    # Get functions and parameters
    lnlikefunc = sampler.args[1]
    lnpriorfunc = sampler.args[2]

    lnlikeargs = [sampler.args[0],]
    lnpriorargs = [sampler.args[0],]
    lnlikeargs.extend(sampler.kwargs['lnlikeargs'])
    lnpriorargs.extend(sampler.kwargs['lnpriorargs'])

    # Change this ugly thing!
    def lnl(x, *args):
        y = np.empty(len(x))
        for i, xx in enumerate(x):
            y[i] = lnlikefunc(xx, *args)
        return y

    def lnp(x, *args):
        y = np.empty(len(x))
        for i, xx in enumerate(x):
            y[i] = lnpriorfunc(xx, *args)
        return y
        
        
    # Compute perrakis
    lnZ =  perr.compute_perrakis_estimate(marginal, lnl, lnp,
                                          lnlikeargs, lnpriorargs)

    # Correct for missing term in likelihood
    datadict = sampler.kwargs['lnlikeargs'][1]
    nobs = 0
    for inst in datadict:
        nobs += len(datadict[inst]['data'])
    #print('{} datapoints.'.format(nobs))
    lnZ += -0.5 * nobs * log(2*pi)

    return lnZ, lnZ/log(10)


def multirun_perrakis(lnzdict,
                      sampler, nsamples=5000, bi=0, cind=None, nruns=1000):

    lnz = np.zeros([nruns, 2])
    for i in range(nruns):
        if (i+1)%10 == 0:
            print('Step {} out of {}'.format(i+1, nruns))
        lnz[i] = emcee_perrakis(sampler, nsamples, bi, cind)

    """
    f = open('/Users/rodrigo/Desktop/test.dat', 'a+')

    for i in range(len(lnz)):
        f.write('{:.5f}\t{:.5f}\n'.format(lnz[i][0], lnz[i][1]))
    f.close()
    """
    import time
    lnzdict[time.time()] = lnz

    return
                
