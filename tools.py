# Functions to transform input dictionary to list of parameters
# and to update this dictionary to feed it back to the Object Builder
import os
import numpy as n
import numpy as np
import pickle
from math import log, pi
import time
import warnings
import multiprocessing as mp
from multiprocessing.queues import Empty

import bayev.perrakis as perr
import emcee

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
                    priorsize = 0.683 * (parlist[4] - parlist[3])

                elif parlist[2] in ('Normal', 'LogNormal', 'TruncatedUNormal'):
                    priorsize = parlist[4] * 2.0

                elif parlist[2] == 'Binormal':
                    priorsize = 2.0 * max(parlist[4], parlist[6])

                elif parlist[2] == 'AssymetricNormal':
                    priorsize = 2.0 * max(parlist[4], parlist[5])

                elif parlist[2] == 'PowerLaw':
                    priorsize = 0.683 * (parlist[5] - parlist[4])

                elif parlist[0] != 0:
                    priorsize = 2.0 * abs(parlist[0] * 0.1)

                else:
                    priorsize = 1.0

                jumpsize = priorsize * 0.5

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
    sampler can be an emcee.Sampler instance or an iterable. If the latter,
    first element must be chain array of shape [nwalkers, nsteps, dim],
    second element must be lnprobability array [nwalkers, nsteps],
    third element is acceptance rate (nwalkers,)
    fouth element is the sampler.args attribute of the emcee.Sampler isntance.
    lnprior function]

    chainindexes must be boolean
    """

    if bi is None:
        bi = 0
    else:
        bi = int(bi)

    if isinstance(sampler, emcee.Sampler):
        nwalkers, nsteps, dim = sampler.chain.shape
        chain = sampler.chain
    elif np.iterable(sampler):
        nwalkers, nsteps, dim = sampler[0].shape
        chain = sampler[0]
    else:
        raise TypeError('Unknown type for sampler')

    if chainindexes is None:
        chainind = np.array([True] * nwalkers)
    else:
        chainind = np.array(chainindexes)
        assert len(chainind) == chain.shape[0]

    fc = chain[chainind, bi:, :].reshape(sum(chainind) * (nsteps - bi), dim)

    # Shuffle once to loose correlations
    np.random.shuffle(fc)
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

    if isinstance(sampler, emcee.Sampler):
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

    See docstring in emcee_flatten for format of sampler
    """

    # Flatten chain first
    fc = emcee_flatten(sampler, bi=bi, chainindexes=cind)

    # Get functions and arguments
    lnlikefunc, lnpriorfunc, lnlikeargs, lnpriorargs = get_func_args(sampler)

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

    # Construct marginal samples
    marginal = perr.make_marginal_samples(fc, nsamples)

    # Compute perrakis
    ln_z = perr.compute_perrakis_estimate(marginal, lnl, lnp,
                                          lnlikeargs, lnpriorargs)

    # Correct for missing term in likelihood
    datadict = get_datadict(sampler)
    nobs = 0
    for inst in datadict:
        nobs += len(datadict[inst]['data'])
    # print('{} datapoints.'.format(nobs))
    ln_z += -0.5 * nobs * log(2 * pi)

    return ln_z, ln_z / log(10)


def get_func_args(sampler):
    """
    Read functions and arguments from sampler to compute lnprior and lnlike

    :param sampler:
    :return:
    """
    if isinstance(sampler, emcee.Sampler):
        # Get functions and parameters
        lnlikefunc = sampler.args[1]
        lnpriorfunc = sampler.args[2]

        lnlikeargs = [sampler.args[0], ]
        lnpriorargs = [sampler.args[0], ]
        lnlikeargs.extend(sampler.kwargs['lnlikeargs'])
        lnpriorargs.extend(sampler.kwargs['lnpriorargs'])

    elif np.iterable(sampler):
        # Get functions and parameters
        lnlikefunc = sampler[-2][1]
        lnpriorfunc = sampler[-2][2]

        lnlikeargs = [sampler[-2][0], ]
        lnpriorargs = [sampler[-2][0], ]
        lnlikeargs.extend(sampler[-1]['lnlikeargs'])
        lnpriorargs.extend(sampler[-1]['lnpriorargs'])

    else:
        raise TypeError('Unknown type for sampler')

    return lnlikefunc, lnpriorfunc, lnlikeargs, lnpriorargs


def get_datadict(sampler):
    if isinstance(sampler, emcee.Sampler):
        return sampler.kwargs['lnlikeargs'][1]

    elif np.iterable(sampler):
        return sampler[-1]['lnlikeargs'][1]
    else:
        raise TypeError('Unknown type for sampler')


def multi_emcee_perrakis(sampler, nsamples=5000, bi=0, thin=1, nrepetitions=1,
                         cind=None, ncpu=None, datacorrect=False,
                         outputfile='./perrakis_out.txt'):
    """
    Compute the Perrakis estimate of ln(Z) for a given sampler 
    repeateadly using multicore

    WRITE DOC
    """

    # Flatten chain first
    fc = emcee_flatten(sampler, bi=bi, chainindexes=cind)[::thin]

    # Get functions and arguments
    lnlikefunc, lnpriorfunc, lnlikeargs, lnpriorargs = get_func_args(sampler)

    # Change this ugly thing!
    def lnl(x, *args):
        y = np.empty(len(x))
        for ii, xx in enumerate(x):
            y[ii] = lnlikefunc(xx, *args)
        return y

    def lnp(x, *args):
        y = np.empty(len(x))
        for ii, xx in enumerate(x):
            y[ii] = lnpriorfunc(xx, *args)
        return y

    # Prepare multiprocessing
    if ncpu is None:
        ncpu = mp.cpu_count()

    # Check if number of requested repetitions below ncpu
    ncpu = min(ncpu, nrepetitions)

    # Instantiate output queue
    q = mp.Queue()

    print('Running {} repetitions on {} CPU(s).'.format(nrepetitions, ncpu))

    if ncpu == 1:
        # Do not use multiprocessing
        lnz = single_perrakis(fc, nsamples, lnl, lnp, lnlikeargs, lnpriorargs,
                              nrepetitions, None)

    else:
        # Number of repetitions per process
        nrep_proc = int(nrepetitions / ncpu)

        # List of jobs to run
        jobs = []

        for i in range(ncpu):
            p = mp.Process(target=single_perrakis, args=[fc, nsamples, lnl, lnp,
                                                         lnlikeargs,
                                                         lnpriorargs,
                                                         nrep_proc, q])
            jobs.append(p)
            p.start()
            time.sleep(1)

        # Wait until all jobs are done
        for p in jobs:
            p.join()

        # Recover output from jobs
        try:
            print(q.empty())
            lnz = np.concatenate([q.get(block=False) for p in jobs])
        except Empty:
            warnings.warn('At least one of the jobs failed to produce output.')

        try:
            len(lnz)
        except UnboundLocalError:
            raise UnboundLocalError('Critical error! No job produced any '
                                    'output. Aborting!')

    if datacorrect:
        # Correct for missing term in likelihood
        datadict = get_datadict(sampler)
        nobs = 0
        for inst in datadict:
            nobs += len(datadict[inst]['data'])
        lnz += -0.5 * nobs * log(2 * pi)

    # Write to file
    f = open(outputfile, 'a+')
    for ll in lnz:
        f.write('{:.6f}\t{:.6f}\n'.format(ll, ll / log(10)))
    f.close()

    return lnz, lnz / log(10)


def single_perrakis(fc, nsamples, lnl, lnp, lnlargs, lnpargs, nruns,
                    output_queue):
    # Prepare output array
    lnz = np.empty(nruns)

    #
    np.random.seed()

    for i in range(nruns):
        if i % 10 == 0:
            print(i)

        # Construct marginal samples
        marginal = perr.make_marginal_samples(fc, nsamples)

        # Compute perrakis
        lnz[i] = perr.compute_perrakis_estimate(marginal, lnl, lnp,
                                                lnlargs, lnpargs)

    if output_queue is None:
        return lnz
    else:
        output_queue.put(lnz)
    return
