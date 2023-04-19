'''Functions comparing execution times of (semi-automatically-generated) varieties of
numba-compilations of a given function, including
lazy vs eager compilation, vectorisation, parallelisation, as well as varieties of rolling (see Generic_Util.numba.higher_order).'''

from numba import vectorize

from .higher_order import *
from ..benchmarking import compare_implementations

from typing import Callable


def compare_compilations(f: Callable = None, numba_signature = None, separate_numba_signatures = None,
              f_scalar: Callable = None, numba_signatures_scalar = None, separate_numba_signaturess_scalar = None, parallel = False,
              fs_with_own_args: dict[str, tuple[Callable, list, dict]] = None, n = 200, wait = 1, prefix = None, verbose = True,
              args: list = None, kwargs: dict = None):
    '''Combine available ingredients to benchmark possible numba versions of the given function.

        numba_signature and parallel are only used if f is given.
        numba_signatures_scalar is only used if f_scalar is given.
        separate_numba_signatures and separate_numba_signaturess_scalar are, respectively, a list and a list of lists of
        additional signatures to be tried in addition to, respectively, numba_signature and numba_signatures_scalar.'''
    f_dict = dict()
    if f:
        f_dict['Base'], f_dict['Lazy'] = f, njit(f)
        if parallel: f_dict['Parallel'] = njit(parallel = True)(f)
        if numba_signature:
            f_dict['Eager'] = njit(numba_signature)(f)
            if parallel: f_dict['Parallel Eager'] = njit(numba_signature, parallel = True)(f)
        if separate_numba_signatures:
            for i, sig in enumerate(separate_numba_signatures):
                f_dict[f'Eager{i+2}'] = njit(sig)(f)
                if parallel: f_dict[f'Parallel Eager{i+2}'] = njit(sig, parallel = True)(f)
    if f_scalar:
        f_dict['Vec Lazy'] = vectorize(f_scalar)
        if numba_signatures_scalar: f_dict['Vec Eager'] = vectorize(numba_signatures_scalar)(f_scalar)
        if separate_numba_signaturess_scalar:
            for i, sigs in enumerate(separate_numba_signaturess_scalar): f_dict[f'Vec Eager{i+2}'] = vectorize(sigs)(f_scalar)
    if prefix: f_dict = {(prefix + k): v for k, v in f_dict.items()}
    return compare_implementations(f_dict, n, wait, verbose, fs_with_own_args, args, kwargs)


def compare_rolls(f_pairs: dict[str, tuple[Callable, Callable]], test_array, test_n = 200,
                  n = 3, nb_type = f8, nb_out_type = None, np_out_type = np.float64, input_is_2d = False):
    '''For each labelled (1-char keys recommended for neatness) implementation of a given function
        (and possibly a stencil-optimised version of it, otherwise a None in its tuple),
        first check that all outputs agree and then benchmark the possible stencil variations:
            no stencil (NONE), automatic stencil (AUTO) and provided stencil (GIVE)'''
    rolled = dict()
    res = dict()
    for k, f_pair in f_pairs.items():
        temp_res = dict()
        rolled[label := f'{k}_NONE'] = roll(f_pair[0], n = n, nb_type = nb_type, nb_out_type = nb_out_type, np_out_type = np_out_type, use_stencil = False, input_is_2d = input_is_2d)
        temp_res['NONE'] = rolled[label](test_array)

        rolled[label := f'{k}_AUTO'] = roll(f_pair[0], n = n, nb_type = nb_type, nb_out_type = nb_out_type, np_out_type = np_out_type, input_is_2d = input_is_2d)
        temp_res['AUTO'] = rolled[label](test_array)

        rolled[label := f'{k}_GIVE'] = roll(f_pair[0], f_pair[1], n = n, nb_type = nb_type, nb_out_type = nb_out_type, np_out_type = np_out_type, input_is_2d = input_is_2d)
        temp_res['GIVE'] = rolled[label](test_array)

        assert np.isclose(temp_res['NONE'], temp_res['AUTO']).all(), f'The AUTO result of implementation {k} differs from the NONE one'
        assert np.isclose(temp_res['NONE'], temp_res['GIVE']).all(), f'The GIVE result of implementation {k} differs from the NONE one'
        res[k] = temp_res['NONE']

    keys = list(res.keys())
    for ki in range(len(f_pairs) - 1): assert np.isclose(res[keys[ki]], res[keys[ki+1]]).all(), f'The {keys[ki+1]} output differs from previous ones'

    benchmark = compare_implementations(rolled, n = test_n, args = [test_array])

    return benchmark, rolled, res[keys[0]]


