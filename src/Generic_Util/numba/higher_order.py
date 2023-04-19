'''Higher-order numba-compilation functions, currently only functions to "roll" simpler functions
(1d-to-scalar or 2d-to-scalar/1d) over arrays, with a few combinations of input and output type signatures.'''

import numpy as np
from functools import cache
from numba import njit, stencil, guvectorize, f8
from numba.core.types import Array


@cache
def roll(f, f_stencil_version = None, n = 3, nb_type = f8, nb_out_type = None, np_out_type = np.float64, use_stencil = True, input_is_2d = False):
    '''Produce a compiled function which roll-applies a 1d-to-scalar f (and possibly a stencil-able version of f too) to a 1d or 2d input array
        for the given window size, numba input type (output type too if different) and numpy output type.
        I.e. a function which applies f to n-slices of the input array (full-width, sliced-height in the 2d case)
        and returns an array of the same shape as the input.
        Arguments to the produced function are expected to be F-contiguous if input_is_2d and C-contiguous otherwise
        (can impose either with np.asfortranarray/np.ascontiguousarray and inspect it with typeof(ARRAY).layout == 'F/C'),
        while outputs are always C-contiguous.

            NOTE: Unless willing to manually check the first n-1 entries for correctness when applying the produced function,
            avoid using explicit references to the slice elements by index (e.g. xs[-2] + xs[-1] + xs[0]) in f's definition;
            use collective operators instead (e.g. xs[-2:1].sum(), which is fine because slicing deals with missing entries).

        :param f: A 1d-to-scalar function compilable as nb_out_type(nb_type[::1]) (or, same but nominally different, nb_out_type(Array(nb_type, 1, 'F')) if input_is_2d)
        :param f_stencil_version: Optional - A function identical to f except in that it indexes its input with stencil-compatible window indices
                coherent with the given n, i.e. same code as f but having xs[1-n:1] instead of xs (note the n-1 nature of modifications).
                If the underlying code is optimised in particular in numba (e.g. numpy vectorised operators and methods),
                then the output will be considerably faster with this argument than not;
                HOWEVER, if the code is NOT as optimised in numba, then the output could be considerably SLOWER than without this argument.
        :param n: The window size (how many previous xs, including the current one, should be included in the processed slices)
        :param nb_type: The scalar numba type of f's (and the produced function's) input
        :param nb_out_type: Optional - The scalar numba type of f's (and the produced function's) output; provide only if different from nb_type
        :param np_out_type: The scalar numpy type of f's (and the produced function's) output (required for array allocation, while the input one is not)
        :param use_stencil: [True by default] Whether the produced function should be built from stencil (and guvectorize) rather than njit;
                see the caveat in f_stencil_version's description; the best practice is to provide a f_stencil_version and benchmark the output over use_stencil
        :param input_is_2d: Whether the input to the produced function is going to be an F-contiguous 2d array (on which f will be applied column-wise) instead of a C-contiguous 1d array
        :returns: A function applying f to every n-slice (or [n,:]-slice) of the input array, with signature nb_out_type[::1](nb_type[::1]) or nb_out_type[:, ::1](nb_type[::1, :]) if input_is_2d'''
    n -= 1
    if not nb_out_type: nb_out_type = nb_type

    if input_is_2d:
        # Numba complains even though a 1d C-contiguous array is the same as an F-contiguous one
        compiled_f = njit(nb_out_type(Array(nb_type, 1, 'F')))(f)

        if use_stencil:
            def generated_kernel(xs): return compiled_f(xs[-n:1])
            kernel = stencil(neighborhood = ((-n, 0),))(f_stencil_version if f_stencil_version else generated_kernel)

            @guvectorize([(nb_type[::1, :], nb_out_type[:, ::1])], '(n,m)->(n,m)', nopython = True)
            def wrapped(xs, out):
                for c in range(xs.shape[1]):
                    out[:, c] = kernel(xs[..., c])
                    for i in range(n): out[i, c] = compiled_f(xs[:i+1, c]) # Because of current stencil limitations ('constant' is the only func_or_mode option)
            def final(xs):
                res = np.zeros(xs.shape, dtype = np_out_type)
                wrapped(xs, res)
                return res
        else:
            @njit(nb_out_type[:, ::1](nb_type[::1, :]))
            def final(xs):
                res = np.zeros(xs.shape, dtype = np_out_type)
                for c in range(xs.shape[1]):
                    for i in range(n): res[i, c] = compiled_f(xs[:i+1, c])
                    for i in range(n, len(xs)): res[i, c] = compiled_f(xs[i-n:i+1, c])
                return res
    else:
        compiled_f = njit(nb_out_type(nb_type[::1]))(f)

        if use_stencil:
            def generated_kernel(xs): return compiled_f(xs[-n:1])
            kernel = stencil(neighborhood = ((-n, 0),))(f_stencil_version if f_stencil_version else generated_kernel)

            @guvectorize([(nb_type[::1], nb_out_type[::1])], '(n)->(n)', nopython = True)
            def wrapped(xs, out):
                out[:] = kernel(xs)
                for i in range(n): out[i] = compiled_f(xs[:i+1]) # Because of current stencil limitations ('constant' is the only func_or_mode option)
            def final(xs):
                res = np.zeros(len(xs), dtype = np_out_type)
                wrapped(xs, res)
                return res
        else:
            @njit(nb_out_type[::1](nb_type[::1]))
            def final(xs):
                res = np.zeros(len(xs), dtype = np_out_type)
                for i in range(n): res[i] = compiled_f(xs[:i+1])
                for i in range(n, len(xs)): res[i] = compiled_f(xs[i-n:i+1])
                return res

    return final


@cache
def roll2d(f, n = 3, nb_type = f8, nb_out_type = None, np_out_type = np.float64, output_is_2d = False):
    '''Produce a compiled function which roll-applies a 2d-to-scalar or 2d-to-1d(*) f to a 2d input array
        for the given window size, numba input type (output type too if different) and numpy output type.
        I.e. a function which applies f to [n,:]-slices of the input array and returns either a 1d or 2d array.
        Arguments to the produced function are expected to be C-contiguous
        (can impose it with np.ascontiguousarray and inspect it with typeof(ARRAY).layout == 'C'),
        and outputs are always C-contiguous.
            (*): The 2d-to-1d function should actually be (2d,1d)-to-void, i.e. it needs to take in the 1d output array as an argument
            and modify it in-place, returning nothing (providing pre-allocated output arrays this way drastically speeds up the process).

            NOTE: Unless willing to manually check the first n-1 entries for correctness when applying the produced function,
            avoid using explicit references to the slice elements by index (e.g. xs[-2,:] + xs[-1,:] + xs[0,:]) in f's definition;
            use collective operators instead (e.g. xs[-2:1,:].sum(), which is fine because slicing deals with missing entries).

        :param f: A 2d-to-scalar or (2d,1d)-to-void function, respectively compilable as nb_out_type(nb_type[:, ::1]) or void(nb_type[:, ::1], nb_out_type[::1]) if f_is_1d_void
        :param n: The window size (how many previous xs rows, including the current one, should be included in the processed slices)
        :param nb_type: The scalar numba type of f's (and the produced function's) input
        :param nb_out_type: Optional - The scalar numba type of f's (and the produced function's) output; provide only if different from nb_type
        :param np_out_type: The scalar numpy type of f's (and the produced function's) output (required for array allocation, while the input one is not)
        :param output_is_2d: Whether, rather than scalar, f's output is void but it takes a 1d pre-allocated array which it fills with the true output (see type signatures in f description), leading to the produced function's output being 2d
        :returns: A function applying f to every [n,:]-slice of the input array, with signature nb_out_type[::1](nb_type[:, ::1]) or nb_out_type[:, ::1](nb_type[:, ::1]) if f_is_1d_void'''
    n -= 1
    if not nb_out_type: nb_out_type = nb_type

    if output_is_2d:
        compiled_f = njit((nb_type[:, ::1], nb_out_type[::1]))(f)

        @njit(nb_out_type[:, ::1](nb_type[:, ::1]))
        def final(xs):
            res = np.zeros(xs.shape, dtype = np_out_type)
            for i in range(n): compiled_f(xs[:i+1, ...], res[i, :]) # Ellipsis here and below due to https://github.com/numba/numba/issues/8131
            for i in range(n, xs.shape[0]): compiled_f(xs[i-n:i+1, ...], res[i, :])
            return res
    else:
        compiled_f = njit(nb_out_type(nb_type[:, ::1]))(f)

        @njit(nb_out_type[::1](nb_type[:, ::1]))
        def final(xs):
            res = np.zeros(xs.shape[0], dtype = np_out_type)
            for i in range(n): res[i] = compiled_f(xs[:i+1, ...]) # Ellipsis here and below due to https://github.com/numba/numba/issues/8131
            for i in range(n, xs.shape[0]): res[i] = compiled_f(xs[i-n:i+1, ...])
            return res

    return final


