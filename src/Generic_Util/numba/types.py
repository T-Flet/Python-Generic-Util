'''Convenient shorthands for frequently used numba (and respective numpy) types, with a focus on
C-contiguity of arrays; these are useful in declaring eager-compilation function signatures.'''

from numba import njit, vectorize, guvectorize, stencil, typeof, b1, f8, i8, i1
from numba.core.types.npytypes import UnicodeCharSeq
from numba.core.types import Array
from numba.core.types.containers import Tuple, List
import numba.typed as nt

from typing import Union
from numpy.typing import NDArray
import numpy as np


## This script serves to export common numba functions and type signatures easily and without namespace pollution
__all__ = ['njit', 'vectorize', 'guvectorize', 'stencil', 'typeof',
                # USEFUL NOTE: typeof is not helpful if called on functions; use their .nopython_signatures method instead
           'b1', 'f8', 'i8', 'i1', # The basic types used in the project
           'b1_NP', 'f8_NP', 'i8_NP', 'i1_NP', # Matching numpy versions
           'b1A', 'f8A', 'i8A', 'i1A', # Shorthand for their C-consecutive 1d arrays
           'b1A2', 'f8A2', 'i8A2', 'i1A2', # Shorthand for their C-consecutive 2d arrays
                # NOTE: If an array is made from a pandas dataframe which was created by assembling columns, it will instead be
                #   F-contiguous (both from HOMOGENEOUS_DF.values or np.array(HOMOGENEOUS_DF, dtype = ch_NP)).
                #   Therefore both for uniformity and for numba processing speed and convenience it is best to apply
                #   np.ascontiguousarray to it before feeding it to compiled functions.
                #   The speed & convenience reason is the following:
                #       - If a 2D array is C-contiguous, row-slicing does not affect it, but slices including column slicing are non-contiguous
                #       - If a 2D array is F-contiguous, column-slicing does not affect it, but slices including row slicing are non-contiguous
                #       - Interesting notes:
                #           - if the problematic slices in either above cases are empty, then, understandably, C-contiguity comes out
                #           - if the problematic slices in either above cases include out-of-bounds indices, then, somewhat understandably, the original contiguity comes out
                #   So mention that if a df is constructed column by column it will be F and therefore need transforming to C
           'b1A_NP', 'f8A_NP', 'i8A_NP', 'i1A_NP', 'b1A2_NP', 'f8A2_NP', 'i8A2_NP', 'i1A2_NP', # Matching numpy versions
           'ch', # A single unicode character
           'chA', 'chA2', # Arrays of characters (C-consecutive 1d and F-consecutive 2d)
                # NOTE: To construct compliant empty ones within compiled functions, use np.zeros(SHAPE, dtype = ch_NP)
           'ch_NP', 'chA_NP', 'cA2_NP', # Matching numpy versions
           'nTup', 'nList', 'Union', # numba's Tuple and List type constructors (renamed to nTup and nList to avoid clashes) and typing's Union
                # NOTE: Since tuples have fixed lengths, all entry types need to be provided (numba does provide UniTuple for shorthand for uniform
                #       tuples, but that is not imported here); on the other hand, lists ARE uniform in numba, so only one type is required
           'aList', # numba's concrete list constructor, i.e. for actual arguments, not their signatures
                # NOTE: nList and aList are provided but discouraged, since they are significantly slower (e.g. 10x) than finding some alternative function structure
           'a1dF'] # A shorthand for declaring 1d F-consecutive arrays (which are the same as C-consecutive ones, but numba can complain anyway)


b1A = b1[::1]
f8A = f8[::1]
i8A = i8[::1]
i1A = i1[::1]

b1A2 = b1[:, ::1]
f8A2 = f8[:, ::1]
i8A2 = i8[:, ::1]
i1A2 = i1[:, ::1]

# To get numpy's typechar of these: np.dtype(ANY_OF_THEM).char # Table here: https://numpy.org/doc/stable/reference/generated/numpy.typename.html
# To declare scalars of these types: np.dtype(ANY_OF_THEM_OR_THEIR_STRING_NAME).type(VALUE)
b1_NP = np.bool8
f8_NP = np.float64
i8_NP = np.int64
i1_NP = np.int8

b1A_NP = b1A2_NP = NDArray[b1_NP]
f8A_NP = f8A2_NP = NDArray[f8_NP]
i8A_NP = i8A2_NP = NDArray[i8_NP]
i1A_NP = i1A2_NP = NDArray[i1_NP]

ch = UnicodeCharSeq(1)
chA = ch[::1] # Same as typeof(np.array(['a', '']))
chA2 = ch[:, ::1] # Same as typeof(np.array([['a', ''],['','b']]))

ch_NP = '<U1'
chA_NP = cA2_NP = NDArray[ch_NP]

def nTup(*args): return Tuple(args)
def nList(dtype, reflected = False, initial_value = None): return List(dtype, reflected, initial_value)

aList = nt.List

def a1dF(nb_type): return Array(nb_type, 1, 'F')


