'''Functions with less generic purpose than in the above; currently mostly to do with min/max-based operations.'''


from typing import TypeVar, Callable, Union, Sequence, Iterable, Iterator, Generator, Any, Generic, Mapping
_a = TypeVar('_a')
_b = TypeVar('_b')


## Min/Max-Based Functions

def interval_overlap(ab: tuple[float, float], cd: tuple[float, float]) -> float:
    '''Compute the overlap of two intervals (expressed as tuples of start and end values)'''
    return max(0., min(ab[1], cd[1]) - max(ab[0], cd[0]))

def min_max(xs: Sequence[_a]) -> tuple[_a, _a]:
    '''Mathematically most efficient joint identification of min and max (minimum comparisons = 3n/2 - 2).
        Note:
            - This function is numba-compilable, e.g. as `njit(nTup(f8,f8)(f8[::1]))(min_max)` (see `Generic_Util.numba.types` for `nTup` shorthand),
            - If using numpy arrays, min and max are cached for O(1) lookup, and one would imagine this is the used algorithm'''
    if xs[0] > xs[1]: min, max = xs[1], xs[0] # Initialise
    else: min, max = xs[0], xs[1]

    i, even_len = 2, len(xs) - 1 if len(xs) % 2 else len(xs)
    while i < even_len: # Sort every consecutive pair (skip first pair and ignore last if len is odd), then only compare mins and maxes separately
        if xs[i] > xs[i+1]:
            if xs[i+1] < min: min = xs[i+1]
            if xs[i]   > max: max = xs[i]
        else:
            if xs[i]   < min: min = xs[i]
            if xs[i+1] > max: max = xs[i+1]
        i += 2

    if len(xs) % 2: # If len is odd compare both with last
        if   xs[-1] < min: min = xs[-1]
        elif xs[-1] > max: max = xs[-1]
    return min, max


