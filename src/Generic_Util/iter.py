'''Iterable-focussed functions, covering multiple varieties of
flattening, iterable combining,
grouping and predicate/property-based processing (including topological sorting),
element-comparison-based operations, value interspersal, and finally batching.'''

from itertools import chain, combinations, islice
from functools import reduce
from collections import defaultdict
from sortedcontainers import SortedKeyList
import operator as op
import numpy as np
from numpy.typing import NDArray

from typing import TypeVar, Callable, Union, Sequence, Iterable, Iterator, Generator, Any, Generic, Mapping
_a = TypeVar('_a')
_b = TypeVar('_b')


## Flattening Functions

def flatten(xss: Iterable[Iterable], as_generator = False) -> list | Generator:
    '''Standard flattening (returning a list by default).'''
    return chain.from_iterable(xss) if as_generator else list(chain.from_iterable(xss))

def deep_flatten(xss_: Iterable) -> Generator:
    '''Flatten any nested combination of iterables to its leaves.
        The flattening ignores keys of Mappings and stops at tuples, strings, bytes or non-Iterables.
        Same as a recursive chain.from_iterable but handling values of Mappings instead of keys.'''
    if isinstance(xss_, Iterable) and not isinstance(xss_, (tuple, str, bytes)):
        for xs_ in xss_.values() if isinstance(xss_, Mapping) else xss_: yield from deep_flatten(xs_)
    else: yield xss_

def deep_extract(xss_: Iterable[Iterable], *key_path) -> Generator:
    '''Given a nested combination of iterables and a path of keys in it, return a deep_flatten-ed list of entries under said path.
        Note: `deep_extract(xss_, *key_path) == deep_flatten(Generic_Util.operator.get_nested(xss_, *key_path))`'''
    level = xss_
    for k in key_path: level = level[k]
    return deep_flatten(level)



## Iterable-Combining (and Combinatory) Functions

def unzip(list_of_ntuples: Iterable[Iterable], as_generator = False) -> list[list] | Generator[list, None, None]:
    '''Standard unzip (i.e. zip(*...)) but with concretised outputs (as lists).'''
    return (list(t) for t in zip(*list_of_ntuples)) if as_generator else [list(t) for t in zip(*list_of_ntuples)]

def zip_maps(*maps: list[Mapping[_a, _b]]) -> Generator[tuple[_a, tuple], None, None]:
    for key in reduce(set.intersection, map(set, maps)): yield key, tuple(map(op.itemgetter(key), maps))

def update_dict_with(d0: dict, d1: dict, f: Callable[[_a, _b], Union[_a, _b]]) -> dict[Any, Union[_a, _b]]:
    '''Update a dictionary's entries with those of another using a given function, e.g. appending (operator.add is ideal for this).
    NOTE: This modifies d0, so a prior copy or deepcopy is advisable.
    '''
    for k, v in d1.items(): d0[k] = f(d0[k], v) if k in d0 else v
    return d0

def all_combinations(xs: Sequence, min_size = 1, max_size = None) -> list:
    '''Produce all combinations of elements of xs between min_size and max_size subset size (i.e. the powerset elements of certain sizes).'''
    return list(chain.from_iterable(combinations(xs, i) for i in range(min_size, (max_size if max_size else len(xs)) + 1)))



## Predicate/Property-Based Functions

def partition(p: Callable[[_a], bool], xs: Iterable[_a]) -> tuple[Iterable[_a], Iterable[_a]]:
    '''Haskell's partition function, partitioning xs by some boolean predicate p: `partition p xs == (filter p xs, filter (not . p) xs)`.'''
    acc = ([],[])
    for x in xs: acc[not p(x)].append(x)
    return acc

def group_by(f: Callable[[_a], _b], xs: Iterable[_a]) -> dict[_b, list[_a]]:
    '''Generalisation of partition to any-output key-function.
        Notes:
            - 'Retrieval' functions from the operator package are typical f values (`itemgetter(...)`, `attrgetter(...)` or `methodcaller(...)`)
            - This is NOT Haskell's groupBy function'''
    acc = defaultdict(list)
    for x in xs: acc[f(x)].append(x)
    return acc

def first(c: Callable[[_a], bool], xs: Iterable[_a], default: _a = None) -> _a:
    '''Return the first value in xs which satisfies condition c.'''
    return next((x for x in xs if c(x)), default)

def foldq(f: Callable[[_b, _a], _b], g: Callable[[_b, _a, list[_a]], list[_a]], c: Callable[[_a], bool], xs: Sequence[_a], acc: _b) -> tuple[_b, list[_a]]:
    '''
    Fold-like higher-order function where xs is traversed by consumption conditional on c, and remaining xs are updated by g
    (therefore consumption order is not known a priori):
      - the first/next item to be ingested is the first in the remaining xs to fulfil condition c
      - at every x ingestion, the item is removed from (a copy of) xs, and all the remaining ones are potentially modified by function g
      - this function always returns a tuple of `(acc, remaining_xs)`, unlike the stricter `foldq_`, which raises an exception for leftover xs

            Note: `fold(f, xs, acc) == foldq(f, lambda acc, x, xs: xs, lambda x: True, xs, acc)`.

            Sequence of suitable names leading to the current one: consumption_fold, condition_update_fold, cu_fold, q_fold, qfold or foldq
    :param f: 'Traditional' fold function :: acc -> x -> acc
    :param g: 'Update' function for all remaining xs at every iteration :: acc -> x -> xs -> xs
    :param c: 'Condition' function to select the next x (first which satisfies it) :: x -> Bool
    :param xs: Structure to consume
    :param acc: Starting value for the accumulator
    :returns: (acc, remaining_xs)
    '''
    #####################################
    # ALTERNATIVE (BETTER?) BASE VERSION:
    #   do not provide a condition c but a more generic function h which takes in the xs and returns selected x and new remaining xs
    #   This removes the non-deterministic-ness from this function and leaves it to h (e.g. h selecting the first x satisfying c is so)
    #####################################
    xs = list(xs) # Copy xs in order not to modify the actual input
    def full_step(acc, xs): # Alternative implementation: move function content inside the while and use a 'broke' flag to trigger a continue before the raise
        for i in range(len(xs)):
            x = xs[i]
            if c(x):
                del xs[i]
                return f(acc, x), g(acc, x, xs)
        return None
    while xs:
        if (res := full_step(acc, xs)): acc, xs = res
        else: break
    return acc, xs

def foldq_(f: Callable[[_b, _a], _b], g: Callable[[_b, _a, list[_a]], list[_a]], c: Callable[[_a], bool], xs: list[_a], acc: _b) -> _b:
    r'''Stricter version of foldq (see its description for details); only returns the accumulator and raises an exception on leftover xs.
    :raises ValueError on leftover xs'''
    acc, xs = foldq(f, g, c, xs, acc)
    if xs: raise ValueError('No suitable next element found for given condition while elements remain')
    else: return acc



## Sorting Functions

def topological_sort(nodes_incoming_edges_tuples: Iterable[tuple[_a, list[_b]]]) -> list[_a]:
    '''Topological sort, i.e. sort (non-uniquely) DAG nodes by directed path, e.g. sort packages by dependency order.'''
    return foldq_(lambda acc, x: acc + [x[0]],
                  lambda acc, x, xs: [(a, [d for d in deps if d != x[0]]) for a, deps in xs],
                  lambda x: not x[1], nodes_incoming_edges_tuples, [])



## Element-Comparison Functions

# Appearance-order-preserving unique/distinct functions
def unique(xs: Iterable[_a]) -> list[_a]:
    '''Order-preserving uniqueness. If the values of xs are hashable, use unique_a instead.'''
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if x not in seen and not seen.append(x)] # Neat short-circuit 'and' trick

def unique_a(xs: NDArray) -> NDArray:
    '''Order-preserving uniqueness for an array (of hashable objects).
    Much faster than a compiled version of the non-hashing-using algorithm, even including the final sort.'''
    _, inds = np.unique(xs, return_index = True)
    return xs[np.sort(inds)]

def unique_by(f: Callable[[_a], Any], xs: Iterable[_a]) -> list[_a]:
    '''Order-preserving uniqueness by some property f. If the values of xs are hashable, it is recommended to use unique_a.'''
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if (fx := f(x), ) if fx not in seen and not seen.append(fx)] # Neat true-tuple assignment and neat short-circuit 'and' trick

def eq_elems(xs: Iterable[_a], ys: Iterable[_a]) -> bool:
    '''Equality of iterables by their elements'''
    cys = list(ys) # make a mutable copy
    try:
        for x in xs: cys.remove(x)
    except ValueError: return False
    return not cys

def diff(xs: Iterable[_a], ys: Iterable[_a]) -> list[_a]:
    '''Difference of iterables.
        Notes:
            - not a set difference, so strictly removing as many xs duplicate entries as there are in ys
            - preserves order in xs'''
    cxs = list(xs) # make a mutable copy
    try:
        for y in ys: cxs.remove(y)
    except ValueError: pass
    return cxs



## Interspersing Functions

def intersperse(xs: Sequence[_a], ys: list[_a], n: int, prepend = False, append = False) -> Sequence[_a]:
    '''Intersperse elements of ys every n elements of xs.'''
    n += 1 # Moving this after the assert would save the two (n - 1)s, but this way all n expressions are coherent
    unwanted_append = not len(xs) % (n - 1) and not append
    assert len(ys) >= (m := prepend + len(xs) // (n - 1) - unwanted_append), f'ys has too few elements ({len(ys)}); at least {m} are needed to cover xs with the given parameters'
    if not prepend: ys.insert(0, None) # The +-1s below are respectively for: indices starting at 0, the prepended y, context
    return [xs[i - 1 - i // n] if i % n else ys[i // n] for i in range(1 + len(xs) + len(xs) // (n - 1) - unwanted_append)][not prepend:]

def intersperse_val(xs: Sequence[_a], y: _a, n: int, prepend = False, append = False) -> Sequence[_a]:
    '''Intersperse y every n elements of xs.'''
    n += 1 # The +-1s below are respectively for: indices starting at 0, the prepended y, context
    res = [xs[i - 1 - i // n] if i % n else y for i in range(1 + len(xs) + len(xs) // (n - 1))]
    return res[not prepend : len(res) - (not len(xs) % (n - 1) and not append)]



## Batching functions

def batch_iter(n: int, xs: Iterable[_a]) -> Generator[_a, None, None]:
    '''Batch an iterable in batches of size n (possibly except the last). If len(xs) is knowable use batch_seq instead.'''
    iterator = iter(xs)
    while batch := list(islice(iterator, n)): yield batch

def batch_iter_by(by: Callable[[_a], float], size: int, xs: Iterable[_a], keep_totals = False) -> Generator[_a, None, None]:
    '''Batch an iterable by sum of some weight/score/cost of each element, with no batch exceeding size.
    :param keep_totals: if True, the function returns tuples of (batch_total_value, batch) instead of just batches'''
    batch, count = [], 0
    if keep_totals:
        for x in xs:
            if count + (weight := by(x)) > size:
                yield count, batch
                batch, count = [], 0
            batch.append(x)
            count += weight
        if batch: yield count, batch
    else:
        for x in xs:
            if count + (weight := by(x)) > size:
                yield batch
                batch, count = [], 0
            batch.append(x)
            count += weight
        if batch: yield batch

def batch_seq(n: int, xs: Sequence[_a], n_is_number_of_batches = False) -> Generator[_a, None, None]:
    '''Batch an iterable of knowable length in batches of size n (possibly except the last).
    :param n_is_number_of_batches: If True, divide into n batches of equal length (possibly except the last) rather than in batches of n elements'''
    return (xs[i:i + n] for i in range(0, len(xs), len(xs) // n if n_is_number_of_batches else n))

def batch_seq_by_into(by: Callable[[_a], float], k: int, xs: Sequence[_a], keep_totals = False, optimal_but_reordered = False) -> Generator[_a, None, None]:
    '''Batch an iterable of knowable length into k batches containing elements whose sum of some weight/score/cost (by) is roughly equal.
    :param keep_totals: if True, the function returns tuples of (batch_total_value, batch) instead of just batches
    :param optimal_but_reordered: if True, the function produces the optimal (hence not order-preserving) batching
        of summable xs into a given number of batches containing roughly equal totals.
        If False, the process retains order but may return more batches than requested.
    '''
    assert k <= len(xs), 'The requested batches should be fewer than the number of elements'
    if k == 1: yield xs if keep_totals else (sum(xs), xs)
    elif k == len(xs): yield from ([x] for x in xs) if keep_totals else ((x, [x]) for x in xs)

    if optimal_but_reordered:
        batches = SortedKeyList([[0, []] for _ in range(k)], op.itemgetter(0)) # Always avoid comparing lists directly
        for x in sorted(xs, key = by, reverse = True): # Sort is expensive but guarantees optimality
            min_batch = batches.pop(0)
            min_batch[0] += by(x)
            min_batch[1].append(x) # No advantage (usually disadvantage) in keeping batches separately and keeping their index here instead
            batches.add(min_batch)
        yield from (tuple(batch) for batch in batches) if keep_totals else (batch[1] for batch in batches)
    else:
        size = sum(weights := [by(x) for x in xs]) // k
        batch, count = [], 0
        if keep_totals:
            for x, weight in zip(xs, weights):
                if count + weight > size:
                    yield count, batch
                    batch, count = [], 0
                batch.append(x)
                count += weight
            if batch: yield count, batch
        else:
            for x, weight in zip(xs, weights):
                if count + weight > size:
                    yield batch
                    batch, count = [], 0
                batch.append(x)
                count += weight
            if batch: yield batch


