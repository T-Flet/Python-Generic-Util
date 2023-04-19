'''Functions covering typical code-timing scenarios,
such as a "with" statement context, an n-executions timer,
and a convenient function for comparing and summarising n execution times of different implementations
of the same function.'''

import time
from contextlib import contextmanager
from pandas import DataFrame, concat

from typing import Callable


@contextmanager
def time_context(name: str = None):
    '''"with" statement context for timing the execution of the enclosed code block, i.e. `with time_context('Name of code block'): ...` '''
    start = time.perf_counter()
    yield # No need to yield anything or for the above to be in a "try" and the below in a customary "finally" since there is no dangling resource
    end = time.perf_counter()
    print(f'{name if name else "The timed code"} took {end - start:.3f}s')#\t\t(start: {start:.3f}, end: {end:.3f})')


def time_n(f: Callable, n = 2, *args, **kwargs):
    '''Run f (with given arguments) n times and return the execution intervals'''
    ts = []
    for i in range(n):
        ts.append(time.perf_counter())
        f(*args, **kwargs)
        ts[-1] = time.perf_counter() - ts[-1]
    return ts


# TODO: Write a class wrapping some function or object, recording (and by default printing call-time intervals):
#  - fields:
#       - the wrapped object/function
#       - a label to refer to the wrapped object when printing about it
#       - time of last call
#       - list of time differences between calls
#       -? worth recording the nature of calls in case the wrapped object is a function?
#  - methods:
#       - to take in a label and print-and-record when they are
#  executed relative to the last call.


# TODO: compare_implementations improvements: add options to
#   - break up the n iterations into m subcalls to compare_implementations (with final merge_bench_tables)
#       - option to set verbose to False in subcalls
#   - have the subcall n arguments be not uniform
#   - have the order of the functions in the subcall change to all permutations (or as many fit in the subcall total)
#   - reflect these two subcall characteristics (n value and function order) in the naming of the final merge_bench_tables
#   - Produce another table with only the averages all things considered (i.e. identical to a single-call call)
#   - Produce another table with statistics or scores on
#       - how well each function performed in each position
#       - how often the function at each position won (to see whether ordering had a big effect)
#       - how well each function performed in each sample size
#       - no need to check how smaller-sample calls were better, as they should be worse than large-sample ones
#   - MAYBE NOT: option to check whether the outputs of (single) calls to each function are equal
#       - Difficulty: not all objects are amenable to ==; could ask for the comparison function, but finnicky
def compare_implementations(fs_with_shared_args: dict[str, Callable], n = 200, wait = 1, verbose = True,
                            fs_with_own_args: dict[str, tuple[Callable, list, dict]] = None, args: list = None, kwargs: dict = None):
    '''Benchmark multiple implementations of the same function called n times (each with the same args and kwargs), with a break between functions.
    Recommended later output view if verbose is False: `print(table.to_markdown(index = False))`.
    :param fs_with_own_args: alternative to fs_with_shared_args, args and kwargs arguments: meant for additional functions taking different *args and **kwargs.'''
    assert n >= 3
    table = []
    for name, f in fs_with_shared_args.items():
        time.sleep(wait)
        if args:
            if kwargs: times = time_n(f, n, *args, **kwargs)
            else: times = time_n(f, n, *args)
        elif kwargs: times = time_n(f, n, **kwargs)
        else: times = time_n(f, n)
        table.append([name, sum(times) / len(times), sum(times[1:]) / (len(times)-1), times[0], times[1], times[2]])
        if verbose: print(f'Benchmarked {name} - mean {table[-1][1]} and mean excluding 1st run {table[-1][2]}')

    if fs_with_own_args:
        for name, f_a_k in fs_with_own_args.items():
            f, args, kwargs = f_a_k
            time.sleep(wait)
            if args:
                if kwargs: times = time_n(f, n, *args, **kwargs)
                else: times = time_n(f, n, *args)
            elif kwargs: times = time_n(f, n, **kwargs)
            else: times = time_n(f, n)
            table.append([name, sum(times) / len(times), sum(times[1:]) / (len(times)-1), times[0], times[1], times[2]])
            if verbose: print(f'Benchmarked {name} - mean {table[-1][1]} and mean excluding 1st run {table[-1][2]}')

    table = sorted(table, key = lambda row: row[1])

    last, last1 = table[0][1], table[0][2]
    table = [(name, m0, m1, m0 / table[0][1], m1 / table[0][2], next_mean_ratio, next_mean_ratio1, t0, t1, t2)
             for name, m0, m1, t0, t1, t2 in table
             if (next_mean_ratio := m0 / last) if (next_mean_ratio1 := m1 / last1)
             if (last := m0) if (last1 := m1)]

    df = DataFrame(table, columns = ['f', 'mean', 'mean excl. 1st', 'best mean ratio', 'best mean1 ratio', 'next mean ratio', 'next mean1 ratio', 't0', 't1', 't2'])
    if verbose: print('\n', df.to_markdown(index = False), sep = '')
    return df


def merge_bench_tables(*tables, verbose = True):
    table = concat(tables).sort_values(by = 'mean').reset_index(drop = True)
    table['best mean ratio'] = table['mean'] / table['mean'].values[0]
    table['best mean1 ratio'] = table['mean excl. 1st'] / table['mean excl. 1st'].values[0]
    table['next mean ratio'] = table['mean'] / table['mean'].shift()
    table['next mean1 ratio'] = table['mean excl. 1st'] / table['mean excl. 1st'].shift()
    table.loc[0, 'next mean ratio'], table.loc[0, 'next mean1 ratio'] = 1, 1
    if verbose: print('\n', table.to_markdown(index = False), sep = '')
    return table


