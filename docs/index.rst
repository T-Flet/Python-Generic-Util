
Welcome to the documentation!
=============================


Introduction
------------

This package contains convenient functions not found in Python's Standard Library
(they would mostly fall within
`itertools <https://docs.python.org/3/library/itertools.html>`_,
`functools <https://docs.python.org/3/library/functools.html>`_,
`operator <https://docs.python.org/3/library/operator.html>`_, and
`time <https://docs.python.org/3/library/time.html>`_).
It also contains a variety of convenience functions for the
`numba <https://numba.pydata.org/>`_ JIT compiler library.

The functions are grouped as follows:

- Generic_Util.benchmarking: functions covering typical code-timing scenarios,
  such as a "with" statement context, an n-executions timer,
  and a convenient function for comparing and summarising n execution times of different implementations
  of the same function.
- Generic_Util.iter: iterable-focussed functions, covering multiple varieties of
  flattening, iterable combining,
  grouping and predicate/property-based processing (including topological sorting),
  element-comparison-based operations, value interspersal, and finally batching.
- Generic_Util.operator: functions regarding item retrieval, and syntactic sugar for patterns of function application.
- Generic_Util.misc: functions with less generic purpose than in the above; currently mostly to do with min/max-based operations.

Many functions which would have been included here were dropped in favour of using those in the wonderful
`more-itertools <https://github.com/more-itertools/more-itertools>`_ package
(and where overlaps remain, there are convenient differences).
Separately, although used only in one instance in this repository,
the `sortedcontainers <https://grantjenks.com/docs/sortedcontainers/>`_ package is another great
source of algorithm-simplifying ingredients.

Then a sub-package is dedicated to utility functions for the `numba <https://numba.pydata.org/>`_ JIT compiler library:

- Generic_Util.numba.benchmarking: functions comparing execution times of (semi-automatically-generated) varieties of
  numba-compilations of a given function, including
  lazy vs eager compilation, vectorisation, parallelisation, as well as varieties of rolling (see Generic_Util.numba.higher_order).
- Generic_Util.numba.higher_order: higher-order numba-compilation functions, currently only functions to "roll" simpler functions
  (1d-to-scalar or 2d-to-scalar/1d) over arrays, with a few combinations of input and output type signatures.
- Generic_Util.numba.types: convenient shorthands for frequently used numba (and respective numpy) types, with a focus on
    C-contiguity of arrays; these are useful in declaring eager-compilation function signatures.





.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents
   :glob:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
