# Generic-Util

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI Version][pypi-version]][pypi-link]
[![PyPI Downloads][pypi-downloads-badge]][pypi-downloads-link]
[![Python Versions][python-versions-badge]][python-versions-link]
[![License][license-badge]][license-link]
[![Actions Status][actions-badge]][actions-link]

[//]: # ([![Conda-Forge][conda-badge]][conda-link])
[//]: # ([![PyPI platforms][pypi-platforms]][pypi-link])
[//]: # ([![GitHub Discussion][github-discussions-badge]][github-discussions-link])
[//]: # ([![Gitter][gitter-badge]][gitter-link])

This package contains convenient functions not found in Python's Standard Library
(they would mostly fall within
[itertools](https://docs.python.org/3/library/itertools.html),
[functools](https://docs.python.org/3/library/functools.html),
[operator](https://docs.python.org/3/library/operator.html), and
[time](https://docs.python.org/3/library/time.html)).
It also contains a variety of convenience functions for the
[numba](https://numba.pydata.org/) JIT compiler library.

See the [documentation][rtd-link] for details, but functions are grouped as follows:
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

Then a sub-package is dedicated to utility functions for the [numba](https://numba.pydata.org/) JIT compiler library:
- Generic_Util.numba.benchmarking: functions comparing execution times of (semi-automatically-generated) varieties of
    numba-compilations of a given function, including
    lazy vs eager compilation, vectorisation, parallelisation, as well as varieties of rolling (see Generic_Util.numba.higher_order).
- Generic_Util.numba.higher_order: higher-order numba-compilation functions, currently only functions to "roll" simpler functions
  (1d-to-scalar or 2d-to-scalar/1d) over arrays, with a few combinations of input and output type signatures.
- Generic_Util.numba.types: convenient shorthands for frequently used numba (and respective numpy) types, with a focus on
    C-contiguity of arrays; these are useful in declaring eager-compilation function signatures.

Many functions which would have been included in this package were dropped in favour of using those in the wonderful
[more-itertools](https://github.com/more-itertools/more-itertools) package
(and where overlaps remain, there are convenient differences).
Separately, although used only in one instance in this repository,
the [sortedcontainers](https://grantjenks.com/docs/sortedcontainers/) package is another great
source of algorithm-simplifying ingredients.


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/T-Flet/Generic-Util/workflows/CI/badge.svg
[actions-link]:             https://github.com/T-Flet/Generic-Util/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/Generic-Util
[conda-link]:               https://github.com/conda-forge/Generic-Util-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/T-Flet/Generic-Util/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/T-Flet/Generic-Util/community.svg
[gitter-link]:              https://gitter.im/https://github.com/T-Flet/Generic-Util/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/Generic-Util/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/Generic-Util
[pypi-version]:             https://img.shields.io/pypi/v/Generic-Util
[pypi-downloads-badge]:     https://pepy.tech/badge/Generic-Util
[pypi-downloads-link]:      https://pepy.tech/project/Generic-Util
[python-versions-badge]:    https://img.shields.io/pypi/pyversions/Generic-Util.svg
[python-versions-link]:     https://pypi.python.org/pypi/Generic-Util/
[rtd-badge]:                https://readthedocs.org/projects/python-generic-util/badge/?version=latest
[rtd-link]:                 https://python-generic-util.readthedocs.io/en/latest/?badge=latest
[license-badge]:            https://img.shields.io/pypi/l/Generic-Util.svg
[license-link]:             https://github.com/T-Flet/Generic-Util/blob/master/LICENSE

<!-- prettier-ignore-end -->
