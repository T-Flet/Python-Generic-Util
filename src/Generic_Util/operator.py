'''Functions regarding item retrieval, and syntactic sugar for patterns of function application.'''

from typing import TypeVar, Callable, Union, Sequence, Iterable, Iterator, Generator, Any, Generic, Mapping
_a = TypeVar('_a')
_b = TypeVar('_b')


## Retrieval Functions

def fst(ab: tuple[_a, _b]) -> _a:
    '''Same as operator.itemgetter(0), but intended (and type-annotated) specifically for 2-tuple use'''
    return ab[0]

def snd(ab: tuple[_a, _b]) -> _b:
    '''Same as operator.itemgetter(1), but intended (and type-annotated) specifically for 2-tuple use'''
    return ab[1]

def get_nested(xss_: Iterable[Iterable], *key_path) -> Generator:
    '''Follow a path of keys for a nested combination of iterables'''
    level = xss_
    for k in key_path: level = level[k]
    return level



## Syntactic-Sugar Functions

def on(f: Callable, xs: Iterable[_a], g: Callable[[_a, ...], _b], *args, **kwargs):
    '''Transform xs by element-wise application of g and call f with them as its arguments.
        E.g. `on(operator.gt, (a, b), len)`
        Notes:
            - *args, **kwargs are for g, not f
            - 'Retrieval' functions from the operator package are reasonable g values (`itemgetter(...)`, `attrgetter(...)` or `methodcaller(...)`),
                BUT on_a and on_m are shorthands for the attribute and method cases'''
    return f(*[g(x, *args, **kwargs) for x in xs])

def on_a(f: Callable, xs: Iterable, a: str):
    '''Extract attribute a from xs elements and call f with them as its arguments.
        E.g. `on_a(operator.eq, [a, b], '__class__')`'''
    return f(*[getattr(x, a) for x in xs])

def on_m(f: Callable, xs: Iterable, m: str, *args, **kwargs):
    '''Call method m on xs elements and call f with their results as its arguments.
        E.g. `on_m(operator.gt, [a, b], 'count', 'hello')`
        Notes:
            - *args, **kwargs are for method m, not f'''
    return f(*[getattr(x, m)(*args, **kwargs) for x in xs])


