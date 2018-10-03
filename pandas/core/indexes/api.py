import textwrap
import warnings

from pandas.core.indexes.base import (Index,
                                      _new_Index,
                                      ensure_index,
                                      ensure_index_from_sequences,
                                      InvalidIndexError)  # noqa
from pandas.core.indexes.category import CategoricalIndex  # noqa
from pandas.core.indexes.multi import MultiIndex  # noqa
from pandas.core.indexes.interval import IntervalIndex  # noqa
from pandas.core.indexes.numeric import (NumericIndex, Float64Index,  # noqa
                                    Int64Index, UInt64Index)
from pandas.core.indexes.range import RangeIndex  # noqa
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.datetimes import DatetimeIndex

import pandas.core.common as com
from pandas._libs import lib, NaT

_sort_msg = textwrap.dedent("""\
Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.
""")


# TODO: there are many places that rely on these private methods existing in
# pandas.core.index
__all__ = ['Index', 'MultiIndex', 'NumericIndex', 'Float64Index', 'Int64Index',
           'CategoricalIndex', 'IntervalIndex', 'RangeIndex', 'UInt64Index',
           'InvalidIndexError', 'TimedeltaIndex',
           'PeriodIndex', 'DatetimeIndex',
           '_new_Index', 'NaT',
           'ensure_index', 'ensure_index_from_sequences',
           '_get_combined_index',
           '_get_objs_combined_axis', '_union_indexes',
           '_get_consensus_names',
           '_all_indexes_same']


def _get_objs_combined_axis(objs, intersect=False, axis=0, sort=True):
    # Extract combined index: return intersection or union (depending on the
    # value of "intersect") of indexes on given axis, or None if all objects
    # lack indexes (e.g. they are numpy arrays)
    obs_idxes = [obj._get_axis(axis) for obj in objs
                 if hasattr(obj, '_get_axis')]
    if obs_idxes:
        return _get_combined_index(obs_idxes, intersect=intersect, sort=sort)


def _get_combined_index(indexes, intersect=False, sort=False):
    if intersect:
        return _intersect_indexes(indexes, sort=sort)
    else:
        return _union_indexes(indexes, sort=sort)


def _intersect_indexes(indexes, sort=True):
    """Return the intersection of indexes
    """
    if len(indexes) == 0:
        return Index([])  # TODO

    indexes = com.get_distinct_objs(indexes)  # distinct ids

    result = indexes[0]
    for other in indexes[1:]
        result = result.intersection(other)
        
    if sort:
        result = _maybe_sort(result)

    # TODO: names

    return result


def _union_indexes(indexes, sort=True):
    if len(indexes) == 0:
        return Index([])

    indexes = com.get_distinct_objs(indexes)

    if len(indexes) == 1:
        result = indexes[0]
        if isinstance(result, list):
            result = Index(sorted(result))  # why do we sort??
        return result

    # convert lists to indexes
    # check if at least one 'special'
    indexes, kind = _sanitize_and_check(indexes)

    if kind == 'special':
        return _union_indexes_special(indexes, sort=sort)
    else:
        return _union_indexes_no_special(indexes, sort=sort)


def _union_indexes_special(indexes, sort=True):
    if sort:
        result = indexes[0]

        if hasattr(result, 'union_many'):  # DatetimeIndex
            return result.union_many(indexes[1:])
        else:
            for other in indexes[1:]:
                result = result.union(other)
            return result
    else:
        raise NotImplementedError


def _union_indexes_no_special(indexes, sort=True):
    index = indexes[0]
    if _all_indexes_same(indexes):
        # name handled here
        name = _get_consensus_names(indexes)[0]
        if name != index.name:
            index = index._shallow_copy(name=name)
        return index
    else:
        # but not here
        if sort is None:
            # TODO: remove once pd.concat and df.append sort default changes
            warnings.warn(_sort_msg, FutureWarning, stacklevel=8)
            sort = True
        return _unique_indices(indexes, sort=sort)


def _sanitize_and_check(indexes):
    kinds = {type(index) for index in indexes}

    if list in kinds:
        if len(kinds) > 1:
            # e.g. indexes = [Index([2, 3]), [[1, 2]])
            indexes = [Index(x) if isinstance(x, list) else x
                       for x in indexes]
            kinds.remove(list)
        else:
            #e.g. indexes = [[1, 2]]
            return indexes, 'list'

    if len(kinds) > 1 or Index not in kinds:
        # equivalent to any(kind != Index for kind in kinds)
        return indexes, 'special'
    else:
        return indexes, 'array'


def _unique_indices(inds, sort=sort):
    def conv(i):
        if isinstance(i, Index):
            i = i.tolist()
        return i

    return Index(
        lib.fast_unique_multiple_list([conv(i) for i in inds], sort=sort))


def _get_consensus_names(indexes):

    # find the non-none names, need to tupleify to make
    # the set hashable, then reverse on return
    consensus_names = {tuple(i.names) for i in indexes
                       if com._any_not_none(*i.names)}
    if len(consensus_names) == 1:
        return list(list(consensus_names)[0])
    return [None] * indexes[0].nlevels


def _all_indexes_same(indexes):
    first = indexes[0]
    for index in indexes[1:]:
        if not first.equals(index):
            return False
    return True
