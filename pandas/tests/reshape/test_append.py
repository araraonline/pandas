import datetime as dt
from itertools import combinations, product

import numpy as np
import pytest

import pandas as pd
import pandas.util.testing as tm
from pandas import concat, DataFrame, Index, isna, Series, Timestamp
from pandas.compat import lrange
from pandas.core.indexes.base import InvalidIndexError
from pandas.util.testing import assert_frame_equal


indexes = [
    # indexes listed here must be sorted

    # base
    pd.Index(['A', 'B', 'C']),
    pd.Index(['A', 'B', 'C'], name='foo'),

    # numeric
    pd.RangeIndex(3),
    pd.Int64Index([3, 4, 5]),
    pd.UInt64Index([6, 7, 8]),
    pd.Float64Index([3.5, 4.5, 5.5]),
    pd.Index([9, 10, 11], dtype=object),  # fake int64

    # datetime
    pd.to_datetime(['2013-01-01', '2013-01-10', '2013-01-15']),
    pd.to_timedelta(['1 day', '2 days', '3 days']),
    pd.PeriodIndex(start='2000', periods=3),

    # interval
    pd.interval_range(start=0, end=3),

    # categorical
    pd.CategoricalIndex('A B C'.split()),
    pd.CategoricalIndex('D E F'.split(), ordered=True),

    # multi-index
    pd.MultiIndex.from_arrays(['A B C'.split(), 'D E F'.split()]),
]


indexes_with_dups = [
    # base
    pd.Index(['A', 'B', 'B']),
    pd.Index(['B', 'B', 'A']),
    pd.Index(['A', 'B', 'B'], name='foo'),
    pd.Index(['B', 'B', 'A'], name='bar'),

    # numeric
    pd.Index([9, 10, 10], dtype=object),
    pd.Int64Index([3, 4, 4]),
    pd.UInt64Index([6, 7, 7]),
    pd.Float64Index([3.5, 4.5, 4.5]),

    # datetime
    pd.to_datetime(['2013-01-01', '2013-01-10', '2013-01-10']),
    pd.to_timedelta(['1 day', '2 days', '2 days']),
    pd.PeriodIndex([2000, 2001, 2001], freq='A'),

    # interval
    pd.IntervalIndex.from_arrays([0, 1, 1], [1, 2, 2]),

    # categorical
    pd.CategoricalIndex('A B B'.split()),
    pd.CategoricalIndex('D E E'.split(), ordered=True),

    # multi-index
    pd.MultiIndex.from_arrays(['A B B'.split(), 'D E E'.split()]),
]


index_sort_groups = [
    # When indexes from the same group are joined, the result is sortable.
    # When indexes from different groups are joined, the result is not
    # sortable.

    [  # joining produces a string index
     pd.Index(['A', 'B', 'C']),
     pd.CategoricalIndex('A B C'.split()),
     pd.CategoricalIndex('D E F'.split(), ordered=True)],

    [  # numeric indexes
     pd.RangeIndex(3),
     pd.Int64Index([3, 4, 5]),
     pd.UInt64Index([6, 7, 8]),
     pd.Float64Index([3.5, 4.5, 5.5]),
     pd.Index([9, 10, 11], dtype=object)],

    [pd.to_datetime(['2013-01-01', '2013-01-10', '2013-01-15'])],
    [pd.to_timedelta(['1 day', '2 days', '3 days'])],
    [pd.PeriodIndex(start='2000', periods=3)],
    [pd.interval_range(start=0, end=3)],
    [pd.MultiIndex.from_arrays(['A B C'.split(), 'D E F'.split()])],
]


def cls_name(obj):
    return obj.__class__.__name__


@pytest.fixture(params=[True, False])
def sort(request):
    """Boolean sort keyword for DataFrame.append
    """
    return request.param


@pytest.fixture(params=[True, False, None])
def sort_with_none(request):
    """Boolean sort keyword for concat and DataFrame.append.

    Includes the default of None
    """
    # TODO: Replace with sort once keyword changes.
    return request.param


class TestAppendBasic(object):
    def test_different_types_of_input(self, sort):
        # There are 7 types of accepted input by append:
        #
        # dict
        # Series
        # DataFrame
        # empty list
        # list of dicts
        # list of Series
        # list of DataFrames
        #
        # Using one or another should always be interchangeable.

        # append to dict
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        map = {
            0: 7,
            1: 8,
            2: 9
        }
        result = df.append(map, ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to Series
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        result = df.append(ser, ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to DataFrame
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df2 = pd.DataFrame([[7, 8, 9]])
        result = df1.append(df2, ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to empty list
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        result = df1.append([], sort=sort)
        expected = df
        assert_frame_equal(result, expected)

        # append to list of dicts
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        map = {
            0: 7,
            1: 8,
            2: 9
        }
        result = df.append([map], ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to list of Series
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        result = df.append([ser], ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to list of DataFrames
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df2 = pd.DataFrame([[7, 8, 9]])
        result = df1.append([df2], ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to list of dicts (2 dicts)
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        map = {
            0: 7,
            1: 8,
            2: 9
        }
        result = df.append([map, map], ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to list of Series (2 series)
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        result = df.append([ser, ser], ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        assert_frame_equal(result, expected)

        # append to list of DataFrames (2 dframes)
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df2 = pd.DataFrame([[7, 8, 9]])
        result = df1.append([df2, df2], ignore_index=True, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        assert_frame_equal(result, expected)

    def test_bad_input_type(self, sort):
        # When appending a bad input type, the function
        # should raise an exception.

        bad_input_msg = r'The value of other must be .*'
        mixed_list_msg = r'When other is a list, its .*'

        # integer input
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append(1, ignore_index=True, sort=sort)

        # string input
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append("1 2 3", ignore_index=True, sort=sort)

        # tuple input
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append((df, ), ignore_index=True, sort=sort)

        # list of integers
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append([1], ignore_index=True, sort=sort)

        # list of strings
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append(["1 2 3"], ignore_index=True, sort=sort)

        # list of lists
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append([[df]], ignore_index=True, sort=sort)

        # list of tuples
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append([(df, )], ignore_index=True, sort=sort)

        # mixed list
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        dict = {
            0: 10,
            1: 11,
            2: 12
        }
        with pytest.raises(TypeError, match=mixed_list_msg):
            df.append([ser, dict], ignore_index=True, sort=sort)
        with pytest.raises(TypeError, match=mixed_list_msg):
            df.append([dict, ser], ignore_index=True, sort=sort)

        # mixed list with bad first element
        # (when the first element is bad, display the
        #  bad input msg instead of the mixed list one)
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        with pytest.raises(TypeError, match=bad_input_msg):
            df.append([1, ser, ser], ignore_index=True, sort=sort)

        # mixed list with bad second element
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        with pytest.raises(TypeError, match=mixed_list_msg):
            df.append([ser, 1, ser], ignore_index=True, sort=sort)

        # mixed list with bad third element
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        ser = pd.Series([7, 8, 9])
        with pytest.raises(TypeError, match=mixed_list_msg):
            df.append([ser, ser, 1], ignore_index=True, sort=sort)

    def test_no_unecessary_upcast(self, sort):
        # GH: 22621
        # When appending, the result columns should
        # not be float64 without necessity.

        # basic
        df1 = pd.DataFrame([[1, 2, 3]])
        df2 = pd.DataFrame([[4, 5, 6]], index=[1])
        result = df1.append(df2, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        assert_frame_equal(result, expected)

        # 0 rows 0 columns
        df1 = pd.DataFrame([[1, 2, 3]])
        df2 = pd.DataFrame()
        result = df1.append(df2, sort=sort)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        df1 = pd.DataFrame()
        df2 = pd.DataFrame([[1, 2, 3]])
        result = df1.append(df2, sort=sort)
        expected = df2.copy()
        assert_frame_equal(result, expected)

        # 0 rows 2 columns
        df1 = pd.DataFrame([[1, 2, 3]], columns=[0, 1, 2])
        df2 = pd.DataFrame(columns=[3, 4])
        result = df1.append(df2, sort=sort)
        expected = pd.DataFrame([[1, 2, 3, np.nan, np.nan]])
        assert_frame_equal(result, expected)

        df1 = pd.DataFrame(columns=[0, 1])
        df2 = pd.DataFrame([[1, 2, 3]], columns=[2, 3, 4])
        result = df1.append(df2, sort=sort)
        expected = pd.DataFrame([[np.nan, np.nan, 1, 2, 3]])
        assert_frame_equal(result, expected)

        # big.append(small)
        big = pd.DataFrame([[1, 2, 3]])
        small = pd.DataFrame([[4, 5]], index=[1])
        result = big.append(small, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, np.nan]])
        assert_frame_equal(result, expected)

        # small.append(big)
        small = pd.DataFrame([[1, 2]])
        big = pd.DataFrame([[3, 4, 5]], index=[1])
        result = small.append(big, sort=sort)
        expected = pd.DataFrame([[1, 2, np.nan], [3, 4, 5]])
        assert_frame_equal(result, expected)


class TestAppendColumnsIndex(object):
    @pytest.mark.parametrize('idx_name3', [None, 'foo', 'bar', 'baz'])
    @pytest.mark.parametrize('idx_name2', [None, 'foo', 'bar', 'baz'])
    @pytest.mark.parametrize('idx_name1', [None, 'foo', 'bar', 'baz'])
    def test_preserve_index_name(self, sort, idx_name1, idx_name2, idx_name3):
        # When appending, the name of the indexes
        # of the base DataFrame must always be
        # preserved in the result.

        df1 = pd.DataFrame([[1, 2, 3]])
        df2 = pd.DataFrame([[4, 5, 6]], index=[1])
        df3 = pd.DataFrame([[7, 8, 9]], index=[2])

        df1.columns.name = idx_name1
        df2.columns.name = idx_name2
        df3.columns.name = idx_name3

        # append []
        result = df1.append([], sort=sort)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        # append [df]
        result = df1.append([df2], sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        expected.columns.name = idx_name1
        assert_frame_equal(result, expected)

        # append [df, df]
        result = df1.append([df2, df3], sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected.columns.name = idx_name1
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index', indexes, ids=cls_name)
    def test_preserve_index_type(self, sort, index):
        # when there's only one index type in the inputs,
        # it must be preserved in the output.

        # basic
        df1 = pd.DataFrame([[1, 2, 3]], columns=index)
        df2 = pd.DataFrame([[4, 5, 6]], index=[1], columns=index)
        result = df1.append(df2, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=index)
        assert_frame_equal(result, expected)

        # big.append(small)
        big = pd.DataFrame([[1, 2, 3]], columns=index)
        small = pd.DataFrame([[4, 5]], index=[1], columns=index[:2])
        result = big.append(small, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, np.nan]], columns=index)
        assert_frame_equal(result, expected)

        # small.append(big)
        small = pd.DataFrame([[1, 2]], columns=index[:2])
        big = pd.DataFrame([[3, 4, 5]], index=[1], columns=index)
        result = small.append(big, sort=sort)
        expected = pd.DataFrame([[1, 2, np.nan], [3, 4, 5]], columns=index)
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index2', indexes, ids=cls_name)
    @pytest.mark.parametrize('index1', indexes, ids=cls_name)
    def test_preserve_index_values_without_sort(self, index1, index2):
        # When appending indexes of different types, we want
        # the resulting index to preserve the exact indexes
        # values.

        # Related to GH13626
        from pandas.core.dtypes.generic import (
            ABCDatetimeIndex, ABCMultiIndex, ABCTimedeltaIndex
        )
        if isinstance(index1, ABCMultiIndex):
            if isinstance(index2, ABCDatetimeIndex):
                pytest.xfail("MultiIndex + DatetimeIndex produces bad value")
            if isinstance(index2, ABCTimedeltaIndex):
                pytest.xfail("MultiIndex + TimedeltaIndex produces bad value")

        df1 = pd.DataFrame([[1, 2, 3]], columns=index1)
        df2 = pd.DataFrame([[4, 5, 6]], columns=index2, index=[1])
        result = df1.append(df2, sort=False)
        for value in index1:
            assert value in result.columns
        for value in index2:
            assert value in result.columns

    @pytest.mark.parametrize(
        'index1, index2',
        [(i1, i2)
            for group in index_sort_groups
            for i1, i2 in product(group, repeat=2)],
        ids=cls_name
    )
    def test_preserve_index_values_with_sort(self, index1, index2):
        # When appending indexes of different types, we want
        # the resulting index to preserve the exact indexes
        # values.

        df1 = pd.DataFrame([[1, 2, 3]], columns=index1)
        df2 = pd.DataFrame([[4, 5, 6]], columns=index2, index=[1])
        result = df1.append(df2, sort=True)
        for value in index1:
            assert value in result.columns
        for value in index2:
            assert value in result.columns

    @pytest.mark.parametrize('col_index', indexes_with_dups, ids=cls_name)
    def test_good_duplicates_without_sort(self, col_index):
        # When all indexes have the same identity (a is b), duplicates should
        # be allowed and append works.

        df1 = pd.DataFrame([[1, 2, 3]], columns=col_index)
        df2 = pd.DataFrame([[4, 5, 6]], columns=col_index)

        # df1.append([])
        result = df1.append([], sort=False)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        # df1.append([df2])
        result = df1.append([df2], ignore_index=True, sort=False)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        expected.columns = col_index
        assert_frame_equal(result, expected)

        # df1.append([df2, df2])
        result = df1.append([df2, df2], ignore_index=True, sort=False)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
        expected.columns = col_index
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('col_index', indexes_with_dups, ids=cls_name)
    def test_bad_duplicates_without_sort(self, col_index):
        # When the indexes do not share a common identity, duplicates are not
        # allowed and append raises.

        df1 = pd.DataFrame([[1, 2, 3]], columns=col_index)
        df2 = pd.DataFrame([[4, 5, 6]], columns=col_index)
        df3 = pd.DataFrame([[7, 8, 9]], columns=col_index.copy())  # different
        ctx = pytest.raises(InvalidIndexError,
                            match=r'Indexes with duplicates.*a is b.*')
        with ctx:
            result = df1.append([df3], sort=False)
        with ctx:
            result = df1.append([df2, df3], sort=False)
        with ctx:
            result = df1.append([df3, df2], sort=False)
        with ctx:
            result = df1.append([df3, df3], sort=False)

    @pytest.mark.parametrize('col_index', indexes_with_dups, ids=cls_name)
    def test_duplicates_with_sort(self, col_index):
        # When sort=True, indexes with duplicate values are not be allowed.

        df1 = pd.DataFrame([[1, 2, 3]], columns=col_index)
        df2 = pd.DataFrame([[4, 5, 6]], columns=col_index.copy())
        ctx = pytest.raises(InvalidIndexError,
                            match=r'When sort=True, indexes with dupl.*')

        with ctx:
            result = df1.append([], sort=True)
        with ctx:
            result = df1.append([df1], sort=True)
        with ctx:
            result = df1.append([df2], sort=True)
        with ctx:
            result = df1.append([df1, df1], sort=True)
        with ctx:
            result = df1.append([df1, df2], sort=True)
        with ctx:
            result = df1.append([df2, df1], sort=True)
        with ctx:
            result = df1.append([df2, df2], sort=True)

    def test_nosort_basic(self):
        # When sort=False, the resulting columns come
        # in the order that they appear in the inputs.

        nan = np.nan

        # NUMERIC INDEX TESTS

        # append []
        df = pd.DataFrame([[1, 2, 3]], columns=[0, 1, 2])
        result = df.append([], sort=False)
        expected = df[[0, 1, 2]]
        assert_frame_equal(result, expected)

        df = pd.DataFrame([[1, 2, 3]], columns=[2, 1, 0])
        result = df.append([], sort=False)
        expected = df[[2, 1, 0]]
        assert_frame_equal(result, expected)

        # append [df]
        df1 = pd.DataFrame([[1, 2]], columns=[0.0, 1.0])
        df2 = pd.DataFrame([[1, 2]], columns=[0.5, 1.5], index=[1])
        result = df1.append(df2, sort=False)
        expected = pd.DataFrame([[1, 2, nan, nan],
                                 [nan, nan, 1, 2]],
                                columns=[0.0, 1.0, 0.5, 1.5])
        assert_frame_equal(result, expected)

        # append [df, df]
        df1 = pd.DataFrame([[1, 2]], columns=[0.0, 1.0])
        df2 = pd.DataFrame([[1, 2]], columns=[0.3, 1.3], index=[1])
        df3 = pd.DataFrame([[1, 2]], columns=[0.6, 1.6], index=[2])
        result = df1.append([df2, df3], sort=False)
        expected = pd.DataFrame([[1, 2, nan, nan, nan, nan],
                                 [nan, nan, 1, 2, nan, nan],
                                 [nan, nan, nan, nan, 1, 2]],
                                columns=[0.0, 1.0, 0.3, 1.3, 0.6, 1.6])
        assert_frame_equal(result, expected)

        # STRING INDEX TESTS

        # append []
        df = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
        result = df.append([], sort=False)
        expected = df[['a', 'b', 'c']]
        assert_frame_equal(result, expected)

        df = pd.DataFrame([[1, 2, 3]], columns=['c', 'b', 'a'])
        result = df.append([], sort=False)
        expected = df[['c', 'b', 'a']]
        assert_frame_equal(result, expected)

        # append [df]
        df1 = pd.DataFrame([[1, 2]], columns=['a', 'c'])
        df2 = pd.DataFrame([[1, 2]], columns=['b', 'd'], index=[1])
        result = df1.append(df2, sort=False)
        expected = pd.DataFrame([[1, 2, nan, nan],
                                 [nan, nan, 1, 2]],
                                columns=['a', 'c', 'b', 'd'])
        assert_frame_equal(result, expected)

        # append [df, df]
        df1 = pd.DataFrame([[1, 2]], columns=['a', 'd'])
        df2 = pd.DataFrame([[1, 2]], columns=['b', 'e'], index=[1])
        df3 = pd.DataFrame([[1, 2]], columns=['c', 'f'], index=[2])
        result = df1.append([df2, df3], sort=False)
        expected = pd.DataFrame([[1, 2, nan, nan, nan, nan],
                                 [nan, nan, 1, 2, nan, nan],
                                 [nan, nan, nan, nan, 1, 2]],
                                columns=['a', 'd', 'b', 'e', 'c', 'f'])
        assert_frame_equal(result, expected)

    def test_sort_basic(self):
        # When sort=True, the resulting columns must come
        # out sorted.

        nan = np.nan

        # NUMERIC INDEX TESTS

        # append []
        df = pd.DataFrame([[1, 2, 3]], columns=[0, 1, 2])
        result = df.append([], sort=True)
        expected = df[[0, 1, 2]]
        assert_frame_equal(result, expected)

        df = pd.DataFrame([[1, 2, 3]], columns=[2, 1, 0])
        result = df.append([], sort=True)
        expected = df[[0, 1, 2]]
        assert_frame_equal(result, expected)

        # append [df]
        df1 = pd.DataFrame([[1, 2]], columns=[0.0, 1.0])
        df2 = pd.DataFrame([[1, 2]], columns=[0.5, 1.5], index=[1])
        result = df1.append(df2, sort=True)
        expected = pd.DataFrame([[1, nan, 2, nan],
                                 [nan, 1, nan, 2]],
                                columns=[0.0, 0.5, 1.0, 1.5])
        assert_frame_equal(result, expected)

        # append [df, df]
        df1 = pd.DataFrame([[1, 2]], columns=[0.0, 1.0])
        df2 = pd.DataFrame([[1, 2]], columns=[0.3, 1.3], index=[1])
        df3 = pd.DataFrame([[1, 2]], columns=[0.6, 1.6], index=[2])
        result = df1.append([df2, df3], sort=True)
        expected = pd.DataFrame([[1, nan, nan, 2, nan, nan],
                                 [nan, 1, nan, nan, 2, nan],
                                 [nan, nan, 1, nan, nan, 2]],
                                columns=[0.0, 0.3, 0.6, 1.0, 1.3, 1.6])
        assert_frame_equal(result, expected)

        # STRING INDEX TESTS

        # append []
        df = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
        result = df.append([], sort=True)
        expected = df[['a', 'b', 'c']]
        assert_frame_equal(result, expected)

        df = pd.DataFrame([[1, 2, 3]], columns=['c', 'b', 'a'])
        result = df.append([], sort=True)
        expected = df[['a', 'b', 'c']]
        assert_frame_equal(result, expected)

        # append [df]
        df1 = pd.DataFrame([[1, 2]], columns=['a', 'c'])
        df2 = pd.DataFrame([[1, 2]], columns=['b', 'd'], index=[1])
        result = df1.append(df2, sort=True)
        expected = pd.DataFrame([[1, nan, 2, nan],
                                 [nan, 1, nan, 2]],
                                columns=['a', 'b', 'c', 'd'])
        assert_frame_equal(result, expected)

        # append [df, df]
        df1 = pd.DataFrame([[1, 2]], columns=['a', 'd'])
        df2 = pd.DataFrame([[1, 2]], columns=['b', 'e'], index=[1])
        df3 = pd.DataFrame([[1, 2]], columns=['c', 'f'], index=[2])
        result = df1.append([df2, df3], sort=True)
        expected = pd.DataFrame([[1, nan, nan, 2, nan, nan],
                                 [nan, 1, nan, nan, 2, nan],
                                 [nan, nan, 1, nan, nan, 2]],
                                columns=['a', 'b', 'c', 'd', 'e', 'f'])
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index2', indexes, ids=cls_name)
    @pytest.mark.parametrize('index1', indexes, ids=cls_name)
    def test_index_types_without_sort(self, index1, index2):
        # We should be able to append to a DataFrame
        # regardless of the type of its index.

        # TODO: check end of append and create tests (empty / IntervalIndex)
        # TODO: implement different way for df.append([])
        from pandas.core.dtypes.generic import ABCIntervalIndex
        if isinstance(index1, ABCIntervalIndex):
            pytest.xfail("Cannot do df[interval] for IntervalIndex")

        # the code below should not raise any exceptions
        df1 = pd.DataFrame([[1, 2, 3]], columns=index1)
        df2 = pd.DataFrame([[4, 5, 6]], columns=index2, index=[1])
        df1.append([], sort=False)
        df1.append([df2], sort=False)
        df1.append([df2, df2], sort=False)

    @pytest.mark.parametrize(
        'index1, index2',
        [(i1, i2)
            for group in index_sort_groups
            for i1, i2 in product(group, repeat=2)],
        ids=cls_name
    )
    def test_index_types_with_possible_sort(self, index1, index2):
        # When the result of joining two indexes is sortable,
        # we should not raise any exceptions.

        # TODO: check end of append and create tests (empty / IntervalIndex)
        # TODO: implement different way for df.append([])
        from pandas.core.dtypes.generic import ABCIntervalIndex
        if isinstance(index1, ABCIntervalIndex):
            pytest.xfail("Cannot do df[interval] for IntervalIndex")

        df1 = pd.DataFrame([[1, 2, 3]], columns=index1)
        df2 = pd.DataFrame([[4, 5, 6]], columns=index2, index=[1])
        df1.append([], sort=True)  # sorts the original frame
        df1.append([df2], sort=True)
        df1.append([df2, df2], sort=True)

    @pytest.mark.parametrize(
        'index1, index2',
        [(i1, i2)
            for g1, g2 in product(index_sort_groups, repeat=2)
                # different sort groups
                if type(g1[0]) != type(g2[0])
            for i1, i2 in product(g1, g2)],
        ids=cls_name
    )
    def test_index_types_with_impossible_sort(self, index1, index2):
        # When the result of joining two indexes is not sortable,
        # we should raise an exception.

        # TODO: check end of append and create tests (empty / IntervalIndex)
        # TODO: implement different way for df.append([])
        from pandas.core.dtypes.generic import ABCIntervalIndex
        if isinstance(index1, ABCIntervalIndex):
            pytest.xfail("Cannot do df[interval] for IntervalIndex")

        err_msg = r'The resulting columns could not be sorted\..*'

        df1 = pd.DataFrame([[1, 2, 3]], columns=index1)
        df2 = pd.DataFrame([[4, 5, 6]], columns=index2, index=[1])

        with pytest.raises(TypeError, match=err_msg):
            df1.append([df2], sort=True)
        with pytest.raises(TypeError, match=err_msg):
            df1.append([df2, df2], sort=True)


class TestAppendRowsIndex(object):
    @pytest.mark.parametrize('idx_name3', [None, 'foo', 'bar', 'baz'])
    @pytest.mark.parametrize('idx_name2', [None, 'foo', 'bar', 'baz'])
    @pytest.mark.parametrize('idx_name1', [None, 'foo', 'bar', 'baz'])
    def test_preserve_index_name(self, sort, idx_name1, idx_name2, idx_name3):
        # When appending, the name of the indexes
        # of the base DataFrame must always be
        # preserved in the result.

        df1 = pd.DataFrame([[1, 2, 3]])
        df2 = pd.DataFrame([[4, 5, 6]], index=[1])
        df3 = pd.DataFrame([[7, 8, 9]], index=[2])

        df1.index.name = idx_name1
        df2.index.name = idx_name2
        df3.index.name = idx_name3

        # append []
        result = df1.append([], sort=sort)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        # append [df]
        result = df1.append([df2], sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        expected.index.name = idx_name1
        assert_frame_equal(result, expected)

        # append [df, df]
        result = df1.append([df2, df3], sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected.index.name = idx_name1
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index', indexes, ids=cls_name)
    def test_preserve_index_type(self, sort, index):
        # when there's only one index type in the inputs,
        # it must be preserved in the output.

        index1 = index[:1]
        index2 = index[1:2]
        index_comb = index1.append(index2)

        df1 = pd.DataFrame([[1, 2, 3]], index=index1)
        df2 = pd.DataFrame([[4, 5, 6]], index=index2)
        result = df1.append(df2, sort=sort)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=index_comb)
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index2', indexes, ids=cls_name)
    @pytest.mark.parametrize('index1', indexes, ids=cls_name)
    def test_preserve_index_values(self, sort, index1, index2):
        # When appending indexes of different types, we want
        # the resulting index to preserve the exact indexes
        # values.

        # Related to GH13626
        from pandas.core.dtypes.generic import (
            ABCDatetimeIndex, ABCMultiIndex, ABCTimedeltaIndex
        )
        if isinstance(index1, ABCMultiIndex):
            if isinstance(index2, ABCDatetimeIndex):
                pytest.xfail("MultiIndex + DatetimeIndex produces bad value")
            if isinstance(index2, ABCTimedeltaIndex):
                pytest.xfail("MultiIndex + TimedeltaIndex produces bad value")

        # Concat raises a TypeError when appending a CategoricalIndex
        # with another type
        from pandas.core.dtypes.generic import ABCCategoricalIndex
        if isinstance(index1, ABCCategoricalIndex):
            pytest.xfail("Cannot have a CategoricalIndex append to another typ")

        df1 = pd.DataFrame([[1, 2, 3]], index=index1[:1])
        df2 = pd.DataFrame([[4, 5, 6]], index=index2[:1])
        result = df1.append(df2, sort=sort)
        assert index1[0] in result.index
        assert index2[0] in result.index

    def test_duplicates_without_verify_integrity(self):
        # When verify_integrity=False, the function should
        # allow duplicate values in the rows index.

        raise NotImplementedError

    def test_duplicates_with_verify_integrity(self):
        # When verify_integrity=True, the function should
        # not allow duplicate values in the rows index (whether
        # in the input or output).

        raise NotImplementedError

    def test_ignore_index(self):
        # When ignore_index=True, the function should completely
        # ignore the input indexes and generate one that is brand
        # new (RangeIndex).

        raise NotImplementedError

    def test_warning_ignore_index_and_verify_integrity(self):
        # It makes no sense to set verify_integrity=True when
        # ignore_index=True. To warn of a possible user
        # misunderstanding, append should raise a warning in
        # this situation.

        raise NotImplementedError


class TestAppendBefore(object):
    """Tests that were written before the append refactor
    """
    # tests below came from pandas/tests/reshape/test_concat.py

    def setup_method(self, method):
        self.frame = DataFrame(tm.getSeriesData())
        self.mixed_frame = self.frame.copy()
        self.mixed_frame['foo'] = 'bar'

    def test_append(self, sort):
        begin_index = self.frame.index[:5]
        end_index = self.frame.index[5:]

        begin_frame = self.frame.reindex(begin_index)
        end_frame = self.frame.reindex(end_index)

        appended = begin_frame.append(end_frame)
        tm.assert_almost_equal(appended['A'], self.frame['A'])

        del end_frame['A']
        partial_appended = begin_frame.append(end_frame, sort=sort)
        assert 'A' in partial_appended

        partial_appended = end_frame.append(begin_frame, sort=sort)
        assert 'A' in partial_appended

        # mixed type handling
        appended = self.mixed_frame[:5].append(self.mixed_frame[5:])
        tm.assert_frame_equal(appended, self.mixed_frame)

        # what to test here
        mixed_appended = self.mixed_frame[:5].append(self.frame[5:], sort=sort)
        mixed_appended2 = self.frame[:5].append(self.mixed_frame[5:],
                                                sort=sort)

        # all equal except 'foo' column
        tm.assert_frame_equal(
            mixed_appended.reindex(columns=['A', 'B', 'C', 'D']),
            mixed_appended2.reindex(columns=['A', 'B', 'C', 'D']))

        # append empty
        empty = DataFrame({})

        appended = self.frame.append(empty)
        tm.assert_frame_equal(self.frame, appended)
        assert appended is not self.frame

        appended = empty.append(self.frame)
        tm.assert_frame_equal(self.frame, appended)
        assert appended is not self.frame

        # Overlap
        with pytest.raises(ValueError):
            self.frame.append(self.frame, verify_integrity=True)

        # see gh-6129: new columns
        df = DataFrame({'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}})
        row = Series([5, 6, 7], index=['a', 'b', 'c'], name='z')
        expected = DataFrame({'a': {'x': 1, 'y': 2, 'z': 5}, 'b': {
                             'x': 3, 'y': 4, 'z': 6}, 'c': {'z': 7}})
        result = df.append(row)
        tm.assert_frame_equal(result, expected)

    def test_append_length0_frame(self, sort):
        df = DataFrame(columns=['A', 'B', 'C'])
        df3 = DataFrame(index=[0, 1], columns=['A', 'B'])
        df5 = df.append(df3, sort=sort)

        expected = DataFrame(index=[0, 1], columns=['A', 'B', 'C'])
        assert_frame_equal(df5, expected)

    def test_append_records(self):
        arr1 = np.zeros((2,), dtype=('i4,f4,a10'))
        arr1[:] = [(1, 2., 'Hello'), (2, 3., "World")]

        arr2 = np.zeros((3,), dtype=('i4,f4,a10'))
        arr2[:] = [(3, 4., 'foo'),
                   (5, 6., "bar"),
                   (7., 8., 'baz')]

        df1 = DataFrame(arr1)
        df2 = DataFrame(arr2)

        result = df1.append(df2, ignore_index=True)
        expected = DataFrame(np.concatenate((arr1, arr2)))
        assert_frame_equal(result, expected)

    # rewrite sort fixture, since we also want to test default of None
    def test_append_sorts(self, sort_with_none):
        df1 = pd.DataFrame({"a": [1, 2], "b": [1, 2]}, columns=['b', 'a'])
        df2 = pd.DataFrame({"a": [1, 2], 'c': [3, 4]}, index=[2, 3])

        if sort_with_none is None:
            # only warn if not explicitly specified
            # don't check stacklevel since its set for concat, and append
            # has an extra stack.
            ctx = tm.assert_produces_warning(FutureWarning,
                                             check_stacklevel=False)
        else:
            ctx = tm.assert_produces_warning(None)

        with ctx:
            result = df1.append(df2, sort=sort_with_none)

        # for None / True
        expected = pd.DataFrame({"b": [1, 2, None, None],
                                 "a": [1, 2, 1, 2],
                                 "c": [None, None, 3, 4]},
                                columns=['a', 'b', 'c'])
        if sort_with_none is False:
            expected = expected[['b', 'a', 'c']]
        tm.assert_frame_equal(result, expected)

    def test_append_different_columns(self, sort):
        df = DataFrame({'bools': np.random.randn(10) > 0,
                        'ints': np.random.randint(0, 10, 10),
                        'floats': np.random.randn(10),
                        'strings': ['foo', 'bar'] * 5})

        a = df[:5].loc[:, ['bools', 'ints', 'floats']]
        b = df[5:].loc[:, ['strings', 'ints', 'floats']]

        appended = a.append(b, sort=sort)
        assert isna(appended['strings'][0:4]).all()
        assert isna(appended['bools'][5:]).all()

    def test_append_many(self, sort):
        chunks = [self.frame[:5], self.frame[5:10],
                  self.frame[10:15], self.frame[15:]]

        result = chunks[0].append(chunks[1:])
        tm.assert_frame_equal(result, self.frame)

        chunks[-1] = chunks[-1].copy()
        chunks[-1]['foo'] = 'bar'
        result = chunks[0].append(chunks[1:], sort=sort)
        tm.assert_frame_equal(result.loc[:, self.frame.columns], self.frame)
        assert (result['foo'][15:] == 'bar').all()
        assert result['foo'][:15].isna().all()

    def test_append_preserve_index_name(self):
        # #980
        df1 = DataFrame(data=None, columns=['A', 'B', 'C'])
        df1 = df1.set_index(['A'])
        df2 = DataFrame(data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]],
                        columns=['A', 'B', 'C'])
        df2 = df2.set_index(['A'])

        result = df1.append(df2)
        assert result.index.name == 'A'

    indexes_can_append = [
        pd.RangeIndex(3),
        pd.Index([4, 5, 6]),
        pd.Index([4.5, 5.5, 6.5]),
        pd.Index(list('abc')),
        pd.CategoricalIndex('A B C'.split()),
        pd.CategoricalIndex('D E F'.split(), ordered=True),
        pd.DatetimeIndex([dt.datetime(2013, 1, 3, 0, 0),
                          dt.datetime(2013, 1, 3, 6, 10),
                          dt.datetime(2013, 1, 3, 7, 12)]),
    ]

    indexes_cannot_append_with_other = [
        pd.IntervalIndex.from_breaks([0, 1, 2, 3]),
        pd.MultiIndex.from_arrays(['A B C'.split(), 'D E F'.split()]),
    ]

    all_indexes = indexes_can_append + indexes_cannot_append_with_other

    @pytest.mark.parametrize("index",
                             all_indexes,
                             ids=lambda x: x.__class__.__name__)
    def test_append_same_columns_type(self, index):
        # GH18359

        # df wider than ser
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=index)
        ser_index = index[:2]
        ser = pd.Series([7, 8], index=ser_index, name=2)
        result = df.append(ser)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]],
                                index=[0, 1, 2],
                                columns=index)
        assert_frame_equal(result, expected)

        # ser wider than df
        ser_index = index
        index = index[:2]
        df = pd.DataFrame([[1, 2], [4, 5]], columns=index)
        ser = pd.Series([7, 8, 9], index=ser_index, name=2)
        result = df.append(ser)
        expected = pd.DataFrame([[1, 2, np.nan], [4, 5, np.nan], [7, 8, 9]],
                                index=[0, 1, 2],
                                columns=ser_index)
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("df_columns, series_index",
                             combinations(indexes_can_append, r=2),
                             ids=lambda x: x.__class__.__name__)
    def test_append_different_columns_types(self, df_columns, series_index):
        # GH18359
        # See also test 'test_append_different_columns_types_raises' below
        # for errors raised when appending

        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=df_columns)
        ser = pd.Series([7, 8, 9], index=series_index, name=2)

        result = df.append(ser)
        idx_diff = ser.index.difference(df_columns)
        combined_columns = Index(df_columns.tolist()).append(idx_diff)
        expected = pd.DataFrame([[1., 2., 3., np.nan, np.nan, np.nan],
                                 [4, 5, 6, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, 7, 8, 9]],
                                index=[0, 1, 2],
                                columns=combined_columns)
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index_can_append', indexes_can_append,
                             ids=lambda x: x.__class__.__name__)
    @pytest.mark.parametrize('index_cannot_append_with_other',
                             indexes_cannot_append_with_other,
                             ids=lambda x: x.__class__.__name__)
    def test_append_different_columns_types_raises(
            self, index_can_append, index_cannot_append_with_other):
        # GH18359
        # Dataframe.append will raise if IntervalIndex/MultiIndex appends
        # or is appended to a different index type
        #
        # See also test 'test_append_different_columns_types' above for
        # appending without raising.

        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=index_can_append)
        ser = pd.Series([7, 8, 9], index=index_cannot_append_with_other,
                        name=2)
        with pytest.raises((AttributeError, ValueError, TypeError)):
            df.append(ser)

        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]],
                          columns=index_cannot_append_with_other)
        ser = pd.Series([7, 8, 9], index=index_can_append, name=2)
        with pytest.raises((AttributeError, ValueError, TypeError)):
            df.append(ser)

    def test_append_dtype_coerce(self, sort):

        # GH 4993
        # appending with datetime will incorrectly convert datetime64

        df1 = DataFrame(index=[1, 2], data=[dt.datetime(2013, 1, 1, 0, 0),
                                            dt.datetime(2013, 1, 2, 0, 0)],
                        columns=['start_time'])
        df2 = DataFrame(index=[4, 5], data=[[dt.datetime(2013, 1, 3, 0, 0),
                                             dt.datetime(2013, 1, 3, 6, 10)],
                                            [dt.datetime(2013, 1, 4, 0, 0),
                                             dt.datetime(2013, 1, 4, 7, 10)]],
                        columns=['start_time', 'end_time'])

        expected = concat([Series([pd.NaT,
                                   pd.NaT,
                                   dt.datetime(2013, 1, 3, 6, 10),
                                   dt.datetime(2013, 1, 4, 7, 10)],
                                  name='end_time'),
                           Series([dt.datetime(2013, 1, 1, 0, 0),
                                   dt.datetime(2013, 1, 2, 0, 0),
                                   dt.datetime(2013, 1, 3, 0, 0),
                                   dt.datetime(2013, 1, 4, 0, 0)],
                                  name='start_time')],
                          axis=1, sort=sort)
        result = df1.append(df2, ignore_index=True, sort=sort)
        if sort:
            expected = expected[['end_time', 'start_time']]
        else:
            expected = expected[['start_time', 'end_time']]

        assert_frame_equal(result, expected)

    def test_append_missing_column_proper_upcast(self, sort):
        df1 = DataFrame({'A': np.array([1, 2, 3, 4], dtype='i8')})
        df2 = DataFrame({'B': np.array([True, False, True, False],
                                       dtype=bool)})

        appended = df1.append(df2, ignore_index=True, sort=sort)
        assert appended['A'].dtype == 'f8'
        assert appended['B'].dtype == 'O'

    # tests below came from pandas/tests/frame/test_combine_concat.py

    def test_append_series_dict(self):
        df = DataFrame(np.random.randn(5, 4),
                       columns=['foo', 'bar', 'baz', 'qux'])

        series = df.loc[4]
        with tm.assert_raises_regex(ValueError,
                                    'Indexes have overlapping values'):
            df.append(series, verify_integrity=True)
        series.name = None
        with tm.assert_raises_regex(TypeError,
                                    'Can only append a Series if '
                                    'ignore_index=True'):
            df.append(series, verify_integrity=True)

        result = df.append(series[::-1], ignore_index=True)
        expected = df.append(DataFrame({0: series[::-1]}, index=df.columns).T,
                             ignore_index=True)
        assert_frame_equal(result, expected)

        # dict
        result = df.append(series.to_dict(), ignore_index=True)
        assert_frame_equal(result, expected)

        result = df.append(series[::-1][:3], ignore_index=True)
        expected = df.append(DataFrame({0: series[::-1][:3]}).T,
                             ignore_index=True, sort=True)
        assert_frame_equal(result, expected.loc[:, result.columns])

        # can append when name set
        row = df.loc[4]
        row.name = 5
        result = df.append(row)
        expected = df.append(df[-1:], ignore_index=True)
        assert_frame_equal(result, expected)

    def test_append_list_of_series_dicts(self):
        df = DataFrame(np.random.randn(5, 4),
                       columns=['foo', 'bar', 'baz', 'qux'])

        dicts = [x.to_dict() for idx, x in df.iterrows()]

        result = df.append(dicts, ignore_index=True)
        expected = df.append(df, ignore_index=True)
        assert_frame_equal(result, expected)

        # different columns
        dicts = [{'foo': 1, 'bar': 2, 'baz': 3, 'peekaboo': 4},
                 {'foo': 5, 'bar': 6, 'baz': 7, 'peekaboo': 8}]
        result = df.append(dicts, ignore_index=True, sort=True)
        expected = df.append(DataFrame(dicts), ignore_index=True, sort=True)
        assert_frame_equal(result, expected)

    def test_append_empty_dataframe(self):

        # Empty df append empty df
        df1 = DataFrame([])
        df2 = DataFrame([])
        result = df1.append(df2)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        # Non-empty df append empty df
        df1 = DataFrame(np.random.randn(5, 2))
        df2 = DataFrame()
        result = df1.append(df2)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        # Empty df with columns append empty df
        df1 = DataFrame(columns=['bar', 'foo'])
        df2 = DataFrame()
        result = df1.append(df2)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        # Non-Empty df with columns append empty df
        df1 = DataFrame(np.random.randn(5, 2), columns=['bar', 'foo'])
        df2 = DataFrame()
        result = df1.append(df2)
        expected = df1.copy()
        assert_frame_equal(result, expected)

    def test_append_dtypes(self):

        # GH 5754
        # row appends of different dtypes (so need to do by-item)
        # can sometimes infer the correct type

        df1 = DataFrame({'bar': Timestamp('20130101')}, index=lrange(5))
        df2 = DataFrame()
        result = df1.append(df2)
        expected = df1.copy()
        assert_frame_equal(result, expected)

        df1 = DataFrame({'bar': Timestamp('20130101')}, index=lrange(1))
        df2 = DataFrame({'bar': 'foo'}, index=lrange(1, 2))
        result = df1.append(df2)
        expected = DataFrame({'bar': [Timestamp('20130101'), 'foo']})
        assert_frame_equal(result, expected)

        df1 = DataFrame({'bar': Timestamp('20130101')}, index=lrange(1))
        df2 = DataFrame({'bar': np.nan}, index=lrange(1, 2))
        result = df1.append(df2)
        expected = DataFrame(
            {'bar': Series([Timestamp('20130101'), np.nan], dtype='M8[ns]')})
        assert_frame_equal(result, expected)

        df1 = DataFrame({'bar': Timestamp('20130101')}, index=lrange(1))
        df2 = DataFrame({'bar': np.nan}, index=lrange(1, 2), dtype=object)
        result = df1.append(df2)
        expected = DataFrame(
            {'bar': Series([Timestamp('20130101'), np.nan], dtype='M8[ns]')})
        assert_frame_equal(result, expected)

        df1 = DataFrame({'bar': np.nan}, index=lrange(1))
        df2 = DataFrame({'bar': Timestamp('20130101')}, index=lrange(1, 2))
        result = df1.append(df2)
        expected = DataFrame(
            {'bar': Series([np.nan, Timestamp('20130101')], dtype='M8[ns]')})
        assert_frame_equal(result, expected)

        df1 = DataFrame({'bar': Timestamp('20130101')}, index=lrange(1))
        df2 = DataFrame({'bar': 1}, index=lrange(1, 2), dtype=object)
        result = df1.append(df2)
        expected = DataFrame({'bar': Series([Timestamp('20130101'), 1])})
        assert_frame_equal(result, expected)


class TestAppendDangling(object):
    """Tests that have not been concretized yet
    """

    def test_append_unnamed_series_raises(self, sort):
        dict_msg = 'Can only append a dict if ignore_index=True'
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        dict = {
            0: 7,
            1: 8,
            2: 9
        }
        with pytest.raises(TypeError, match=dict_msg):
            df.append(dict, sort=sort)
        with pytest.raises(TypeError, match=dict_msg):
            df.append([dict], sort=sort)
        with pytest.raises(TypeError, match=dict_msg):
            df.append([dict, dict], sort=sort)

        series_msg = 'Can only append a Series if ignore_index=True or .*'
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        series = pd.Series([7, 8, 9])
        with pytest.raises(TypeError, match=series_msg):
            df.append(series, sort=sort)
        with pytest.raises(TypeError, match=series_msg):
            df.append([series], sort=sort)
        with pytest.raises(TypeError, match=series_msg):
            df.append([series, series], sort=sort)

    indexes = [
        None,
        pd.Index([0, 1]),
        pd.Index(['a', 'b']),
        pd.Index(['a', 'b'], name='foo')
    ]

    @pytest.mark.parametrize('index1', indexes, ids=lambda x: repr(x))
    @pytest.mark.parametrize('index2', indexes, ids=lambda x: repr(x))
    def test_append_ignore_index(self, sort, index1, index2):
        # when appending with ignore_index=True,
        # all index content must be forgotten
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=index1)
        df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=index2)

        result = df1.append(df2, ignore_index=True)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6],
                                 [1, 2, 3], [4, 5, 6]])
        assert_frame_equal(result, expected)

        result = df1.append([df2], ignore_index=True)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6],
                                 [1, 2, 3], [4, 5, 6]])
        assert_frame_equal(result, expected)

        result = df1.append([df2, df2], ignore_index=True)
        expected = pd.DataFrame([[1, 2, 3], [4, 5, 6],
                                 [1, 2, 3], [4, 5, 6],
                                 [1, 2, 3], [4, 5, 6]])
        assert_frame_equal(result, expected)
