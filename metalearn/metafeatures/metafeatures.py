from copy import deepcopy
import time
import typing

import numpy as np
import pandas as pd

from metalearn.metafeatures.base import collectordict, ResourceComputer
from metalearn.metafeatures.common_operations import dtype_is_numeric
import metalearn.metafeatures.constants as consts

from metalearn.metafeatures.decision_tree_metafeatures import resources_info as dt_resources
from metalearn.metafeatures.general_resource_computers import resources_info as general_resources
from metalearn.metafeatures.text_metafeatures import resources_info as text_resources

from metalearn.metafeatures.decision_tree_metafeatures import metafeatures_info as dt_metafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import metafeatures_info as info_theoretic_metafeatures
from metalearn.metafeatures.landmarking_metafeatures import metafeatures_info as landmarking_metafeatures
from metalearn.metafeatures.simple_metafeatures import metafeatures_info as simple_metafeatures
from metalearn.metafeatures.statistical_metafeatures import metafeatures_info as statistical_metafeatures
from metalearn.metafeatures.text_metafeatures import metafeatures_info as text_metafeatures


class Metafeatures(object):
    """
    Computes metafeatures on a given tabular dataset (pandas.DataFrame) with
    categorical targets. These metafeatures are particularly useful for
    computing summary statistics on a dataset and for machine learning and
    meta-learning applications.
    """

    _resources_info = collectordict()
    _resources_info.update(dt_resources)
    _resources_info.update(general_resources)
    _resources_info.update(text_resources)

    # noop resource computers for the user-provided resources
    # `_get_arguments` and `_resource_is_target_dependent` assumes ResourceComputer's
    for resource_name in ["X_raw", "Y", "column_types", "sample_shape", "seed_base", "n_folds"]:
        _resources_info[resource_name] = ResourceComputer(lambda: None, [resource_name])

    _mfs_info = [
        dt_metafeatures,
        info_theoretic_metafeatures,
        landmarking_metafeatures,
        simple_metafeatures,
        statistical_metafeatures,
        text_metafeatures,
    ]

    for mf_info in _mfs_info:
        _resources_info.update(mf_info)

    IDS: typing.Sequence[str] = [mf_id for mfs_info in _mfs_info for mf_id in mfs_info.keys()]

    @classmethod
    def list_metafeatures(cls, group: str = 'all') -> typing.Sequence[str]:
        """
        Lists the ids of computable metafeatures.

        Parameters
        ----------
        group: str
            the name of a grouping of metafeatures

        Returns
        -------
        A list of metafeature ids
        """

        # todo make group for intractable metafeatures for wide datasets or
        # datasets with high cardinality categorical columns:
        # PredPCA1, PredPCA2, PredPCA3, PredEigen1, PredEigen2, PredEigen3,
        # PredDet, kNN1NErrRate, kNN1NKappa, LinearDiscriminantAnalysisKappa,
        # LinearDiscriminantAnalysisErrRate

        valid_metafeature_group_values = [_group.value for _group in consts.MetafeatureGroup]
        if group not in valid_metafeature_group_values:
            raise ValueError('Invalid group: {}. Must be one of {}'.format(group, valid_metafeature_group_values))

        if group == consts.MetafeatureGroup.ALL.value:
            return [id_ for id_ in cls.IDS]
        else:
            return [id_ for id_ in cls.IDS if consts.MetafeatureGroup(group) in cls._resources_info[id_].groups]

    # TODO: make most args kw only
    def compute(
        self, X: pd.DataFrame, Y: pd.Series = None, column_types: typing.Mapping[str, str] = None,
        metafeature_ids: typing.Sequence[str] = None, exclude: typing.Sequence[str] = None, sample_shape = None,
        seed: int = None, n_folds: int = 2, verbose: bool = False, timeout = None, return_times: bool = False,
        groups: typing.Sequence[str] = None, exclude_groups: typing.Sequence[str] = None
    ) -> dict:
        """
        Parameters
        ----------
        X: pandas.DataFrame
            The dataset features
        Y: pandas.Series
            The dataset targets
        column_types: Mapping[str, str]
            Maps column names to semantic types. Valid typs include "NUMERIC", "CATEGORICAL", and "TEXT".
            Must include Y.name if Y is not None.
        metafeature_ids: Sequence[str]
            The metafeatures to compute. None indicates to compute all metafeatures. At least one of `metafeature_ids`
            and `exclude` must be None.
        exclude: Sequence[str]
            All metafeatures except those listed in `exclude` will be computed. At least one of `metafeature_ids` and
            `exclude` must be None.
            **TODO** This parameter will be renamed in a future version. See
            https://github.com/byu-dml/metalearn/issues/210.
        sample_shape: tuple
            The shape of X after sampling (X,Y) uniformly. Default is (None, None), indicate not to sample rows or
            columns.
        seed: int
            The seed used to generate pseudo-random numbers. When None is given, a seed will be generated
            pseudo-randomly. This can be used for reproducibility of metafeatures. A generated seed can be accessed
            through the 'seed' property, after calling this method.
        n_folds: int
            the number of cross validation folds used by the
            landmarking metafeatures. also affects the sample_shape validation
        verbose: bool
            When True, prints the ID of each
            metafeature right before it is about to be computed.
        timeout: float
            If timeout is None, compute_metafeatures
            will be run to completion. Otherwise, execution will halt after
            approximately timeout seconds. Any metafeatures that have not been
            computed will be labeled 'TIMEOUT'.
        return_times: bool
            When true, includes compute times for each metafeature. **Note** compute times are overestimated.
            See https://github.com/byu-dml/metalearn/issues/205.
        groups: list
            The metafeature groups to be computed. Must consist only of values enumerated in
            constants.MetafeatureGroup. At least one of groups and exclude_groups must be None.
        exclude_groups: list
            The metafeature groups to exclude from computation. Must consist only of values enumerated in
            constants.MetafeatureGroup. At least one of groups and exclude_groups must be None.

        Returns
        -------
        A dictionary mapping the metafeature id to another dictionary containing
        the `value` and `compute_time` (if requested) of the referencing
        metafeature. The value is typically a number, but can be a string
        indicating a reason why the value could not be computed.
        """
        start_time = time.time()
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
            n_folds, verbose, return_times, groups, exclude_groups
        )
        if timeout is None:
            def check_time():
                pass
        else:
            def check_time():
                if time.time() - start_time > timeout:
                    raise TimeoutError()
        self._check_timeout = check_time

        if column_types is None:
            column_types = self._infer_column_types(X, Y)
        metafeatures_to_compute = self._get_metafeatures_to_compute(metafeature_ids, exclude, groups, exclude_groups)
        exclude = None
        if sample_shape is None:
            sample_shape = (None, None)
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        self._validate_compute_arguments(
            X, Y, column_types, metafeatures_to_compute, exclude, sample_shape, seed,
            n_folds, verbose, return_times, groups, exclude_groups
        )

        self._init_resources(
            X, Y, column_types, sample_shape, seed, n_folds
        )

        if metafeatures_to_compute is not None:
            computed_metafeatures = {
                name: self._format_resource(consts.TIMEOUT, 0)
                for name in metafeatures_to_compute
            }
            try:
                for mf_id in metafeatures_to_compute:
                    self._check_timeout()
                    if verbose:
                        print(mf_id)
                    if (
                        consts.MetafeatureGroup.TARGET_DEPENDENT in self._resources_info[mf_id].groups
                    ) and (
                        Y is None or column_types[Y.name] == consts.NUMERIC
                    ):
                        if Y is None:
                            value = consts.NO_TARGETS
                        else:
                            value = consts.NUMERIC_TARGETS
                        compute_time = None
                    else:
                        value, compute_time = self._get_resource(mf_id)

                    computed_metafeatures[mf_id] = self._format_resource(value, compute_time)
            except TimeoutError:
                pass

        if not return_times:
            for mf, result_dict in computed_metafeatures.items():
                del result_dict[consts.COMPUTE_TIME_KEY]

        return computed_metafeatures

    def _format_resource(self, value, compute_time):
        """Formats the resource data as a dict"""
        return {
            consts.VALUE_KEY: value,
            consts.COMPUTE_TIME_KEY: compute_time
        }

    def _init_resources(
        self, X, Y, column_types, sample_shape, seed, n_folds
    ):
        # Add the base resources to our resources hash
        self._resources = {
            "X_raw": self._format_resource(X, 0.),  # TODO: rename to X
            "Y": self._format_resource(Y, 0.),
            "column_types": self._format_resource(column_types, 0.),
            "sample_shape": self._format_resource(sample_shape, 0.),
            "seed_base": self._format_resource(seed, 0.),
            "n_folds": self._format_resource(n_folds, 0.)
        }

    def _validate_compute_arguments(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        for f in [
            self._validate_X, self._validate_Y, self._validate_column_types,
            self._validate_metafeature_ids, self._validate_sample_shape,
            self._validate_n_folds, self._validate_verbose,
            self._validate_return_times, self._validate_groups
        ]:
            f(
                X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
                n_folds, verbose, return_times, groups, exclude_groups
            )

    def _validate_X(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pandas.DataFrame')
        if X.empty:
            raise ValueError('X must not be empty')

    def _validate_Y(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not isinstance(Y, pd.Series) and not Y is None:
            raise TypeError('Y must be of type pandas.Series')
        if Y is not None and Y.shape[0] != X.shape[0]:
            raise ValueError('Y must have the same number of rows as X')

    def _validate_column_types(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not column_types is None:
            invalid_column_types = {}
            columns = list(X.columns)
            if not Y is None:
                columns.append(Y.name)
            for col in columns:
                if col not in column_types:
                    raise ValueError(
                        f"Column type not specified for column {col}"
                    )
                col_type = column_types[col]
                # todo: add consts.TEXT to check. Additionally add consts.TEXT to all tests that check for column types
                if not col_type in [consts.NUMERIC, consts.CATEGORICAL, consts.TEXT]:
                    invalid_column_types[col] = col_type
            if len(invalid_column_types) > 0:
                raise ValueError(
                    f"Invalid column types: {invalid_column_types}. Valid types " +
                    f"include {consts.NUMERIC} and {consts.CATEGORICAL} and {consts.TEXT}."
                )

    def _validate_metafeature_ids(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        ids = None
        if metafeature_ids is not None and exclude is not None:
            raise ValueError("metafeature_ids and exclude cannot both be non-null")
        elif metafeature_ids is not None:
            ids = metafeature_ids
            list_label = 'requested'
        elif exclude is not None:
            ids = exclude
            list_label = 'excluded'
        if ids is not None:
            invalid_metafeature_ids = [
                mf for mf in ids if mf not in self._resources_info
            ]
            if len(invalid_metafeature_ids) > 0:
                raise ValueError(
                    'One or more {} metafeatures are not valid: {}'.
                    format(list_label, invalid_metafeature_ids)
                )

    def _validate_sample_shape(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not sample_shape is None:
            if not type(sample_shape) in [tuple, list]:
                raise ValueError(
                    "`sample_shape` must be of type `tuple` or `list`"
                )
            if len(sample_shape) != 2:
                raise ValueError("`sample_shape` must be of length 2")
            if not sample_shape[0] is None and sample_shape[0] < 1:
                raise ValueError("Cannot sample less than one row")
            if not sample_shape[1] is None and sample_shape[1] < 1:
                raise ValueError("Cannot sample less than 1 column")
            if not sample_shape[0] is None and not Y is None:
                min_samples = Y.unique().shape[0] * n_folds
                if sample_shape[0] < min_samples:
                    raise ValueError(
                        f"Cannot sample less than {min_samples} rows from Y"
                    )

    def _validate_n_folds(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not dtype_is_numeric(type(n_folds)) or (n_folds != int(n_folds)):
            raise ValueError(f"`n_folds` must be an integer, not {n_folds}")
        if n_folds < 2:
            raise ValueError(f"`n_folds` must be >= 2, but was {n_folds}")
        if (
            Y is not None and column_types is not None and column_types[Y.name] != consts.NUMERIC and
            metafeature_ids is not None
        ):
            # when computing landmarking metafeatures, there must be at least
            # n_folds instances of each class of Y
            landmarking_mfs = self.list_metafeatures(group=consts.MetafeatureGroup.LANDMARKING.value)
            if len(list(filter(
                lambda mf_id: mf_id in landmarking_mfs,metafeature_ids
            ))):
                Y_grouped = Y.groupby(Y)
                for group_id, group in Y_grouped:
                    if group.shape[0] < n_folds:
                        raise ValueError(
                            "The minimum number of instances in each class of" +
                            f" Y is n_folds={n_folds}. Class {group_id} has " +
                            f"{group.shape[0]}."
                        )

    def _validate_verbose(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not type(verbose) is bool:
            raise ValueError("`verbose` must be of type bool.")

    def _validate_return_times(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if not type(return_times) is bool:
            raise ValueError("`return_times` must be of type bool.")

    def _validate_groups(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose, return_times, groups, exclude_groups
    ):
        if groups is not None and exclude_groups is not None:
            raise ValueError('at least one of `groups` and `exclude_groups` must be `None`')

        if groups is not None:
            invalid_groups = [group for group in groups if consts.MetafeatureGroup(group) not in consts.MetafeatureGroup]
            if invalid_groups:
                raise ValueError('Invalid groups: {}. Valid groups include: {}'.format(
                    invalid_groups, [group.value for group in consts.MetafeatureGroup])
                )

        if exclude_groups is not None:
            invalid_groups = [group for group in exclude_groups if consts.MetafeatureGroup(group) not in consts.MetafeatureGroup]
            if invalid_groups:
                raise ValueError('Invalid exclude_groups: {}. Valid exclud_groups include: {}'.format(
                    invalid_groups, [group.value for group in consts.MetafeatureGroup])
                )

    # todo: intelligently infer TEXT data type
    def _infer_column_types(self, X, Y):
        column_types = {}
        for col_name in X.columns:
            if dtype_is_numeric(X[col_name].dtype):
                column_types[col_name] = consts.NUMERIC
            else:
                column_types[col_name] = consts.CATEGORICAL
        if not Y is None:
            if dtype_is_numeric(Y.dtype):
                column_types[Y.name] = consts.NUMERIC
            else:
                # todo: get number of unique values in col_name, compute unique/total ratio. Use ratio to infer type

                column_types[Y.name] = consts.CATEGORICAL
        return column_types

    def _get_metafeatures_to_compute(self, metafeature_ids, exclude, groups, exclude_groups) -> typing.Set[str]:
        assert metafeature_ids is None or exclude is None
        assert groups is None or exclude_groups is None

        metafeatures_to_compute: typing.Set[str] = set()

        if metafeature_ids is None and groups is None:
            metafeatures_to_compute = set(self.list_metafeatures())

        if metafeature_ids is not None:
            metafeatures_to_compute.update(metafeature_ids)

        if groups is not None:
            for group in groups:
                metafeatures_to_compute.update(self.list_metafeatures(group))

        if exclude is not None:
            for mf in exclude:
                metafeatures_to_compute.discard(mf)

        if exclude_groups is not None:
            for group in exclude_groups:
                metafeatures_to_compute.difference_update(self.list_metafeatures(group))

        return metafeatures_to_compute

    def _get_resource(self, resource_id):
        self._check_timeout()
        if not resource_id in self._resources:
            resource_computer = self._resources_info[resource_id]
            args, total_time = self._get_arguments(resource_id)
            return_resources = resource_computer.returns
            start_timestamp = time.perf_counter()
            computed_resources = resource_computer(**args)
            compute_time = time.perf_counter() - start_timestamp
            total_time += compute_time
            for res_id, computed_resource in zip(
                return_resources, computed_resources
            ):
                self._resources[res_id] = self._format_resource(computed_resource, total_time)
        resource = self._resources[resource_id]
        return resource[consts.VALUE_KEY], resource[consts.COMPUTE_TIME_KEY]

    def _get_arguments(self, resource_id):
        resolved_parameters = {}
        total_time = 0.0
        for parameter, argument in self._resources_info[resource_id].argmap.items():
            argument_type = type(argument)
            if parameter == "seed":
                seed_base, compute_time = self._get_resource("seed_base")
                argument += seed_base
            elif argument_type is str:
                if argument in self._resources_info:
                    argument, compute_time = self._get_resource(argument)
                else:
                    compute_time = 0
            elif dtype_is_numeric(argument_type):
                compute_time = 0
            else:
                raise TypeError(f'unhandled argument type: {argument_type}')
            resolved_parameters[parameter] = argument
            total_time += compute_time
        return (resolved_parameters, total_time)
