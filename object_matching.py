from operator import itemgetter
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
from django.db.models import Model, QuerySet
from itertools import chain, combinations
from functools import lru_cache


class FuzzyModelMatcher:
    """
    Pairing a list of data dictionaries with a query set of Django model instances using fuzzy attribute matching.
    The attributes from each data dictionary will be compared against each Django model attribute.
    For a given pair of data object and model instance, their matching cost is calculated as the number of attributes
    that they do not have equal. The pairing algorithm employed is known as `Hungarian Algorithm` as implemented by
    Scipy. Optimal pairings will be based on minimizing overall matching costs of the entire dataset.
    """

    def __init__(self, data: List[dict], direct_attributes: List[str], mandatory_attributes: List[str],
                 fuzzy_attributes: List[str], cls: type(Model) = None, qs: QuerySet = None, custom_cost_funcs=None):
        """
        :param data: a list of dictionaries each representing a data object. Must be ordered.
        :param direct_attributes: a list of strings representing name of attributes that can be used to identify an
            instance directly without fuzzy matching, such as primary key
        :param mandatory_attributes: a list of strings representing name of attributes that must be matched for a pair
            to be considered a match.
        :param fuzzy_attributes: a list of strings representing name of attributes that only need to be matched fuzzily
        :param cls: Django Model class, must be specified if argument `qs` is None.
        :param qs: A django model QuerySet, must be specified if argument `cls` is None. Must be ordered.
        :param custom_cost_funcs: a dictionary of attribute and function to be applied to modify match cost,
            each function takes argument of a data element and a django instance, returns a match cost.
        """
        self.cls = cls
        self.objs = qs
        self.data = data
        self.direct_attributes = direct_attributes
        self.mandatory_attributes = mandatory_attributes
        self.fuzzy_attributes = fuzzy_attributes
        self.max_match_queries = 20
        self.custom_cost_funcs = custom_cost_funcs

    @property
    @lru_cache(maxsize=None)
    def query_set(self):
        if self.objs is None and self.cls is None:
            raise ValueError("Must specify either `cls` or `qs` argument when initializing this FuzzyModelMatcher.")
        elif self.cls:
            return self.cls.objects.all().order_by('pk')
        else:
            return self.objs

    @property
    @lru_cache(maxsize=None)
    def query_set_ids(self):
        return self.query_set.values_list("pk", flat=True)

    def run(self, infer_last_remaining_pair=False):
        """
        Returns list of tuple containing index of data, Django model instances, and match cost.
        The index of data and Django model instance is relative to each of their given input (ie: init data and qs).

        :param infer_last_remaining_pair Defaults to False. If set to True, when there is only one unmatched data object
        and one unmatched model instance after fuzzy matching, they will be paired up as a match.
        """
        direct_matches = self.get_direct_matches()
        fuzzy_data_indices, fuzzy_instance_indices = self.make_fuzzy_match_candidates(direct_matches)
        matrix_data_index, matrix_instance_index, cost_matrix = self.make_cost_matrix(fuzzy_data_indices,
                                                                                      fuzzy_instance_indices)
        fuzzy_matches = self.make_assignments(matrix_data_index, matrix_instance_index, cost_matrix)
        matches = direct_matches + fuzzy_matches
        matches = self.infer_last_remaining_pair(matches, infer_last_remaining_pair)
        return matches

    def get_direct_matches(self) -> List[Tuple[int, int, int]]:
        """Try to find direct matches. Returns list of tuples containing data index, instance index, and
        match cost (always zero for direct match). """

        direct_matches = []
        for data_index, data_element in enumerate(self.data):
            direct_match = self.get_direct_match_data_element(data_element)
            if direct_match:
                instance_index, instance = direct_match
                direct_matches.append((data_index, instance_index, 0))

        return direct_matches

    def get_direct_match_data_element(self, attribute_data):
        """Given attribute-value dictionary, return direct match Django model instance."""
        direct_attribute_dict = {attr: val for attr, val in attribute_data.items() if attr in self.direct_attributes}
        instance = self.query_set.filter(**direct_attribute_dict).first()
        if instance:
            instance_index = list(self.query_set_ids).index(instance.pk)
            return instance_index, instance

    def make_fuzzy_match_candidates(self, exclude_pairs):
        """Make indices for data and instance to be used for fuzzy matching."""
        exclude_data_indices = set(map(itemgetter(0), exclude_pairs))
        exclude_instance_indices = set(map(itemgetter(1), exclude_pairs))
        fuzzy_data_indices = [i for i in range(len(self.data)) if i not in exclude_data_indices]
        fuzzy_instance_indices = [i for i in range(len(self.query_set_ids)) if i not in exclude_instance_indices]
        return fuzzy_data_indices, fuzzy_instance_indices

    def filter_dictionary_by_keys(self, d, keys):
        """Return dictionary of data with only attribute_names"""
        return {attr: d.get(attr) for attr in keys if attr not in self.custom_cost_funcs.keys() and d.get(attr)}

    def make_fuzzy_param_permutations(self, data_index):
        """Given a dictionary of attribute value pairs, make a list of permutation of attribute data dictionaries."""
        data = self.data[data_index]
        fuzzy_attribute_dict = self.filter_dictionary_by_keys(data, self.fuzzy_attributes)
        # a list of dictionaries containing subset permutations of self.fuzzy_attributes
        param_permutations = sorted(list(chain.from_iterable(map(dict, combinations(fuzzy_attribute_dict.items(), n))
                                                             for n in range(len(fuzzy_attribute_dict) + 1) if n)),
                                    key=lambda p: len(p), reverse=True)
        return param_permutations

    def get_fuzzy_matches(self, data_index):
        """Given attribute dictionary, return list of tuples containing Django models instance and match cost,
         in the form of [(data index, instance index, match cost), ...]
        """
        matches = []
        mandatory_attribute_dict = self.filter_dictionary_by_keys(self.data, self.mandatory_attributes)
        param_permutations = self.make_fuzzy_param_permutations(data_index)

        max_num_params = len(self.mandatory_attributes) + len(self.fuzzy_attributes)

        for i, params in enumerate(param_permutations):
            instance = self.query_set.filter(**mandatory_attribute_dict, **params).first()

            # only append if it is not already in the list
            if instance is not None and any(x for x in matches if x[0] == instance) is False:
                instance_index = list(self.query_set_ids).index(instance.pk)
                match_cost = max_num_params - len(params)

                matches.append((data_index, instance_index, match_cost))

            if i >= self.max_match_queries:
                break

        return self.modify_match_cost_with_custom_funcs(matches)

    def modify_match_cost_with_custom_funcs(self, matches):
        """Given list of match tuples, run custom cost functions on them, before returning modified match tuples."""
        if self.custom_cost_funcs is None:
            return matches
        else:
            modified_matches = []
            for match in matches:
                data_index, instance_index, match_cost = match
                for attribute, func in self.custom_cost_funcs.items():
                    data = self.data[data_index]
                    instance = self.query_set[instance_index]
                    cost_modifier = func(data, instance)

                    # this is a mandatory attribute and does not have perfect match, do not include as a match
                    if cost_modifier > 0 and attribute in self.mandatory_attributes:
                        continue
                    else:
                        modified_matches.append((data_index, instance_index, match_cost + cost_modifier))
            return modified_matches

    def make_cost_matrix(self, fuzzy_data_indices, fuzzy_instance_indices):
        """
        Computes a matrix of pairing costs between each given data element and Django model instance.
        Returns data element index map, model instance index map, and the matrix.
        The data element index map is a dictionary mapping the position of data element in matrix axis to its position
        in self.data.
        The instance index map is a dictionary mapping the position of the instance in matrix axis to its position in
        self.queryset.
        The matrix is a 2d array.
        :param fuzzy_data_indices: list of indexes of self.data to be used for fuzzy matching.
        :param fuzzy_instance_indices: list of indexes of self.queryset to be used for fuzzy matching.
        """
        cost_matrix = np.empty((len(fuzzy_data_indices), len(fuzzy_instance_indices)))  # make empty 2-d cost matrix
        # fill cost matrix with value greater than the greatest cost possible
        cost_matrix.fill(len(self.mandatory_attributes) + len(self.fuzzy_attributes) + 1)

        # row axis dictionary that maps index of data in matrix row to index of data in self.data
        matrix_data_index_map = {i: data_index for i, data_index in enumerate(fuzzy_data_indices)}
        # column axis dictionary that maps index of data in matrix column to index of data in self.query_set
        matrix_instance_index_map = {i: instance_index for i, instance_index in enumerate(fuzzy_instance_indices)}

        for matrix_data_index, data_index in enumerate(fuzzy_data_indices):

            matches = self.get_fuzzy_matches(data_index)

            for match in matches:
                _, instance_index, cost = match

                # the index of instance in matrix_instance_index_map
                matrix_instance_index = next((key for key, val in matrix_instance_index_map.items()
                                              if val == instance_index), None)

                cost_matrix[matrix_data_index][matrix_instance_index] = cost  # fill score matrix

        return matrix_data_index_map, matrix_instance_index_map, cost_matrix

    @staticmethod
    def make_assignments(matrix_data_index_map, matrix_instance_index_map, cost_matrix):
        """
        Given output of self.make_cost_matrix(), compute optimal pairing to minimize matching costs.
        returns a list of tuple, each tuple contains data element index, instance index, and their matching cost.
        """

        matches = []

        try:
            rows, cols = linear_sum_assignment(cost_matrix)
        except ValueError:
            # if unable to compute matches
            return matches

        for matrix_instance_idx, matrix_data_idx in enumerate(cols):
            instance_index = matrix_data_index_map[matrix_instance_idx]
            data_index = matrix_instance_index_map[matrix_data_idx]
            match_cost = cost_matrix[matrix_data_idx][matrix_instance_idx]
            matches.append((data_index, instance_index, match_cost))

        return matches

    def infer_last_remaining_pair(self, matches, infer_last_remaining_pair=False):
        """
        If there is only one unmatched data element and one unmatched Django model instances
        at the end of fuzzy matching, make them a pair.
        """
        if infer_last_remaining_pair is True:
            matched_data_indices = set(map(itemgetter(0), matches))
            matched_instance_indices = set(map(itemgetter(1), matches))
            if len(matched_instance_indices) == len(self.query_set_ids) - 1 and \
               len(matched_data_indices) == len(self.data) - 1:
                remaining_data_index = set(range(len(self.data))).difference(matched_data_indices)
                remaining_instance_index = set(range(len(self.query_set_ids))).difference(matched_instance_indices)
                matches.append((remaining_data_index, remaining_instance_index, None))

        return matches
