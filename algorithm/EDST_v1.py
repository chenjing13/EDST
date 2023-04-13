from __future__ import division, absolute_import
from abc import abstractmethod
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
import interpreter
import pandas as pd
import numpy as np
import random
import math
# 随机交换


class tree_stump(object):

    def __init__(self):
        self.n_samples = 0
        self.tree = None

    def fit(self, x, y):

        x = pd.DataFrame(x)
        x.insert(loc=x.shape[1], column='y', value=y, allow_duplicates=False)
        data = np.array(x)
        self.n_samples = data.shape[0]
        self.tree = self.build_tree_stump(data)

        return self

    @staticmethod
    def test_split(index, value, dataset):  # Split a dataset based on an attribute and an attribute value

        group = []
        for row in dataset:
            if row[index] == value:
                group.append(row)
        group_value = max(list(row[-1] for row in group), key=list(row[-1] for row in group).count)

        return group, group_value

    def get_split(self, dataset):

        groups_raw = []
        values_raw = []
        indexs_raw = []
        for i in sorted(list(set(dataset[:, 0]))):
            group, value = self.test_split(0, i, dataset)
            groups_raw.append(group)
            values_raw.append(value)
            indexs_raw.append(i)
        for i in range(len(groups_raw)):
            groups_final = []
            values_final = []
            indexs_final = indexs_raw.copy()
            index = []
            for num in range(len(groups_raw)):
                if num in index:
                    continue
                else:
                    if num == len(groups_raw)-1:
                        groups_final.append(groups_raw[num])
                        values_final.append(values_raw[num])
                    else:
                        if values_raw[num] == values_raw[num+1]:
                            groups_final.append(groups_raw[num]+groups_raw[num+1])
                            values_final.append(values_raw[num])
                            index.append(num+1)
                            indexs_final.remove(indexs_raw[num])
                        else:
                            groups_final.append(groups_raw[num])
                            values_final.append(values_raw[num])
            groups_raw = groups_final
            values_raw = values_final
            indexs_raw = indexs_final
            if len(index) == 0:
                break
        proportion = []
        for group in groups_raw:
            proportion.append(dict(Counter([row[-1] for row in group])))

        return {'index': indexs_raw, 'value': values_raw, 'group': groups_raw, "proportion": proportion}

    def build_tree_stump(self, dataset):

        root = self.get_split(dataset)

        return root

    def _predict(self, node, row):

        for num, x in enumerate(node["index"]):
            if row <= x:
                return node["value"][num]
            elif num == len(node["index"])-1:
                return node["value"][num]
            else:
                continue

    def _predict_log_proba(self, node, row):

        for num, x in enumerate(node["index"]):
            if row <= x:
                return node["proportion"][num]
            elif num == len(node["index"])-1:
                return node["proportion"][num]
            else:
                continue

    def predict(self, x):
        x = np.array(x)
        y = []
        for num, row in enumerate(x):
            y_pred = self._predict(self.tree, row)
            y.append(y_pred)

        return y


class Decision_trees(BaseEstimator):

    def __init__(self, *, max_depth=10, min_samples_leaf=0.1, criterion="gini"):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_samples = 0
        self.criterion = criterion
        self.sample_weight = None
        self.classes_ = None
        self.tree = []

    def fit(self,  data, y, tree_stp, sample_weight=None):

        self.sample_weight = sample_weight
        self.classes_ = np.array(sorted(list(dict(Counter(y)).keys())))
        self.n_samples = data.shape[0]
        self.min_samples_leaf = self.n_samples * self.min_samples_leaf
        data.insert(loc=data.shape[1], column='y', value=y, allow_duplicates=False)
        data_1 = pd.concat([data.loc[(data['y'] == 2)], data.loc[(data['y'] == 0)]])
        data_2 = pd.concat([data.loc[(data['y'] == 2)], data.loc[(data['y'] == 1)]])
        data_3 = pd.concat([data.loc[(data['y'] == 1)], data.loc[(data['y'] == 0)]])
        data_1 = np.array(data_1)
        tree = self._fit_sta(data_1[:, :-1], data_1[:, -1], tree_stp[0])
        self.tree.append(tree)
        data_2 = np.array(data_2)
        tree = self._fit_sta(data_2[:, :-1], data_2[:, -1], tree_stp[1])
        self.tree.append(tree)
        data_3 = np.array(data_3)
        tree = self._fit_sta(data_3[:, :-1], data_3[:, -1], tree_stp[2])
        self.tree.append(tree)

        return self

    def _fit_sta(self, x, y, tree_stp):

        data = pd.DataFrame(x)
        data.insert(loc=data.shape[1], column='y', value=y, allow_duplicates=False)
        data = np.array(data)
        tree = self.build_tree(data, self.max_depth, self.min_samples_leaf, tree_stp[0])

        return tree

    def build_tree(self, dataset, max_depth, min_size, evaluate):

        x_resampled, y_resampled = SMOTE(random_state=0).fit_resample(dataset[:, :-1], list(map(int, dataset[:, -1])))
        dataset = np.array(pd.concat([pd.DataFrame(x_resampled), pd.DataFrame(y_resampled)], axis=1))
        if self.sample_weight is None:
            sample_weight = [1/dataset.shape[0] for i in range(dataset.shape[0])]
        elif self.sample_weight == "balance":
            sample_weight = [1/list(dataset[:, -1]).count(i) for i in dataset[:, -1]]
        else:
            sample_weight = self.sample_weight
        dataset = np.array(pd.concat([pd.DataFrame(dataset), pd.DataFrame(sample_weight)], axis=1))
        root = self.get_split(dataset, evaluate)
        self.split(root, max_depth, min_size, 1, evaluate)

        return root

    def test_split(self, index, dataset, evaluate):  # Split a dataset based on an attribute and an attribute value

        tree = evaluate[index].tree
        dataset = np.array(dataset)
        x = dataset[:, index]
        groups = [[] for i in range(len(tree["index"]))]
        for num, row in enumerate(x):
            for n, value in enumerate(tree["index"]):
                if row <= value:
                    groups[n].append(dataset[num, :])
                    break
                else:
                    continue

        return groups, tree["index"]

    def get_split(self, dataset, evaluate,  form_index=-1):

        b_index, b_value, b_score, b_groups, b_split = np.inf, np.inf, np.inf, None, None
        for index in range(len(dataset[0]) - 2):
            if index == form_index:
                continue
            else:
                groups, group_index = self.test_split(index, dataset, evaluate)
                groups, group_value, group_index = self.merge(groups, group_index)
                if self.criterion == "gini":

                    gini = self.gini_index(groups, group_value)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups, b_split = index, group_value, gini, groups, group_index

                elif self.criterion == "ID3":

                    entropy = self.ID3_index(groups, group_value)
                    if -entropy < b_score:
                        b_index, b_value, b_score, b_groups, b_split = index, group_value, -entropy, groups, group_index

                elif self.criterion == "C45":

                    entropy = self.C45_index(groups, group_value)
                    if -entropy < b_score:
                        b_index, b_value, b_score, b_groups, b_split = index, group_value, -entropy, groups, group_index

        if len(b_value) !=0:
            return {'index': b_index, 'value': b_value, 'groups': b_groups, "split": b_split}
        else:
            return self.to_terminal(dataset)

    def merge(self, groups, group_index):
        size = 0
        for group in groups:
            size = size + len(group)
        for i in range(len(groups)):
            groups_final = []
            index = []
            indexs_final = group_index.copy()
            for n, group in enumerate(groups):
                if n in index:
                    continue
                else:
                    if len(group) <= math.ceil(size * 0.01):
                        if len(groups) <= 2:
                            groups_final.append(groups[0] + groups[1])
                            indexs_final.remove(group_index[0])
                            break
                        else:
                            if n == 0:
                                groups_final.append(groups[n] + groups[n + 1])
                                groups_final.append(groups[n + 2])
                                index.append(n + 1)
                                index.append(n + 2)
                                indexs_final.remove(group_index[n])
                            elif n == len(groups) - 1:
                                groups_final[-1] = groups_final[-1] + groups[n]
                                indexs_final.remove(group_index[n-1])

                            else:
                                groups_final[-1] = groups_final[-1] + groups[n] + groups[n+1]
                                indexs_final.remove(group_index[n-1])
                                indexs_final.remove(group_index[n])
                                index.append(n + 1)
                                index.append(n + 2)
                                if len(groups) > n + 2:
                                    groups_final.append(groups[n + 2])
                    else:
                        groups_final.append(groups[n])
            groups = groups_final
            group_index = indexs_final
            if len(index) == 0:
                break
        groups_value = []
        del_index = []
        for g, group in enumerate(groups):
            try:
                groups_value.append(max(dict(Counter([row[-2] for row in group])), key=dict(Counter([row[-2] for row in group])).get))
            except:
                del_index.append(g)
        groups = [i for num, i in enumerate(groups) if num not in del_index]
        group_index = [i for num, i in enumerate(group_index) if num not in del_index]

        return groups, groups_value, group_index

    @staticmethod
    def to_terminal(group):  # Create a terminal node value

        outcomes = [row[-2] for row in group]
        if len(outcomes) == 0:
            outcomes = [2]

        return dict(Counter(outcomes))

    def split(self, node, max_depth, min_size, depth, evaluate):  # Create child splits for a node or make terminal

        if "index" in node:
            form_index = node['index']
            groups = node['groups']
            del (node['groups'])
            for i, group in enumerate(groups):
                if len(group) <= min_size or depth >= max_depth:
                    if self.to_terminal(group):
                        node['group_{}'.format(i)] = self.to_terminal(group)
                        continue
                    else:
                        node['group_{}'.format(i)] = {node["value"][i]: 1}
                        continue
                if len(set([row[-2] for row in group])) == 1:  # process left child
                    node['group_{}'.format(i)] = self.to_terminal(group)
                    continue
                else:
                    node['group_{}'.format(i)] = self.get_split(group, evaluate, form_index)
                    self.split(node['group_{}'.format(i)], max_depth, min_size, depth + 1, evaluate)

    def _predict(self, node, row):

        for i, split in enumerate(node["split"]):
            if row[node["index"]] <= split:
                if "index" in node['group_{}'.format(i)]:
                    return self._predict(node['group_{}'.format(i)], row)
                else:
                    return node['group_{}'.format(i)]
            else:
                if i == len(node["split"])-1:
                    if "index" in node['group_{}'.format(i)]:
                        return self._predict(node['group_{}'.format(i)], row)
                    else:
                        return node['group_{}'.format(i)]
                else:
                    continue

    def predict(self, x):
        x = pd.DataFrame(x)
        x = np.array(x)
        y = []
        for num, row in enumerate(x):
            y_p = [0, 0, 0]
            y_pred_1 = self._predict(self.tree[0], row)
            for key, value in y_pred_1.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_1.values())
            y_pred_2 = self._predict(self.tree[1], row)
            for key, value in y_pred_2.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_2.values())
            y_pred_3 = self._predict(self.tree[2], row)
            for key, value in y_pred_3.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_3.values())
            y.append(y_p.index(max(y_p)))

        return y

    def predict_proba(self, x):
        x = pd.DataFrame(x)
        x = np.array(x)
        df_empty = []
        for num, row in enumerate(x):
            y_p = [0, 0, 0]
            y_pred_1 = self._predict(self.tree[0], row)
            for key, value in y_pred_1.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_1.values())
            y_pred_2 = self._predict(self.tree[1], row)
            for key, value in y_pred_2.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_2.values())
            y_pred_3 = self._predict(self.tree[2], row)
            for key, value in y_pred_3.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_3.values())
            df_empty.append(y_p)

        return np.array(df_empty)

    def gini_index(self, groups, groups_values):  # Calculate the Gini index for a split dataset

        gini = 0.0
        number = 0.0
        for group in groups:
            number += len(group)
        for i, group in enumerate(groups):
            size = sum(row[-1] for row in group)
            if size == 0:
                continue
            else:
                proportion = sum([row[-1] for row in group if row[-2] == groups_values[i]]) / float(size)
                gini_group = 1 - (proportion ** 2) - ((1.0 - proportion) ** 2)
                gini += gini_group * len(group) / number

        return gini

    def ID3_index(self, groups, groups_values):

        entropy = 0.0
        number = 0.0
        all_group = []
        for group in groups:
            number += len(group)
            all_group += group
        for value in list(set(groups_values)):
            proportion = sum([row[-1] for row in all_group if row[-2] == value]) / float(sum(row[-1] for row in all_group))
            if proportion == 0.0:
                continue
            else:
                entropy -= proportion * np.log2(proportion)
        entropy_groups = 0.0
        for i, group in enumerate(groups):
            size = sum(row[-1] for row in group)
            if size == 0:
                continue
            proportion = sum([row[-1] for row in group if row[-2] == groups_values[i]]) / float(size)
            if proportion == 1.0 or proportion == 0.0:
                entropy_group = 0.0
            else:
                entropy_group = -proportion * np.log2(proportion)-(1-proportion) * np.log2(1-proportion)
            entropy_groups += entropy_group * len(group) / number
        entropy = entropy - entropy_groups

        return entropy

    def C45_index(self, groups, groups_values):

        entropy = 0.0
        number = 0.0
        all_group = []
        for group in groups:
            number += len(group)
            all_group += group
        for value in list(set(groups_values)):
            proportion = sum([row[-1] for row in all_group if row[-2] == value]) / float(sum(row[-1] for row in all_group))
            if proportion == 0.0:
                continue
            else:
                entropy -= proportion * np.log2(proportion)
        entropy_groups = 0.0
        splitInfo = 0.0
        for i, group in enumerate(groups):
            size = sum(row[-1] for row in group)
            if size == 0:
                continue
            proportion = sum([row[-1] for row in group if row[-2] == groups_values[i]]) / float(size)
            if proportion == 1.0 or proportion == 0.0:
                entropy_group = 0.0
            else:
                entropy_group = -proportion * np.log2(proportion)-(1-proportion) * np.log2(1-proportion)
            splitInfo -= (len(group) / number) * np.log2(len(group) / number)
            entropy_groups += entropy_group * len(group) / number
        entropy = entropy - entropy_groups
        if splitInfo == 0:
            return entropy
        else:
            return entropy/splitInfo


class Bagging(object):

    @abstractmethod
    def __init__(self, base_estimator, estimators=100, change=1/4, rebuild=False):
        self.base_estimator = base_estimator
        self.estimators = estimators
        self.rebuild = rebuild
        self.change = change
        self.label_list = None
        self.minmax = None
        self.feature = None
        self.evaluate = None
        self.group = []
        self.weight_ = None

    def fit(self, data, label):
        scaler = MinMaxScaler()
        data = pd.DataFrame(data)
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data)
        self.feature = data.shape[1]
        data.insert(loc=data.shape[1], column='y', value=label, allow_duplicates=False)
        self.label_list = list(set(label))
        self.minmax = scaler
        if self.rebuild:
            self.weight_ = [1/list(label).count(i) for i in label]
            ts = Parallel(n_jobs=4)(delayed(self._parallel_stump)(data) for i in range(self.estimators * 2))
            ts = self.stump_change(ts)
            group = Parallel(n_jobs=4)(delayed(self._parallel_bagging)(data, ts[i]) for i in range(self.estimators * 2))
        else:
            self.weight_ = [1 for i in range(len(data.iloc[:, -1]))]
            ts = Parallel(n_jobs=4)(delayed(self._parallel_stump)(data) for i in range(self.estimators))
            ts = self.stump_change(ts)
            group = Parallel(n_jobs=4)(delayed(self._parallel_bagging)(data, ts[i]) for i in range(self.estimators))
        for clf, value in group:
            self.group.append((value, clf))
        self.group = sorted(self.group, key=lambda x: x[0], reverse=True)

    def _parallel_stump(self, data):

        nums = []
        data_copy = data.copy()
        sampling_rate = self.weight_.copy()
        sampling_rate = sampling_rate/np.sum(sampling_rate)
        data_copy.insert(loc=data_copy.shape[1], column='sampling_rate', value=sampling_rate, allow_duplicates=False)
        for i in range(len(sampling_rate)):
            nums = nums + [i] * int(sampling_rate[i] * 10000)
        data_copy = data_copy.loc[nums]
        sub_data = data_copy.sample(n=data.shape[0], replace=True)
        sub_data.index = range(len(sub_data))
        data_1 = pd.concat([sub_data.loc[(sub_data['y'] == 2)], sub_data.loc[(sub_data['y'] == 0)]])
        data_2 = pd.concat([sub_data.loc[(sub_data['y'] == 2)], sub_data.loc[(sub_data['y'] == 1)]])
        data_3 = pd.concat([sub_data.loc[(sub_data['y'] == 1)], sub_data.loc[(sub_data['y'] == 0)]])
        sg1 = self.stump_group(data_1.iloc[:, :-2], data_1.iloc[:, -2])
        sg2 = self.stump_group(data_2.iloc[:, :-2], data_2.iloc[:, -2])
        sg3 = self.stump_group(data_3.iloc[:, :-2], data_3.iloc[:, -2])
        group = [[sg1, sg2, sg3], sub_data]

        return group

    def _parallel_bagging(self, data, ts):

        clf = self.base_estimator.fit(ts[1].iloc[:, :-2], ts[1].iloc[:, -2], ts[0])
        
        return clf, f1_score(data.iloc[:, -1], clf.predict(data.iloc[:, :-1]), average='macro')

    def stump_change(self, ts_group):

        ts_group = list(ts_group)
        for o in range(3):
            feature_list = [[] for i in range(self.feature)]
            feature_copy = [[] for i in range(self.feature)]

            for num, i in enumerate(ts_group):
                matrix = i[0][o][1]
                comp_sim = []
                for j in matrix:
                    comp_sim.append(sum(cosine_similarity([j], matrix)))
                similar_indices = np.array(comp_sim).argsort().flatten()[-int(len(matrix)*self.change):]
                for z in similar_indices:
                    feature_list[z].append(i[0][o][0][z])
                    feature_copy[z].append(num)
            for f, feature in enumerate(feature_list):
                random.shuffle(feature)
                for n, num in enumerate(feature_copy[f]):
                    ts_group[num][0][o][0][f] = feature[n]

        return ts_group

    def predict(self, data):

        scaler = self.minmax
        data = scaler.transform(data)
        data = pd.DataFrame(data)
        y_pred_1 = []
        y_pred_2 = []
        predictions = []
        for num, e in enumerate(self.group):
            predictions.append(e[1].predict_proba(data))
        result = []
        for i in range(data.shape[0]):
            row = np.array(predictions)[:, i]
            tree_max = []
            count = [0, 0, 0]
            for j in range(self.estimators):
                tree_max.append(max(row[j]))
                summy = sum(row[j])
                for key, value in enumerate(row[j]):
                    count[key] = count[key] + value/summy
            result.append(count)
            y_pred_1.append(count.index(max(count)))
            y_pred_2.append(max(set(tree_max), key=tree_max.count))

        return y_pred_1

    def predict_proba(self, data):

        scaler = self.minmax
        data = scaler.transform(data)
        data = pd.DataFrame(data)
        predictions = []
        for num, e in enumerate(self.group):
            predictions.append(e[1].predict_proba(data))
        result = []
        for i in range(data.shape[0]):
            row = np.array(predictions)[:, i]
            count = [0, 0, 0]
            for j in range(self.estimators):
                summy = sum(row[j])
                for key, value in enumerate(row[j]):
                    count[key] = count[key] + value/summy
            result.append(count)

        return result

    def stump_group(self, x, y):
        evaluate = []
        result = []
        for m in x.columns.values:
            clf = tree_stump().fit(x[m].replace(-1, 1), y)
            result.append(np.array(clf.predict(x[m].replace(-1, 1))))
            evaluate.append(clf)

        return evaluate, result

    def feature_importance(self, index):

        importance = {}
        for g in self.group[:self.estimators]:
            count = interpreter.global_interpreter(g[1], index)
            n, m = Counter(importance), Counter(count)
            importance = dict(n + m)

        return importance

    def rule_importance(self, index, x, y):

        importance = {}
        for g in self.group[:self.estimators]:
            count = interpreter.global_interpreter(g[1], index, x, y)
            importance.update(count)

        return importance
