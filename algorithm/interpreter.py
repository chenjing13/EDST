import numpy as np
import pandas as pd
from collections import Counter


def local_interpreter(classifier, x, index):

    rules = []
    tree = classifier.tree_view()
    y_p = [[0, 0, 0] for i in range(3)]
    for n, t in enumerate(tree):
        rule = []
        pre = predict(t, x, rule, index)
        for key, value in pre.items():
            y_p[int(key)][n] = value / sum(pre.values())
        rule.append("result: {}".format(pre))
        rules.append(rule)
    li = [sum(y_p[0]), sum(y_p[1]), sum(y_p[2])]
    y = li.index(max(li))
    for n, i in enumerate(y_p[y]):
        if i != 0:
            print(rules[n])


def global_interpreter(classifier, index, x, y):

    rules = {}
    pre_y = classifier.predict(x)
    tree = classifier.tree_view()
    for n, row in enumerate(x):
        r1 = whole_tree_rules(tree[0], row,  index, ["Synergy", "Additive"])
        r2 = whole_tree_rules(tree[1], row, index, ["Synergy", "Antagonism"])
        r3 = whole_tree_rules(tree[2], row, index, ["Additive", "Antagonism"])
        if r1 is not None and r2 is not None and r3 is not None:
            if r1[1] == r2[1] == y[n]:
                rule = r1[0] + "   " + r2[0]
                if rule in rules:
                    rules[rule] = [pre_y[n], rules[rule][1] + 1, rules[rule][2] + 1]
                else:
                    rules[rule] = [pre_y[n], 1, 1]
            elif r1[1] == r2[1] != y[n]:
                rule = r1[0] + "   " + r2[0]
                if rule in rules:
                    rules[rule] = [pre_y[n], rules[rule][1] + 1, rules[rule][2]]
                else:
                    rules[rule] = [pre_y[n], 1, 0]
            elif r1[1] == r3[1] == y[n]:
                rule = r1[0] + "   " + r3[0]
                if rule in rules:
                    rules[rule] = [pre_y[n], rules[rule][1] + 1, rules[rule][2] + 1]
                else:
                    rules[rule] = [pre_y[n], 1, 1]
            elif r1[1] == r3[1] != y[n]:
                rule = r1[0] + "   " + r3[0]
                if rule in rules:
                    rules[rule] = [pre_y[n], rules[rule][1] + 1, rules[rule][2]]
                else:
                    rules[rule] = [pre_y[n], 1, 0]

            elif r2[1] == r3[1] == y[n]:
                rule = r2[0] + "   " + r3[0]
                if rule in rules:
                    rules[rule] = [pre_y[n], rules[rule][1] + 1, rules[rule][2] + 1]
                else:
                    rules[rule] = [pre_y[n], 1, 1]
            elif r2[1] == r3[1] != y[n]:
                rule = r2[0] + "   " + r3[0]
                if rule in rules:
                    rules[rule] = [pre_y[n], rules[rule][1] + 1, rules[rule][2]]
                else:
                    rules[rule] = [pre_y[n], 1, 0]

    distribute = dict(Counter(y))
    for key, r in rules.items():
        coverage = r[1] / distribute[r[0]]
        accuracy = r[2] / r[1]
        score = 2 * (coverage * accuracy)/(coverage + accuracy)
        rules[key].append(score)
    # print(sorted(rules.items(), key=lambda x: x[1][3], reverse=True))

    return rules


def global_importance(classifier, index):

    count = {}
    tree = classifier.tree_view()
    for n, t in enumerate(tree):
        whole_tree_feature_importance(t, index, count)
    # print(sorted(count.items(), key=lambda x: x[1], reverse=True))

    return count


def predict(node, row, rule, index):

    for i, split in enumerate(node["split"]):
        if row[node["index"]] <= split:
            if i == 0:
                rule.append("feature {} <= {}".format(index[node["index"]], split))
            else:
                rule.append("{} < feature {} <= {}".format(node["split"][i-1], index[node["index"]], split))
            if "index" in node['group_{}'.format(i)]:
                return predict(node['group_{}'.format(i)], row, rule, index)
            else:
                return node['group_{}'.format(i)]
        else:
            if i == len(node["split"])-1:
                rule.append("{} < feature {}".format(node["split"][i], index[node["index"]]))
                if "index" in node['group_{}'.format(i)]:
                    return predict(node['group_{}'.format(i)], row, rule, index)
                else:
                    return node['group_{}'.format(i)]
            else:
                continue


def whole_tree_rules(node, row, index, label_list, rule=""):

    for i, split in enumerate(node["split"]):
        if row[node["index"]] <= split:
            if rule != "":
                rule = rule + " and "
            if i == 0:
                rule = rule + "feature {} <= {}".format(index[node["index"]], split)
            else:
                rule = rule + "{} < feature {} <= {}".format(node["split"][i - 1], index[node["index"]], split)
            if "index" in node['group_{}'.format(i)]:
                return whole_tree_rules(node['group_{}'.format(i)], row, index, label_list, rule)
            else:
                sample = node['group_{}'.format(i)]
                label = max(sample, key=sample.get)
                label_list_copy = label_list.copy()
                if label == 0.0:
                    label_list_copy.remove("Additive")
                    rule = rule + " " + "then predict Additive than {}".format(label_list_copy[0])
                elif label == 1.0:
                    label_list_copy.remove("Antagonism")
                    rule = rule + " " + "then predict Antagonism than {}".format(
                        label_list_copy[0])
                else:
                    label_list_copy.remove("Synergy")
                    rule = rule + " " + "then predict Synergy than {}".format(label_list_copy[0])
                return [rule, label]

        else:
            if i == len(node["split"])-1:
                if rule != "":
                    rule = rule + " and "
                rule = rule + "{} < feature {}".format(node["split"][i], index[node["index"]])
                if "index" in node['group_{}'.format(i)]:
                    return whole_tree_rules(node['group_{}'.format(i)], row, index, label_list, rule)
                else:
                    sample = node['group_{}'.format(i)]
                    label = max(sample, key=sample.get)
                    label_list_copy = label_list.copy()
                    if label == 0.0:
                        label_list_copy.remove("Additive")
                        rule = rule + " " + "then predict Additive than {}".format(
                            label_list_copy[0])
                    elif label == 1.0:
                        label_list_copy.remove("Antagonism")
                        rule = rule + " " + "then predict Antagonism than {}".format(
                            label_list_copy[0])
                    else:
                        label_list_copy.remove("Synergy")
                        rule = rule + " " + "then predict Synergy than {}".format(
                            label_list_copy[0])
                    return [rule, label]
            else:
                continue


def whole_tree_feature_importance(node, index, count):

    if index[node["index"]] in count:
        count[index[node["index"]]] = count[index[node["index"]]] + 1
    else:
        count[index[node["index"]]] = 1
    for i, split in enumerate(node["split"]):
        if "index" in node['group_{}'.format(i)]:
            whole_tree_feature_importance(node['group_{}'.format(i)], index, count)




