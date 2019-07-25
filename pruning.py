#%%
from DataGenerator import DataGenerator
from MyDecisionTree import MyDecisionTree, Feature
from DataGenerator import DataSet
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def prune(validate_features, validate_labels):
    max_score_list = []
    for i in range(len(DataSet.rules)):
        max_score = get_score(DataSet.rules[i], validate_features, validate_labels)
        for j in range(len(DataSet.rules[i]) - 1):
            r = DataSet.rules[i].copy()
            r[j] = None
            current_score = get_score(r, validate_features, validate_labels)
            if current_score > max_score:
                print('Update:' + str(max_score) + ' -> ' + str(current_score))
                max_score = current_score
                DataSet.rules[i][j] = None
        max_score_list.append(max_score)
    idx = np.array(max_score_list).argsort()[::-1]
    temp = []
    # print(idx)
    # for i in len(idx):
    #     temp.append(rules[idx[len(idx) - 1 - i]])
    DataSet.rules = np.array(DataSet.rules)[idx]



def filter_data(rule, validate_features, validate_labels):
    selected = range(len(validate_features))
    for f in rule:
        if f is not None and f.label is None:
            selected = [x for x in selected if f.judge(validate_features[x][f.attribute_index])]
    return selected


# score of a rule
def get_score(rule, validate_features, validate_labels):
    selected = filter_data(rule, validate_features, validate_labels)
    predict = rule[-1].label
    acc_num = 0
    for i in selected:
        if validate_labels[i] == predict:
            acc_num += 1

    if len(selected) == 0:
        return 1
    return acc_num / len(selected)


def fit_rule(feature, rule):
    for f in rule:
        if f is not None and f.label is None:
            if f.judge(feature[f.attribute_index]):
                pass
            else:
                return False
    return True


def predict(features):
    res = []
    for feature_index in range(len(features)):
        for rule in DataSet.rules:
            if fit_rule(features[feature_index], rule):
                res.append(rule[-1].label)
                break
        if len(res) < feature_index + 1:
            res.append(0)
    return res

#%%
## without noise
DataGenerator('data/tic-tac-toe/tic-tac-toe.data', True, 0, 1)
X_train_validate, X_test, y_train_validate, y_test = train_test_split(range(len(DataSet.examples)),
                                                                range(len(DataSet.examples[:, -1])),
                                                                      test_size=0.10)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate,
                                                                y_train_validate,
                                                                      test_size=0.20)

#%%
tree = MyDecisionTree()
# Do not pass through the label attribute
tree.train(X_train, list(range(len(DataSet.attributes) - 1)))
DataSet.rules = []
tree.to_rules([], tree.root)

test = np.array(DataSet.examples[X_test])
test_features = test[:, 0:-1]
test_labels = test[:, -1]

res = predict(test_features)
acc_num = 0
for i in range(len(res)):
    if res[i] == test_labels[i]:
        acc_num += 1
print(acc_num / len(res))

#DataSet.print_rules(DataSet.rules, range(10))
validate = np.array(DataSet.examples[X_validate])
features = validate[:, 0:-1]
labels = validate[:, -1]
prune(features, labels)
# DataSet.print_rules(DataSet.rules, range(50))

res = predict(test_features)

acc_num = 0
for i in range(len(res)):
    if res[i] == test_labels[i]:
        acc_num += 1
print(acc_num / len(res))


#%%
Ls = [0, 5, 10, 15]
average_accuracy = []
variance = []
average_accuracy1 = []
variance1 = []

for l in Ls:

    # Noise 1
    DataGenerator('data/glass/glass.data', False, l, 1)

    accs = []
    for times in range(50):
        X_train_validate, X_test, y_train_validate, y_test = train_test_split(range(len(DataSet.examples)),
                                                                              range(len(DataSet.examples[:, -1])),
                                                                              test_size=0.10)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate,
                                                                    y_train_validate,
                                                                    test_size=0.20)
        test = np.array(DataSet.examples[X_test])
        test_features = test[:, 0:-1]
        test_labels = test[:, -1]
        # clf = tree.DecisionTreeClassifier()
        # features = DataSet.examples[X_train][:, 0:-1]
        # features1= DataSet.examples[X_test][:, 0:-1]
        # clf.fit(features, DataSet.examples[X_train][:, -1])
        # res = clf.predict(features1)

        tree = MyDecisionTree()

        # Do not pass through the label attribute
        tree.train(X_train, list(range(len(DataSet.attributes) - 1)))
        DataSet.rules = []
        tree.to_rules([], tree.root)

        validate = np.array(DataSet.examples[X_validate])
        features = validate[:, 0:-1]
        labels = validate[:, -1]
        prune(features, labels)

        res = predict(test_features)

        expect = test_labels

        acc = 0
        for i in range(len(X_test)):
            if res[i] == expect[i]:
                acc += 1
        accs.append(acc / len(X_test))

    print("Average accuracy:")
    print(sum(accs) / len(accs))
    average_accuracy.append(sum(accs) / len(accs))
    print("Variance of accuracy:")
    variance.append(np.var(accs))
    print(np.var(accs))

    # Noise 2
    DataGenerator('data/glass/glass.data', False, l, 2)
    accs = []
    for times in range(10):
        X_train_validate, X_test, y_train_validate, y_test = train_test_split(range(len(DataSet.examples)),
                                                                              range(len(DataSet.examples[:, -1])),
                                                                              test_size=0.10)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate,
                                                                    y_train_validate,
                                                                    test_size=0.20)
        test = np.array(DataSet.examples[X_test])
        test_features = test[:, 0:-1]
        test_labels = test[:, -1]
        # clf = tree.DecisionTreeClassifier()
        # features = DataSet.examples[X_train][:, 0:-1]
        # features1= DataSet.examples[X_test][:, 0:-1]
        # clf.fit(features, DataSet.examples[X_train][:, -1])
        # res = clf.predict(features1)

        tree = MyDecisionTree()

        # Do not pass through the label attribute
        tree.train(X_train, list(range(len(DataSet.attributes) - 1)))
        DataSet.rules = []
        tree.to_rules([], tree.root)

        validate = np.array(DataSet.examples[X_validate])
        features = validate[:, 0:-1]
        labels = validate[:, -1]
        prune(features, labels)

        res = predict(test_features)

        expect = test_labels

        acc = 0
        for i in range(len(X_test)):
            if res[i] == expect[i]:
                acc += 1
        accs.append(acc / len(X_test))

    print("Average accuracy:")
    print(sum(accs) / len(accs))
    average_accuracy1.append(sum(accs) / len(accs))
    print("Variance of accuracy:")
    variance1.append(np.var(accs))
    print(np.var(accs))

#%%

plt.figure()
res_list = list(Ls)
acc = average_accuracy.copy()
acc1 = average_accuracy1.copy()
# for i in range(len(res_list)):
#     acc[i] += res_list[i] / 100
#     acc1[i] += res_list[i] / 100

plt.plot(Ls, acc, label="average_accuracy_noise1")
plt.plot(Ls, acc1, label="average_accuracy_noise2")
# plt.plot(Ls, variance, label="variance")
plt.title("After pruning: Average accuracy with noise - Glass")
plt.xlabel("noise percentage")
plt.legend()

plt.show()
