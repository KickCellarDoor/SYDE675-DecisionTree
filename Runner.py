import pandas as pd
from DataGenerator import DataGenerator
from MyDecisionTree import MyDecisionTree
from DataGenerator import DataSet
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

if __name__ == '__main__':
    Ls = range(2)
    average_accuracy = []
    variance = []
    average_accuracy1 = []
    variance1 = []

    for l in Ls:

        # Noise 1
        DataGenerator('data/glass/glass.data', False, l, 1)

        accs = []
        for times in range(10):
            X_train, X_test, y_train, y_test = train_test_split(range(len(DataSet.examples)),
                                                                range(len(DataSet.examples[:, -1])),
                                                                test_size=0.20)
            # clf = tree.DecisionTreeClassifier()
            # features = DataSet.examples[X_train][:, 0:-1]
            # features1= DataSet.examples[X_test][:, 0:-1]
            # clf.fit(features, DataSet.examples[X_train][:, -1])
            # res = clf.predict(features1)

            tree = MyDecisionTree()

            # Do not pass through the label attribute
            tree.train(X_train, list(range(len(DataSet.attributes) - 1)))

            res = []
            for i in X_test:
                res.append(tree.predict(DataSet.examples[i]))

            expect = DataSet.examples[:, -1].tolist()

            acc = 0
            for i in range(len(X_test)):
                if res[i] == expect[X_test[i]]:
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
            X_train, X_test, y_train, y_test = train_test_split(range(len(DataSet.examples)),
                                                                range(len(DataSet.examples[:, -1])),
                                                                test_size=0.20)
            # clf = tree.DecisionTreeClassifier()
            # features = DataSet.examples[X_train][:, 0:-1]
            # features1= DataSet.examples[X_test][:, 0:-1]
            # clf.fit(features, DataSet.examples[X_train][:, -1])
            # res = clf.predict(features1)

            tree = MyDecisionTree()

            # Do not pass through the label attribute
            tree.train(X_train, list(range(len(DataSet.attributes) - 1)))

            res = []
            for i in X_test:
                res.append(tree.predict(DataSet.examples[i]))

            expect = DataSet.examples[:, -1].tolist()

            acc = 0
            for i in range(len(X_test)):
                if res[i] == expect[X_test[i]]:
                    acc += 1
            accs.append(acc / len(X_test))

        print("Average accuracy:")
        print(sum(accs) / len(accs))
        average_accuracy1.append(sum(accs) / len(accs))
        print("Variance of accuracy:")
        variance1.append(np.var(accs))
        print(np.var(accs))

    plt.figure()
    plt.plot(Ls, average_accuracy, label="average_accuracy_noise1")
    plt.plot(Ls, average_accuracy1, label="average_accuracy_noise2")
    # plt.plot(Ls, variance, label="variance")
    plt.title("Average accuracy with noise - glass")
    plt.xlabel("noise percentage")
    plt.legend()


    plt.show()
    # tree.print()
