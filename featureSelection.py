#%%
from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import EmpiricalCovariance

# load mnist from openML
# num: numbers of data points
# return:
def load_mnist(num):
    L = num
    mnist = fetch_openml(name='mnist_784')
    # in case num is too large
    assert L < len(mnist['data'])
    data = mnist['data'][0:L]
    label = mnist['target'][0:L]
    return data, label


# data: features
# label:
# return: inter_class_distance / intra_class_distance
def analyse_data(data, label):
    # count the labels
    label_count = {}
    for i in range(10):
        label_count[str(i)] = []
    for i in range(len(label)):
        label_count[label[i]].append(i)

    # average of each class
    average = []
    for i in range(10):
        selected = data[label_count[str(i)]]
        average.append(np.mean(selected, axis=0))

    # inter-class distance
    inter_class_distance = euclidean_distances(average, [np.mean(data, axis=0)])

    return np.mean(inter_class_distance)


# feature_index_list: selected features index
# data: train_data
# label: train data label
# return: measurement of the feature selection
def measure(feature_index_list, data, label):
    selected_feature = data[:, feature_index_list]
    return analyse_data(selected_feature, label)


# data: train data
# label: train data label
# average: mean of each class
# feature: index of all the features
# k: list of feature numbers to select
def bidirection_search(data, label, feature_list, k):
    forward = []
    backward = list(feature_list)
    times = 0
    res = {}
    selected = []

    print('Start:')
    while times < np.max(k):
        print('Times:')
        print(times)
        # start SFS search:
        max_measure = -np.inf
        selected_forward = -1
        cm = 0
        if len(forward) > 0:
            cm = measure(forward, data, label)
        for feature in backward:
            if feature not in forward:
                temp = forward + [feature]
                current_measure = measure(temp, data, label)
                # if current_measure > cm:
                if current_measure > max_measure:
                    max_measure = current_measure
                    selected_forward = feature
                # else:
                #     backward.remove(feature)
        if selected_forward != -1:
            forward = forward + [selected_forward]
            times = times + 1
            selected.append(selected_forward)
        if times in k:
            res[times] = selected.copy()
            k.remove(times)
            if len(k) == 0:
                return res
        print('forward:')
        print(forward)
        # start SBS search:
        max_measure = -np.inf
        selected_backward = -1
        cm = measure(backward, data, label)
        for feature in backward:
            # print(feature)
            if feature not in forward:
                # print(feature)
                temp = backward.copy()
                temp.remove(feature)
                current_measure = measure(temp, data, label)
                if current_measure > max_measure:
                    max_measure = current_measure
                    selected_backward = feature
        if selected_backward != -1:
            backward.remove(selected_backward)
        if len(backward) in k:
            res[len(backward)] = backward.copy()
            k.remove(len(backward))
            if len(k) == 0:
                return res
#         print('backward:')
#         print(backward)
    return res


def res2array(feature_index):
    pic = np.full((28, 28), 255)
    for i in feature_index:
        x = i % 28
        y = i // 28
        pic[y, x] = 1
    return pic


# My KNN implement
class KNNClassifier:
    def __init__(self, train_data, train_label, n_neighbors=3):
        self.train_data = train_data
        self.train_label = train_label
        self.n_neighbors = n_neighbors

    def fit(self, test_data):
        labels = []
        for x in test_data:
            # print(x)
            distances = euclidean_distances(self.train_data, [x]).squeeze()
            max_num_index_list = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            # max_num_index_list = [np.argmin(distances)]
            # print(max_num_index_list)
            # print(distances[max_num_index_list[0]])
            # print(list(max_num_index_list))
            res = []
            for i in max_num_index_list:
                res.append(self.train_label[i])
            labels.append(np.bincount(res).argmax())
        return labels


def get_scatter(features, label):
    # count the labels
    label_count = {}
    for i in range(10):
        label_count[str(i)] = []
    for i in range(len(label)):
        label_count[label[i]].append(i)

    # within class scatter
    within_class_scatter = np.zeros((784, 784))
    average = []
    for i in range(10):
        selected = features[label_count[str(i)]]
        average.append(np.mean(selected, axis=0))
        # within_class_scatter.append(np.mean(selected))
        # for index in range(len(selected)):
        #     x = selected[index] - average[i]
        #     within_class_scatter = within_class_scatter + np.dot(x[None, :], x[:, None])
        # cov = EmpiricalCovariance().fit(selected)
        within_class_scatter = within_class_scatter + np.cov(selected.T).T
    #cov = EmpiricalCovariance().fit(features)
    #total_cov = cov.covariance_


    # between class scatter
    all_mean = np.mean(features, axis=0)
    between_class_scatter = np.zeros((784, 784))
    for i in range(10):
        x = average[i] - all_mean
        between_class_scatter = between_class_scatter + len(label_count[str(i)]) * np.dot(x[:, None], x[None, :])

    return within_class_scatter, between_class_scatter


#%% prepare data
    data1, label1 = load_mnist(10000)
    k_list = [5, 50, 150, 392]
    # # split dataset
    X_train, X_test, y_train, y_test = train_test_split(data1, label1, test_size=0.20)
#%% 3.1 3.2
    selected = bidirection_search(X_train, y_train, range(784), k_list.copy())
    acc = []
    for k in k_list:
        features = X_train[:, selected[k]]
        knn = KNNClassifier(np.array(features), np.array(y_train, dtype=int), 3)
        r = knn.fit(np.array(X_test[:, selected[k]]))
        predict = np.array(y_test, dtype=int)
        acc.append((r == predict).sum() / len(predict))

    print(selected)
    plt.figure()
    plt.plot(k_list, acc, label="KNN Classifier Accuracy (k = 3)")
    for a, b in zip(k_list, acc):
        plt.text(a, b, str(b))
    # plt.plot(Ls, variance, label="variance")
    plt.title("Feature Selection")
    plt.xlabel("Number of features")
    plt.legend()
    plt.show()

    plt.figure()
    fig, axs = plt.subplots(2, 2)

    c = axs[0, 0].pcolor(res2array(selected[5]))
    axs[0, 0].set_title('5 features')
    axs[0, 0].set_aspect('equal', 'box')

    c = axs[0, 1].pcolor(res2array(selected[50]))
    axs[0, 1].set_title('50 features')
    axs[0, 1].set_aspect('equal', 'box')

    c = axs[1, 0].pcolor(res2array(selected[150]))
    axs[1, 0].set_title('150 features')
    axs[1, 0].set_aspect('equal', 'box')

    c = axs[1, 1].pcolor(res2array(selected[392]))
    axs[1, 1].set_title('392 features')
    axs[1, 1].set_aspect('equal', 'box')
    #
    # c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
    # ax1.set_title('thick edges')

    fig.tight_layout()
    plt.show()
#%% 3.3
    res = []
    for index in range(9):
        within, between = get_scatter(X_train, y_train)
        inv = np.linalg.pinv(within)
        wb = np.dot(inv, between)
        w, v = np.linalg.eig(wb)
        idx = w.argsort()[::-1]
        eigenValues = w[idx]
        eigenVectors = v[:, idx]
        lda_selected_trans = eigenVectors[:, 0:index+1].T
        transferred = np.dot(lda_selected_trans, X_test.T)
        transferred_test = transferred.T
        transferred_train = np.dot(lda_selected_trans, X_train.T).T
        features = transferred_train
        knn = KNNClassifier(np.array(features, dtype=float), np.array(y_train, dtype=int), 3)
        r = knn.fit(np.array(transferred_test, dtype=float))
        predict = np.array(y_test, dtype=int)
        res.append((r == predict).sum() / len(predict))
    plt.figure()
    plt.plot(range(1,10), res)
    for a, b in zip(range(1,10), res):
        plt.text(a, b, str(b))
    plt.title('LDA Accuracy:')
    plt.xlabel('Feature Num')
    plt.show()

#%% test
    lda = LinearDiscriminantAnalysis(n_components=5)
    X = lda.fit(X_train, y_train).transform(X_train)
    Y = lda.transform(X_test)
    knn = KNNClassifier(np.array(X, dtype=float), np.array(y_train, dtype=int), 3)
    r = knn.fit(np.array(Y, dtype=float))
    predict = np.array(y_test, dtype=int)
    (r == predict).sum() / len(predict)
