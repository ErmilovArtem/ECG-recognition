from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pyedflib as edf

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

test_files = [
    r'C:\Users\march\PycharmProjects\end\chb24_04.edf',
]

test_file = test_files[0]
file = edf.EdfReader('C:\\Users\\march\\Downloads\\Patterns chb24_03.edf')

ab = [[], []]
read = file.read_annotation()
buff = 0
i = 0
while i < len(read) - 1:
    i += 1
    if (str(read[i][-1])[2:4]) == 'SW':
        while (str(read[i][-1])[2:4]) == 'SW':
            i += 1
            ab[1].append([i - 1, i])
            if i >= len(read) - 3:
                break
        i += 1
    else:
        ab[0].append([i - 1, i])

ab[0].pop(-1)
ab[1].pop(-1)

n = file.signals_in_file
signals = []
for i in range(n):
    signals.append(file.readSignal(i))

for i in range(len(ab[1])):
    for j in range(len(ab[1][1])):
        ab[1][i][j] = read[ab[1][i][j]][0]

for i in range(len(ab[0])):
    for j in range(len(ab[0][1])):
        ab[0][i][j] = read[ab[0][i][j]][0]

checker = False
if checker:
    i = 0
    while i < 28880000000:
        i += 5000000
        if not (2355000000 < i < 2471500000):
            ab[0].append([i, i + 5000000])

    buffered_mean = [[]]
    buffered_maxi = [[]]
    buffered_maximum = [[]]
    buffered_minimum = [[]]
    buffered_amplitude = [[]]
    buffered_ciricPow = [[0]]
    a = ab[1]
    mult = 256 / 100000000
    for arg in ab:
        for elem in arg:
            if (elem[1] - elem[0] != 0):
                for sign in signals:
                    buff_fo_maxi = 0
                    buff_fo_mean = np.mean(np.array(sign)[int(elem[0] * mult): int(elem[1] * mult)])
                    buffered_mean[-1].append(buff_fo_mean * 10)
                    buffered_maximum[-1].append(
                        max(np.array(sign)[int(elem[0] * mult): int(elem[1] * mult)]))
                    buffered_minimum[-1].append(
                        min(np.array(sign)[int(elem[0] * mult): int(elem[1] * mult)]) * 10)
                    buffered_amplitude[-1].append(
                        abs(max(np.array(sign)[int(elem[0] * mult): int(elem[1] * mult)]) - min(
                            np.array(sign)[int(elem[0] * mult): int(elem[1] * mult)])) * 10)
                    for i in range(int(elem[0] * mult), int(elem[1] * mult)):
                        buff_fo_maxi += abs(buff_fo_mean - sign[i])
                    buffered_maxi[-1].append(buff_fo_maxi / (elem[1] * 256 - elem[0] * 256) * 1000000000)
                print(buffered_mean)
                buffered_mean.append([])
                buffered_maxi.append([])
                buffered_maximum.append([])
                buffered_minimum.append([])
                buffered_amplitude.append([])
                buffered_ciricPow.append([])

    for i in range(len(buffered_maximum[0])):
        for j in range(len(buffered_maximum) - 1):
            if j == 0:
                buffered_ciricPow[j].append(
                    pow(abs(buffered_maximum[j + 1][i] - buffered_maximum[j][i]) + 0.00001, -1) * 10 / 2)
                continue
            buffered_ciricPow[j].append(
                pow(abs(buffered_maximum[j][i] - buffered_maximum[j - 1][i]) + 0.00001, -1) * 10 * abs(
                    buffered_maximum[j][i]))
    buffered_ciricPow[0][0] = buffered_ciricPow[0][1] / 2

    buffered_mean.pop(-1)
    buffered_maxi.pop(-1)
    buffered_maximum.pop(-1)
    buffered_minimum.pop(-1)
    buffered_amplitude.pop(-1)
    buffered_ciricPow.pop(-1)

    all = []
    all.append(buffered_mean)
    all.append(buffered_maxi)
    all.append(buffered_maximum)
    all.append(buffered_minimum)
    all.append(buffered_amplitude)
    all.append(buffered_ciricPow)
    with open('listfile.data', 'wb') as filehandle:
        # сохраняем данные как двоичный поток
        pickle.dump(all, filehandle)

with open('listfile.data', 'rb') as filehandle:
    # сохраняем данные как двоичный поток
    all = pickle.load(filehandle)

with open('listfile6.data', 'rb') as filehandle:
    # сохраняем данные как двоичный поток
    allle = pickle.load(filehandle)

for elem in allle:
    all.append(elem)

XY = [[]]
Y = []

for j in range(len(all[0])):
    for i in range(len(all)):
        for k in range(len(all[0][0])):
            XY[-1].append(float(all[i][j][k]))
    if j < len(ab[1]):
        Y.append(1)
    else:
        Y.append(0)
    XY.append([])

XM = [[]]
for j in range(len(all[0])):
    for i in range(len(all)):
        mean_k = 0
        for k in range(len(all[0][0])):
            mean_k += float(all[i][j][k])
        XM[-1].append(float(mean_k / len(all[0][0])))
    XM.append([])

print(len(all), len(all[1]), len(all[0][0]))

XY.pop()
XM.pop()

print(XM)

SVM = svm.LinearSVC()
print(SVM.fit(np.array(XY), np.array(Y)))
print(SVM.predict(np.array(XY))[0:len(ab[1])])
print(round(SVM.score(XY, Y), 4))

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print(RF.fit(XY, Y))
print(RF.predict(XY)[0:len(ab[1])])
print(round(RF.score(XY, Y), 4))

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
print(NN.fit(XY, Y))
print(NN.predict(XY)[0:len(ab[1])])
print(round(NN.score(XY, Y), 4))

print(Y[0:len(ab[1])])

SVM = svm.LinearSVC()
print(SVM.fit(np.array(XM), np.array(Y)))
print(SVM.predict(np.array(XM))[0:len(ab[1])])
print(round(SVM.score(XM, Y), 4))

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print(RF.fit(XM, Y))
print(RF.predict(XM)[0:len(ab[1])])
print(round(RF.score(XM, Y), 4))

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
print(NN.fit(XM, Y))
print(NN.predict(XM)[0:len(ab[1])])
print(round(NN.score(XM, Y), 4))

print(Y[0:len(ab[1])])


print("!!!TRAINING ON SMTH!!!")


RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print(RF.fit(XY[0:len(ab[1]) * 2], Y[0:len(ab[1]) * 2]))
print(RF.predict(XY)[0:len(ab[1])])
print(round(RF.score(XY, Y), 4))

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
print(NN.fit(XY[0:len(ab[1]) * 2], Y[0:len(ab[1]) * 2]))
print(NN.predict(XY)[0:len(ab[1])])
print(round(NN.score(XY, Y), 4))

print(Y[0:len(ab[1])])

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print(RF.fit(XM[0:len(ab[1]) * 2], Y[0:len(ab[1]) * 2]))
print(RF.predict(XM)[0:len(ab[1])])
print(round(RF.score(XM, Y), 4))

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
print(NN.fit(XM[0:len(ab[1]) * 2], Y[0:len(ab[1]) * 2]))
print(NN.predict(XM)[0:len(ab[1])])
print(round(NN.score(XM, Y), 4))

print(Y[0:len(ab[1])])


