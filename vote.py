from sklearn import datasets

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import clone
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
from sklearn import neighbors

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

import pandas as pd
import csv
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import numpy as np

def parsing(nom_doc_texte):
	result=[]
	with open(nom_doc_texte, newline='') as csvfile:
		csvfile.readline()
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			integer_format=[int(x) for x in row[1:]]
			result.append([int(row[0]),integer_format])

		return(result)

def entree_sortie(liste):
	entree,sortie=[],[]
	for elt in liste:
		sortie.append(elt[0])
		entree.append(elt[1])
	return(entree,sortie)

infos=parsing("train.csv")
infos_test=parsing("test.csv")

X,y=entree_sortie(infos)
X_test,y_test=entree_sortie(infos_test)

def naiveBayes(X,y,X_test):

	gnb = GaussianNB()
	y_pred = gnb.fit(X, y).predict(X_test)

	return(y_pred)


def randomForests(X,y,X_test):
	predictions=[]
	n_estimators=200
	models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=None),
           n_estimators=n_estimators)]
	
	for model in models:
		clf = clone(model)
		clf = model.fit(X, y)
		prediction = clf.predict(X_test)

		predictions.append(prediction)
	return(predictions)


def svm_prediction(X,y,X_test):

	clf = svm.SVC(verbose=False)
	clf.fit(X, y)
	return(clf.predict(X_test).tolist())


def neirest_nei(X,y,X_test):

	k_neight = 20
	nn = neighbors.KNeighborsClassifier(k_neight)
	nn.fit(X, y)
	return( nn.predict(X_test) )

def convutional_neural_network(X,y,X_test):
	dataset = pd.read_csv("train.csv")
	Y_train = dataset.as_matrix()[:,0]
	X_train = dataset.as_matrix()[:,1:]

	Y_train = label_binarize(Y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

	Y_test = dataset.as_matrix()[:,0]
	X_test = dataset.as_matrix()[:,1:]

	Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

	model = Sequential()
	# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(30, 30, 1)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	model.add(Dropout(0.3))

	model.add(Flatten())

	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(10))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=adam)

	model.fit(X_train.reshape((len(X_train),30,30,1)), Y_train, batch_size=40, nb_epoch=5, verbose = True)

	Y_pred = model.predict_classes(X_test.reshape((len(X_test),30,30,1)))
	return(Y_pred)

def vote(liste_de_listes):
	'''differentes_predictions doit être de la forme [taille_de_y_test]'''
	values=[[0 for i in range(10)] for i in range(len(liste_de_listes[0])) ]

	#print("examinons les tailles des différentes predictions de chacun des algos",len(differentes_predictions))
	for differentes_predictions in liste_de_listes:
		print("differentes_predictions",differentes_predictions)
		for i in range(len(differentes_predictions)):
			values[i][differentes_predictions[i]]+=1
			#print(values)
	return(values)

def vote_v2(liste_de_listes):
	'''maintenant chaque case de liste_de_listes corespond au vote du processus de l'index.
	Par ex, si le deuxième algo a voté 8, on aura [x,8,y, etc]'''
	values=[[0 for i in range(len(liste_de_listes))] for i in range(len(liste_de_listes[0])) ]
	for j in range(len(liste_de_listes)):	#j va de 0 à 7

		for i in range(len(liste_de_listes[j])):
			values[i][j]=liste_de_listes[j][i]
	return(values)

def learning_sur_le_vote(liste_de_listes):
	liste_totale_testee=construit_liste_totale(X_test,y_test,X,y)
	predictions=[]
	n_estimators=200
	votes=vote_v2(liste_totale_testee)
	clf = clone(ExtraTreesClassifier(n_estimators=n_estimators))
	clf = ExtraTreesClassifier(n_estimators=n_estimators).fit(votes, y_test)
	prediction = clf.predict(vote_v2(liste_de_listes))

	return(predictions)

def return_indice_max(liste_de_listes):

	'''L'input est une liste de listes, chacune des listes de taille 10 comportant les votes des différentes méthodes de learning
	elle est de format 8*[taille_X_test]
	La sortie est de format [taille_X_test]'''
	result=[]
	for liste in liste_de_listes:
		result.append(liste.index(max(liste)))
	return(result)


def construit_liste_totale(X_test,y_test,X,y):

	result=[]
	result.append(naiveBayes(X,y,X_test))

	randomForestResults=randomForests(X,y,X_test)
	result=result+randomForestResults

	result.append(np.array(svm_prediction(X,y,X_test)))

	result.append(neirest_nei(X,y,X_test))

	result.append(convutional_neural_network(X,y,X_test))

	return(result)

toutes_preds=construit_liste_totale(X_test,y_test,X,y)

print(vote_v2(toutes_preds))
prediction_finale=return_indice_max(vote(toutes_preds))

print("taille pred finale :", len(prediction_finale))
print("taille y_test :", len(y_test))

def test_score(liste1,liste2):
	if len(liste2)!=len(liste1):
		print("les deux listes sont pas de la même taille ! ")
		return(0)
	else:
		score=0
		for i in range(len(liste1)):
			if liste1[i]==liste2[i]:
				score+=1
		return(score/len(liste1))

print("accuracy finale",test_score(prediction_finale,y_test))
import gc; gc.collect()