import numpy as np
import scipy as scp

from sklearn import svm
import math 
import random
import pandas as pd
import csv
import create_other_numbers

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def parsing(nom_doc_texte):
	result=[]
	with open(nom_doc_texte, newline='') as csvfile:
		csvfile.readline()
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			integer_format=[int(x) for x in row[1:]]
			result.append([int(row[0]),integer_format])

		return(result)

infos=parsing("train.csv")
infos_test=parsing("test.csv")

def entree_sortie(liste):
	entree,sortie=[],[]
	for elt in liste:
		sortie.append(elt[0])
		entree.append(elt[1])
	return(entree,sortie)

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


def main():
	entree,sortie=entree_sortie(infos)

	entree_test,sortie_test=entree_sortie(infos_test)
	clf = svm.SVC(verbose=False)
	clf.fit(entree, sortie)
	print(test_score(clf.predict(entree_test).tolist(),sortie_test))

	create_other_numbers.augmenter_taille_train(entree,sortie)
	clf = svm.SVC(verbose=False)
	clf.fit(entree, sortie)
	print(test_score(clf.predict(entree_test).tolist(),sortie_test))
main()