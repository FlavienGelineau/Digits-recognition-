from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import clone
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

import csv
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


def main():
	n_estimators=200
	models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=None),
           n_estimators=n_estimators)]

	X,y=entree_sortie(infos)
	X_test,y_test=entree_sortie(infos_test)
	
	for model in models:
		clf = clone(model)
		clf = model.fit(X, y)
		scores = clf.score(X_test, y_test)

		model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
		model_details = model_title
		if hasattr(model, "estimators_"):
			model_details += " with {} estimators".format(len(model.estimators_))
			print( model_details + "has a score of", scores )


main()