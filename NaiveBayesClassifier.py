from sklearn import datasets

from sklearn.naive_bayes import GaussianNB
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
X,y=entree_sortie(infos)
X_test,y_test=entree_sortie(infos_test)

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X_test)

erreurs=(y_test != y_pred).sum()
print("accuracy :" + str(1-erreurs/len(X_test)))
