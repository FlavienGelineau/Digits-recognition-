zero=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0
,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,
1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1
,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,
1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,
0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
import numpy as np
import random
import matplotlib.pyplot as plt

def ajout_random(chiffre, proba):
	'''proba n'est pas ici une proba mais le x tel que 0 a une chance sur x de se changer en 1
	parce que c'est carrément plus simple et que j'ai la flemme de le coder autrement,
	cordialement '''
	for i in range(len(chiffre)):
		if chiffre[i] ==0 and random.randint(1,proba)==1:
			chiffre[i]=1
	return(chiffre)

def enleve_random(chiffre, proba):
	'''proba n'est pas ici une proba mais le x tel que 1 a une chance sur x de se changer en 0
	parce que c'est carrément plus simple et que j'ai la flemme de le coder autrement,
	cordialement '''
	for i in range(len(chiffre)):
		if chiffre[i] ==1 and random.randint(1,proba)==1:
			chiffre[i]=0
	return(chiffre)	

def lissage(chiffre):
	'''si le 0 est à coté de 5 uns, il devient un 1'''
	result = np.zeros((30,30))
	chiffre_reshaped = list_to_matrice_reshape(chiffre)
	for i in range(1,30):
		for j in range(1,30):
			somme=2*chiffre_reshaped[i][j]+chiffre_reshaped[i-1][j]+chiffre_reshaped[i+1][j]+chiffre_reshaped[i][j+1]+chiffre_reshaped[i][j-1]+chiffre_reshaped[i+1][j+1]+chiffre_reshaped[i+1][j-1]+chiffre_reshaped[i-1][j+1]+chiffre_reshaped[i-1][j-1]  
			somme=somme/10
			result[i][j]=int(np.round(somme))
	return(result)

def list_to_matrice(chiffre):
	matrice=np.zeros((30,30))
	for i in range(30):
		for j in range(30):
			matrice[i][j] = chiffre[i*30 + j]

	return(matrice)

def list_to_matrice_reshape(chiffre):
	matrice=[[0 for i in range(32)]]
	for j in range(30):
		list_prov=[]
		list_prov.append(0)
		for i in range(30):
			list_prov.append(chiffre[j*30+i])
		list_prov.append(0)
		matrice.append(list_prov)
	matrice.append([0 for i in range(32)])
	return(matrice)

def matrice_to_list(matrice):
	return list(np.ravel(matrice))

def test_affiche_valeurs_lissage():
	nouveau_zero=np.array(lissage(zero))
	plt.subplot(1,2,1)
	plt.imshow(list_to_matrice(zero))
	plt.subplot(1,2,2)
	plt.imshow(nouveau_zero)
	plt.show()

def retourne_images(chiffre, axis=0):
	mat=list_to_matrice(chiffre)

	return np.ravel(np.flip(mat, axis=axis))



def shift(chiffre, decalage):
	matrice=list_to_matrice(chiffre)
	new_liste=[]
	for colonne in matrice:
		liste_prov=colonne[decalage:]
		zeros=[0 for i in range(decalage)]
		result=matrice_to_list(liste_prov)+matrice_to_list(zeros)
		new_liste.append(result)
	return(matrice_to_list(new_liste))

def augmenter_taille_train(liste_infos,liste_etiquettes):
	taille_max=len(liste_infos)
	for i in range(taille_max):
		# liste_infos.append(ajout_random(liste_infos[i],10))
		# liste_etiquettes.append(liste_etiquettes[i])

		# liste_infos.append(enleve_random(liste_infos[i],10))
		# liste_etiquettes.append(liste_etiquettes[i])

		# lissee=matrice_to_list(lissage(liste_infos[i]))
		# liste_etiquettes.append(liste_etiquettes[i])
		# lissee_int=[int(x) for x in lissee]
		# liste_infos.append(lissee_int)

		decalee=shift(liste_infos[i],2)
		liste_infos.append(decalee)
		liste_etiquettes.append(liste_etiquettes[i])

		if liste_etiquettes[i]==0:
			liste_infos.append(retourne_images(liste_infos[i], axis=1))
			liste_etiquettes.append(0)

		if liste_etiquettes[i]==6:
			liste_infos.append(retourne_images(liste_infos[i], axis=0))
			liste_etiquettes.append(9)

		if liste_etiquettes[i]==8:
			liste_infos.append(retourne_images(liste_infos[i], axis=1))
			liste_etiquettes.append(8)

		if (i/500)==int(i/500):
			print(i)

liste_infos_test=[zero]
liste_etiquettes=[0]

#plt.subplot(1,2,1)
#plt.imshow(np.array(np.flip(list_to_matrice(zero), axis=1)))
#plt.subplot(1,2,2)
#plt.imshow(np.array(list_to_matrice(retourne_images(zero))))
#plt.show()
augmenter_taille_train(liste_infos_test,liste_etiquettes)

print(liste_infos_test)
