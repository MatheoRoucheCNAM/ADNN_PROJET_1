import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#
# 1.2 Apprentissage Widrow-Hoff
#
##################################################################################


##################################################################################
# Fonction Apprentissage Widrow
##################################################################################

def apprentissage_widrow(x, yd, epoch, batch_size, learning_rate=0.1):

    n_features, n_samples = x.shape

    # Initialisation aléatoire des poids
    w = np.random.rand(n_features + 1) - 0.5
    erreurs = []

    # Ajouter une colonne de biais (1) à x
    x = np.vstack([np.ones(n_samples), x])

    # Transposer x pour que chaque colonne soit un exemple
    x = x.T
    print(x)
    
    for ep in range(epoch):
        erreur_epoch = 0
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch = x[batch_indices]
            yd_batch = yd[batch_indices]

            # Calcul de la sortie prédite
            y = np.dot(x_batch, w)

            # Calcul de l'erreur
            erreur = yd_batch - y
            erreur_epoch += np.sum(erreur ** 2)

            # Mise à jour des poids avec le taux d'apprentissage
            gradient = np.dot(erreur, x_batch)
            w += (learning_rate / batch_size) * gradient

        erreurs.append(erreur_epoch / n_samples)

        # Afficher l'évolution de l'erreur
        print(f'Époque {ep+1}/{epoch}, Erreur: {erreurs[-1]}')

        # Arrêter si l'erreur est suffisamment petite
        if erreur_epoch == 0:
            break
    
    print(f'Paramètres trouvés: biais = {w[0]}, w1 = {w[1]}, w2 = {w[2]}')

    return w, erreurs


##################################################################################
# Test 1
##################################################################################

# Charger les données depuis le fichier p2_d1.txt
data = np.loadtxt('C:/Users/mathe/Save_OneDrive_CNAM/CNAM_3/transfert/p2_d1.txt')

# Les deux lignes représentent X1 et X2, et les 50 colonnes représentent les individus
x1 = data[0, :]
x2 = data[1, :]

# Créer la matrice x (2 lignes, 50 colonnes)
x = np.vstack((x1, x2))

# Créer le vecteur yd (1 ligne, 50 colonnes)
yd = np.hstack((np.ones(25), -np.ones(25)))

# Paramètres d'apprentissage
epoch = 150
batch_size = 10

# Entraîner le perceptron avec les données chargées
w, erreurs = apprentissage_widrow(x, yd, epoch, batch_size)

# Afficher l'évolution de l'erreur
plt.figure()
plt.plot(erreurs)
plt.xlabel('Époques')
plt.ylabel('Erreur quadratique')
plt.title('Évolution de l\'erreur pendant l\'apprentissage')
plt.grid(True)
plt.show()

# Vérifier la frontière de décision
a = -w[1] / w[2]
b = -w[0] / w[2]

# Définir les points pour tracer la droite de décision
x_min, x_max = x1.min() - 1, x1.max() + 1
x_vals = np.array([x_min, x_max])
y_vals = a * x_vals + b

# Tracer la distribution des points avec la frontière de décision
plt.figure()

# Remplir l'espace en bleu clair pour la classe 2 (yd = -1)
plt.fill_between(x_vals, y_vals, y2=x2.max() + 1, color='lightblue', alpha=0.5)

# Remplir l'espace en rouge clair pour la classe 1 (yd = 1)
plt.fill_between(x_vals, y_vals, y2=x2.min() - 1, color='lightcoral', alpha=0.5)

# Tracer les points de données
plt.scatter(x1[yd == 1], x2[yd == 1], color='red', label='Classe 1')   # Points en rouge pour la classe 1
plt.scatter(x1[yd == -1], x2[yd == -1], color='blue', label='Classe 2') # Points en bleu pour la classe 2

# Tracer la frontière de décision
plt.plot(x_vals, y_vals, 'g-')

# Configurer le graphique
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Distribution des points avec frontière de décision')
plt.legend()
plt.grid(True)
plt.xlim([x_min, x_max])
plt.ylim([x2.min() - 1, x2.max() + 1])
plt.show()