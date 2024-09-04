import numpy as np

##################################################################################
#
# 1.1 Mise en place d’un perceptron simple
#
##################################################################################


##################################################################################
# Perceptron simple
##################################################################################

def perceptron_simple(x, w, active):
    # Calcul du potentiel d'entrée u = w[0] + w[1] * x[0] + w[2] * x[1]
    u = w[0] + w[1] * x[0] + w[2] * x[1]
    
    # Application de la fonction d'activation
    if active == 0:
        y = np.sign(u)
    elif active == 1:
        y = np.tanh(u)
    else:
        raise ValueError("Le paramètre 'active' doit être 0 (sign) ou 1 (tanh)")
    
    return y


##################################################################################
# Test sur le OU logique
##################################################################################

# Combinaisons possibles pour le OU logique
entrées = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
poids = np.array([-0.5, 1, 1])
active = 0

for x in entrées:
    y = perceptron_simple(x, poids, active)
    print(f"Entrée: {x}, Sortie: {y}")


##################################################################################
# Afficher les données d'apprentissage et la droite séparatrice
##################################################################################

import matplotlib.pyplot as plt

# Données pour le OU logique
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 1])  # Sorties attendues pour le OU logique

# Poids et biais initiaux
w = poids

# Droite séparatrice
a = -w[1] / w[2]
b = -w[0] / w[2]

# Affichage de la droite de séparation en bleu
x_vals = np.array([-0.5, 1.5])
y_vals = a * x_vals + b
plt.plot(x_vals, y_vals, 'b-')

# Affichage des points avec des marqueurs différents
for i in range(len(X)):
    if Y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='green', s=100, marker='o')
    else:
        plt.scatter(X[i, 0], X[i, 1], color='green', s=100, marker='s')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron - OU Logique")
plt.grid(True, linestyle='--')
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
plt.show()
