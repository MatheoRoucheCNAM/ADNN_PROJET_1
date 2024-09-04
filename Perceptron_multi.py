import numpy as np

##################################################################################
#
# 1.3 Mise en place d’un perceptron multi-couche
#
##################################################################################


# Fonction Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

##################################################################################
# Perceptron multi
##################################################################################

def multiperceptron(x, w1, w2):
    # Ajout du biais à l'entrée
    x = np.append(1, x) 
    print(x)
    print(w1)

    # Calcul des activations des neurones de la couche cachée
    u1 = np.dot(x, w1)
    print(u1)
    z1 = sigmoid(u1)
    print(z1)

    # Ajout du biais pour la couche de sortie
    z1 = np.append(1, z1)
    print(z1)

    # Calcul de la sortie du neurone de la couche de sortie
    u2 = np.dot(z1, w2)
    print(u2)
    y = sigmoid(u2)
    print(y)

    return y


##################################################################################
# Test avec une entrée [1, 1]
##################################################################################

x = np.array([1, 1])
w1 = np.array([[-0.5, 0.5], [2, 1], [-1, 0.5]])
w2 = np.array([2, -1, 1])

# Calcul de la sortie
y = multiperceptron(x, w1, w2)
print("La sortie du perceptron multicouche est:", y)
# Sortie: La sortie du perceptron multicouche est: 0.9053673095402572