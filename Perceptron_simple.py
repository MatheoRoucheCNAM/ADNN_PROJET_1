import numpy as np

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
poids = np.array([-1.5, 1, 1])
active = 0

for x in entrées:
    y = perceptron_simple(x, poids, active)
    print(f"Entrée: {x}, Sortie: {y}")