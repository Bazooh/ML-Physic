# Conception et évaluation de méthodes ML en physique

Pierre JOURDIN
Aymeric CONTI

## Introduction

On a choisi de considerer le problème de résolution de l'équation de Poisson sur un domaine carré $[0, 1]^2$ :

$$\Delta u(x, y) = -f(x, y), \quad \text{pour } (x, y) \in [0, 1]^2$$

avec les conditions aux limites de Dirichlet homogènes :

$$u(x, y) = 0, \quad \text{pour } (x, y) \in \partial([0, 1]^2)$$

et avec la fonction f définie par :

$$f(x, y) = x sin (a \pi y) + y sin (b \pi x)$$

a et b sont les paramètres a faire varier pour obtenir les différents problèmes de cette famille.

## 1. Simulation par différences finies

On discrétise le domaine à l’aide d’une grille uniforme de taille 64 × 64, soit N = 64 points dans chaque direction. L’espacement entre les points est $(h = \frac{1}{N+1})$.

Nous utilisons la méthode des différences finies centrées pour approximer le Laplacien. Cela donne, pour un point intérieur (i,j) :

$$\Delta u_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$$

Cette approximation conduit à un système linéaire de la forme :

$$A \mathbf{u} = \mathbf{f}$$

où A est une matrice creuse de taille $(N^2 × N^2)$ avec :
- 4 sur la diagonale principale,
- -1 sur les coefficients correspondant aux voisins (haut, bas, gauche, droite), en tenant compte de la structure 2D de la grille.

Voici un exemple de structure pour A (dans le cas N=3 pour la clarté) :

$$
A = \begin{pmatrix}
 4 & -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 \\
-1 &  4 & -1 &  0 &  0 &  0 &  0 &  0 &  0 \\
 0 & -1 &  4 & -1 &  0 &  0 &  0 &  0 &  0 \\
 0 &  0 & -1 &  4 & -1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 & -1 &  4 & -1 &  0 &  0 &  0 \\
 0 &  0 &  0 &  0 & -1 &  4 & -1 &  0 &  0 \\
 0 &  0 &  0 &  0 &  0 & -1 &  4 & -1 &  0 \\
 0 &  0 &  0 &  0 &  0 &  0 & -1 &  4 & -1 \\
 0 &  0 &  0 &  0 &  0 &  0 &  0 & -1 &  4
\end{pmatrix}
$$

La matrice A est construite en utilisant la fonction `kronsum()` de `scipy.sparse`, qui réalise le produit de Kronecker entre deux matrices tridiagonales.

Le système linéaire est résolu grâce à la fonction `spsolve()` de la bibliothèque `scipy.sparse.linalg`, qui est adaptée au traitement de matrices creuses.


---

## 2. Résolution par Réseau de Neurones

Fait office de baseline ML

### 2.1 Principe de la méthode

- Les données d'entraînement et de test sont générées à l'aide de la méthode des différences finies.
- Chaque donnée d'entraînement est constituée de :
  - **Input** : une grille de \( f(x, y) \), calculée via la formule donnée.
  - **Label** : une grille de \( u(x, y) \), solution obtenue par différences finies.
- Un réseau de neurones est entraîné pour approximer la solution \( u \) à partir du champ \( f \).

**Avantages :**
- facile à implémenter
- Prédiction rapide une fois entraîné.

**Inconvénients :**
- Aucune contrainte physique imposée.
- Approxime le résultat donné par les différence finies, qui est déjà une approximation de la réalité

### 2.2 Implémentation

- **Architecture** : réseau de neurones convolutionnel suivi d'une couche dense.
  - 5 couches convolutives avec padding (pas de perte de dimension) et activations ReLU.
  - Une couche dense linéaire en sortie.

- **Fonction de perte** : MSE (Mean Squared Error), classique pour un problème de régression.

---

## 3. Résolution par PINN (Physics-Informed Neural Network)

### 3.1 Principe de la méthode

L’objectif est d’incorporer explicitement la physique du problème dans l’apprentissage du réseau, en forçant le respect de l’équation différentielle.

La structure du réseau est similaire à celle utilisée en apprentissage supervisé, mais la **fonction de perte** inclut maintenant plusieurs composantes :

- **Terme "data"** : MSE entre la sortie du réseau et la solution de référence (comme dans la méthode classique).
- **Terme "résidu EDP"** : calcul des dérivées secondes de la solution prédite, puis MSE entre \(-\Delta u_\theta(x, y)\) et \(f(x, y)\).
- **Terme "conditions aux limites"** : MSE entre la solution du réseau et 0 sur le bord du domaine.

**Avantages :**
- Intégration directe des connaissances physiques.
- Prédiction rapide après entraînement.
- Reste relativement simple à implémenter

**Inconvénients :**
- Approxime le résultat donné par les différence finies, qui est déjà une approximation de la réalité

---

### 3.2 Implémentation

Le terme de conditions aux limites a été supprimé après expérimentation, car il dégradait les performances dans notre cas.
Une étude plus approfondie aurait pu être pertinente, mais nous avons préféré nous focaliser sur la partie concernant le résidu d'EDP dans ce travail.

**Pondérations choisies :**
- Résidu PDE : `1e-7`
- Terme "data" : `1 - 1e-7`
- Terme CL : `0`

---

## 4. Résolution par PENN (Physical-Encoded Neural Network)

### 3.1 Principe de la méthode
- Modification de l’architecture pour imposer les CL "hard"
- Perte basée sur l’énergie potentielle (formulation variationnelle)
- Intégration de la physique dans le réseau

### 3.2 Implémentation
- Structure du réseau
- Formule de la loss (énergie)
- Contraintes physiques encodées

### 3.3 Résultats
- Qualité de la solution
- Comparaison avec PINN et FD
- Avantages/inconvénients

---

## Conclusion
- Résumé des observations
- Avantages et limites de chaque méthode
- Perspectives
