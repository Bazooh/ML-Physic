# Conception et évaluation de méthodes ML en physique

Pierre JOURDIN
Aymeric CONTI

## Introduction
On a choisi le problème de poisson :  

-Δu(x, y) = f(x, y),  pour (x, y) ∈ Ω
u(x, y) = 0,           pour (x, y) ∈ ∂Ω

---

## 1. Simulation par différences finies

### 1.1 Formulation du problème

On considère le problème de résolution de l'équation de Poisson sur un domaine carré $[0, 1]^2$ :

$$\Delta u(x, y) = f(x, y), \quad \text{pour } (x, y) \in (0, 1)^2$$

avec les conditions aux limites de Dirichlet homogènes :

$$u(x, y) = 0, \quad \text{pour } (x, y) \in \partial([0, 1]^2)$$

On discrétise le domaine à l’aide d’une grille uniforme de taille 64 × 64, soit N = 64 points dans chaque direction. L’espacement entre les points est $(h = \frac{1}{N+1})$.

### 1.2 Implémentation

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

## 2. Résolution par PINN (Physics-Informed Neural Network)

### 2.1 Principe de la méthode
- Structure du réseau (MLP, activation, etc.)
- Entraînement avec résidu PDE + conditions aux limites

### 2.2 Implémentation
- Choix des points (collocation)
- Détail de la fonction de perte
- Optimisation

### 2.3 Résultats
- Comparaison à la solution FD
- Discussion (qualité, temps d’entraînement, sensibilité)

---

## 3. Résolution par PENN (Physical-Encoded Neural Network)

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
