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

---

## 4. Résultats

### Comparaison des temps de calcul

| Méthode                               | Temps de calcul |
|---------------------------------------|-----------------|
| Résolution physique                   | 8.33 it/s       |
| Réseau de neurones (batch_size = 1)   | 916.87 it/s     |
| Réseau de neurones (batch_size = 64)  | 3072.00 it/s    |

Le temps total de préparation pour l'utilisation du réseau de neurones est de 204 secondes (dont 119 secondes pour la génération du dataset et 85 secondes pour l'entraînement).

Il devient avantageux d'utiliser le modèle appris lorsque l’on doit générer plus de données que le seuil d’intersection des coûts :

$$204 + \frac{x}{3072} = \frac{x}{8.33} \quad \Rightarrow \quad x \approx 1704$$

Autrement dit, à partir de 1704 prédictions, l'approche par réseau devient plus rapide que la résolution physique directe.

### Comparaison des erreurs

| Méthode                   | Loss                   |
|---------------------------|------------------------|
| NN                        | $7.2561 \cdot 10^{-5}$ |
| PINN $(\gamma = 10^{-6})$ | $7.2542 \cdot 10^{-5}$ |
| PINN $(\gamma = 10^{-7})$ | $7.2579 \cdot 10^{-5}$ |
| PENN                      | $6.9001 \cdot 10^{-5}$ |

On observe que les PINNs obtiennent des performances comparables au réseau classique pour des hyperparamètres appropriés. Le modèle PENN présente une erreur légèrement plus faible, indiquant un meilleur respect des contraintes physiques.





---

## Conclusion
- Résumé des observations
- Avantages et limites de chaque méthode
- Perspectives
