# Résolution d’un problème de Poisson stationnaire par différentes méthodes SciML

## Introduction
Brève présentation du problème étudié et des objectifs.  
Equation de Poisson :  
\[
- \Delta u(x, y) = f(x, y), \quad u|_{\partial\Omega} = 0
\]

---

## 1. Simulation par différences finies

### 1.1 Formulation du problème
- Domaine et conditions aux limites
- Discrétisation de l'équation
- Maillage utilisé

### 1.2 Implémentation
- Schéma utilisé (e.g. central à l’ordre 2)
- Construction de la matrice
- Résolution (solveur direct ou itératif)

### 1.3 Résultats
- Visualisation de la solution
- Erreur si solution exacte connue

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
