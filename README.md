# 📈 Application de Prévision du Profit (Streamlit)

Ce projet est une application web interactive construite avec **Streamlit** permettant de prédire le profit d'entreprises (startups) en fonction de leurs dépenses et de leur localisation. L'application propose un pipeline complet de Machine Learning, allant de l'exploration des données à la prédiction en temps réel, en passant par le prétraitement et l'entraînement de modèle.

## 🚀 Fonctionnalités Principales

L'application guide l'utilisateur à travers les étapes suivantes :

### 1. Chargement et Exploration des Données
* **Importation CSV** : Interface simple pour uploader votre propre dataset.
* **Analyse Automatique** : Identification automatique des variables indépendantes et de la variable cible (*Profit*).
* **Statistiques** : Affichage des dimensions du dataset (lignes/colonnes), aperçu des données et détection des valeurs manquantes.

### 2. Prétraitement des Données (Data Preprocessing)
* **Gestion des valeurs manquantes** : Imputation flexible via `SimpleImputer` (Moyenne, Médiane ou La plus fréquente).
* **Encodage des variables catégorielles** :
    * *Binary Encoder*
    * *Label Encoder*
    * *OneHot Encoder*
* **Normalisation (Scaling)** : Mise à l'échelle des données avec `MaxAbsScaler`, `MinMaxScaler`, `StandardScaler` ou `RobustScaler`.

### 3. Analyse et Visualisation
* **Matrice de Corrélation** : Affichage d'une *heatmap* interactive pour visualiser les relations entre les variables numériques.
* **Réduction de Dimension (PCA)** : Option pour appliquer l'Analyse en Composantes Principales (PCA) avec choix dynamique du nombre de composants.

### 4. Modélisation (Machine Learning)
* **Configuration de l'entraînement** : Choix personnalisé du ratio Entraînement/Test (ex: 80% / 20%).
* **Algorithme** : Utilisation de la **Régression Linéaire** (`LinearRegression` de Scikit-learn).
* **Évaluation des Performances** : Calcul automatique des métriques clés :
    * MSE (Mean Squared Error)
    * MAE (Mean Absolute Error)
    * RMSE (Root Mean Squared Error)
    * R-Squared ($R^2$)
    * Explained Variance Score
* **Visualisation des Résultats** : Graphique comparatif entre les valeurs réelles et les valeurs prédites.

### 5. Prédiction en Temps Réel
* **Interface de Simulation** : Formulaire interactif pour saisir de nouvelles données (Dépenses R&D, Administration, Marketing, État).
* **Calcul Immédiat** : Prédiction du profit basée sur le modèle entraîné (avec prise en compte automatique du pipeline de prétraitement choisi).

---

## 📂 Structure du Dataset

L'application est optimisée pour des fichiers CSV structurés comme l'exemple fourni (`profitentr.csv`), contenant les colonnes suivantes :

* **R&D Spend** : Dépenses en Recherche et Développement (Numérique).
* **Administration** : Dépenses administratives (Numérique).
* **Marketing Spend** : Dépenses en marketing (Numérique).
* **State** : Localisation de l'entreprise (Catégorielle : ex. "New York", "California").
* **Profit** : Le profit net (Variable cible à prédire).

---

## 🛠️ Technologies Utilisées

* **Langage** : Python 3.x
* **Interface Web** : [Streamlit](https://streamlit.io/)
* **Manipulation de Données** : [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Machine Learning** : [Scikit-learn](https://scikit-learn.org/)
* **Encodage** : [Category Encoders](https://contrib.scikit-learn.org/category_encoders/)
* **Visualisation** : [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

---

## ⚙️ Installation et Utilisation

### Prérequis
Assurez-vous d'avoir Python installé sur votre machine.

### 1. Cloner le projet ou télécharger les fichiers
Placez le fichier `Application.py` et votre dataset (ex: `profitentr.csv`) dans un dossier.

### 2. Installer les dépendances
Ouvrez un terminal et exécutez la commande suivante pour installer les bibliothèques nécessaires :

```bash
pip install streamlit pandas scikit-learn numpy category_encoders matplotlib seaborn
