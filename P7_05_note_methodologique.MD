[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/antoine-mle/openclassrooms-p7/main/app.py/)

# Une note méthodologique

## 1. Contexte

Ce mémoire constitue l’un des livrables du projet «Implémentez un modèle de scoring » du parcours
Data Scientist d’Openclassrooms. Il présente le processus de modélisation et d’interprétabilité du modèle mis en place dans le cadre du projet.

Le projet consiste à développer pour la société « Prêt à Dépenser », une société de crédit de
consommation, un modèle de scoring de la probabilité de défaut de paiement d’un client avec pas ou
peu d’historique de prêt.

Les données utilisées pour ce projet sont une base de données de 307 000 clients comportants 121
features (âge, sexe, emploi, logement, revenus, informations relatives au crédit, notation externe,
etc.)

## 2. Méthodologie d’entraînement du modèle

Le modèle entraîné dans le cadre de ce projet a été entraîné sur la base du jeu de données après analyse exploratoire et création de nouvelles features. Avec la reprise **de deux notebooks**  consultable sur le site
Kaggle.
* [Start Here: A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction#Exploratory-Data-Analysis)
* [Tuning Automated Feature Engineering (Exploratory)](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory)

Aprés avoir **fusionné leurs features ingenering**, j'ai confirmé leur hyperparamètre. Polur cela, le jeu de données initial a été séparé en plusieurs parties de façon à disposer :

* D’un jeu de training (80% des individus) qui a été séparé en plusieurs folds pour entraîner les différents modèles et optimiser les paramètres (cross validation) sans overfitting.
* D’un jeu de test (20 % des individus) pour l’évaluation finale du modèle

![Cross validation](https://miro.medium.com/max/3276/1*x8gKAC7dvsOQ6Zsev3zRvg.png)
 
  

### Une classe sous représentée

Le problème est un problème de classification binaire avec une classe sous représentée (9 % de clients en défaut contre 91 % de clients sans défaut). Ce déséquilibre des classes doit être pris en compte dans l’entraînement des modèles puisqu’un modèle « naif » prédisant systématiquement que les clients sont sans défaut aurait une accuracy (justesse) de 0.92 et pourrait être considéré à tort comme un modèle performant alors qu’il ne permettrait pas de détecter les clients à risque.

![Cross validation](https://cdn.discordapp.com/attachments/949493355354677278/949589241061605417/unknown.png)


### Solutions testées

