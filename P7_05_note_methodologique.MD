[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/antoine-mle/openclassrooms-p7/main/app.py/)

# **Note méthodologique**

## **0. Contexte**

Ce mémoire est l'un des livrables du projet « Implémenter un modèle de scoring » de la formation Openclassrooms Data Scientist. Il présente le processus de modélisation et d'interprétabilité du modèle mis en œuvre dans le cadre du projet.

Le projet consiste à développer pour [la société « Prêt à Dépenser »](https://www.kaggle.com/c/home-credit-default-risk/data), une société de crédit de consommation, un modèle de scoring de la probabilité de défaut de paiement d’un client avec pas ou peu d’historique de prêt.

<br /> 

## **1. Méthodologie d’entraînement du modèle**

Le modèle entraîné dans le cadre de ce projet a été entraîné sur la base du jeu de données après analyse exploratoire et création de nouvelles features. Avec la reprise **de deux notebooks**  consultable sur le site
Kaggle.
* [Start Here: A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction#Exploratory-Data-Analysis)
* [Tuning Automated Feature Engineering (Exploratory)](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory)

Aprés avoir **fusionné leurs features ingenering**, j'ai confirmé leur hyperparamètre. Polur cela, le jeu de données initial a été séparé en plusieurs parties de façon à disposer :

* D’un jeu de training (80% des individus) qui a été séparé en plusieurs folds pour entraîner les différents modèles et optimiser les paramètres (cross validation) sans overfitting.
* D’un jeu de test (20 % des individus) pour l’évaluation finale du modèle
<p align="center" width="100%">
<img src="https://miro.medium.com/max/3276/1*x8gKAC7dvsOQ6Zsev3zRvg.png" width="350"/></p>

<br /> 

## **2. La fonction coût métier, l'algorithme d'optimisation et la métrique d'évaluation**

### **2.1 La fonction coût métier**
Un client aura contribution financière différente pour l’organisation selon sa classe :
* le vrai négatif contribue positivement aux résultats en amenant une marge,
* le faux positif contribue négativement aux résultats en amenant une perte,
* le vrai positif est évité à raison par le modèle
* le faux négatif n’est pas validé par le modèle et ne contribue donc pas aux gains ou perte de l’organisation.
<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/950773934264029224/unknown.png" width="500"  /></p>

Sur cette base, on peut en déduire une première formulation de la marge du modèle :

<pre>   <b>Gain = Somme des marges des TN – somme des pertes des FN</b>
</pre>
Si on rajoute le taux d'intérêt pour les marges, on a une perte de 100% pour simplifier si on une perte, et un gain égale au taux d'intérêt totale s'il n'y pas de défaut.
Ce qui donne :
<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/951824198349783040/unknown.png" width="400"  /></p>

Le taux d'intéêt étant égale à environ 10%, il est donc 10 fois plus utile d'agir sur les faux négatifs.

<br /> 

### **2.2 L'algorithme d'optimisation**
J'ai d'abord testé en trois modéles de classification différents comme l'ont fait l'un des deux notebooks mentionnés ci-dessus: un randomforest, un logistic et LightGBM. Il semblerait évident que ce dernier soit plus prometteur, même s'il restait à le configurer au mieux.

**Une classe cible sous représentée**, le problème est une classification binaire avec une classe sous représentée (9 % de clients en défaut contre 91 % de clients sans défaut). Ce déséquilibre des classes doit être pris en compte dans l’entraînement des modèles puisqu’un modèle « naif » prédisant systématiquement que les clients sont sans défaut aurait une accuracy (justesse) de 0.92 et pourrait être considéré à tort comme un modèle performant alors qu’il ne permettrait pas de détecter les clients à risque.

<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/950751421521874994/unknown.png" width="350"  /> </p>


Pour cela, j'ai testé **SMOTE** un algorithme pour ajouter des clients afin d'équilibrer l'équilibre des deux classes cibles. Dans le même but, j'ai également testé **RandomOverSampler, FunctionSampler et class_weight balancing**. C'est ce dernier qui a été retenu ayant un meilleur score avec la métrique retenue pour ce projet.
<br /> 

### **2.3 La métrique d'évaluation**
Comme vus plus haut, c'est le score AUC (ou Area Under the Curve) qui a été retenu. Car il prend en compte les faux négatifs ce qu'on veut réduire au maximum du résonnable.
<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/951817873549033502/unknown.png" width="800"  /></p>

Comme l'on peut voir, ca diminue la précision au profit du recall.
<br /> 

## **3. L’interprétabilité globale et locale du modèle**

Le classifieur LGBM possède dans sa bibliothèque la méthode ‘feature_importances_’.

En appliquant cette méthode aux jeux de données processées, on obtient le poids de chaque variable dans le calcul du score de probabilité de classe. Le graphique ci-dessous indique par ordre décroissant d’importance les premières variables les plus importantes :
<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/950775394922016768/unknown.png" width="500"/></p>

Mais en utilisant la bibliothèque SHAP, on peut avoir ce graphique à la fois au niveau **globale**
<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/950776342247530596/unknown.png" width="500"/></p>

 mais aussi au **niveau locale** : pour un client.
<p align="center" width="100%">
<img src="https://cdn.discordapp.com/attachments/949493355354677278/950776603632365588/unknown.png" width="800"/></p>
<br /> 

## **4. Les limites et les améliorations possibles**
### **4.1 Lenteur du modéle**

Le score AUC obtenu après la fusion des deux notebooks, est de 0,788 contre 0.786 pour le meilleur des deux. Malgré cela, le nombre de features a presque doublé, rendant les pipelines plus complexes et doublant le temps de formation des modèles.

De plus, j'ai grandement repris la plus part des hyperparmètres de [Will Koehrsen](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory), de l'optimisation y est clairement possible.
<br /> 

### **4.2 Performance métier**
Les remarques précédentes étant posées, il convient de rappeler, comme abordé dans la section sur la mesure d’évaluation, que la sélection et l’optimisation du modèle s’appuient sur une mesure technique, le score AUC.
Afin de déterminer le seuil optimal d’un point de vue métier, la fonction ‘gain’ a été construite.Cette même fonction pourrait être utilisée pour la recherche d’optimisation des hyper paramètres avec la fonction ‘make_scorer’.

La fonction gain utilise une série d’hypothèses globales extérieures faisant appel :
* au chiffre d’affaires moyen par client,
* au gain moyen en l’absence de défaut de paiement 
* à la perte moyenne dans le cas d’un défaut de paiement.
  
Ces hypothèses peuvent être affinées dans un premier temps en utilisant les valeurs moyennes constatées dans le jeu de données d’entrainement. Dans un second temps, il est possible de d’utiliser des valeurs moyennes qui caractériseraient par exemple des classes de variables pour les 4 variables les plus importantes.
De façon plus absolue, les profits et pertes pourraient être calculés de façons individuelles.

<br /> 

### **4.3 La détection d'anomalie**
Le but de la détection d’anomalie est de repérer des données qui ne sont pas conformes à ce à quoi l’on peut s’attendre par rapport aux autres données.

Il s’agit, par exemple, de données qui ne suivent pas le même schéma ou qui sont atypiques pour la distribution de probabilité observée. La difficulté du problème provient du fait qu’on ne connait pas au préalable la distribution sous-jacente de l’ensemble des données. C’est à l’algorithme d’apprendre une métrique appropriée pour détecter les anomalies qui est souvent un problème d’apprentissage de type non supervisé. Les algorithmes typiques de détection d’anomalie sont les one-class SVM, les méthodes d’apprentissage de distribution bayésienne et les random forests.

Ce ceci n'a pas été explorer dans ce projet.