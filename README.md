# RecrutIA - Détection et Mitigation des Biais Algorithmiques en Prédiction RH
 
**Projet de Fin de Module** - Python pour la Science des Données  
Master d'Excellence en Intelligence Artificielle  
Faculté des Sciences Ben M'Sick - Université Hassan II de Casablanca
 
---
 
## Aperçu
 
**RecrutIA** est une application d'analyse de CV alimentée par l'IA qui prédit le risque d'attrition des employés tout en garantissant l'équité algorithmique. Ce projet démontre l'application de Python pour résoudre un problème réel de science des données dans le domaine des ressources humaines.
 
### Problématique
 
Les modèles de prédiction du turnover entraînés sur des données RH historiques héritent souvent de biais structurels, conduisant à des discriminations contre certains groupes (genre, âge, situation familiale) et à des risques légaux (non-conformité RGPD, lois anti-discrimination).
 
### Solution Proposée
 
Un pipeline complet combinant :
- Analyse exploratoire avec Pandas et visualisations
- Modélisation avec Random Forest et techniques de rééquilibrage
- Détection de biais via métriques d'équité (Disparate Impact, Statistical Parity)
- Mitigation par Reweighing (prétraitement)
- Déploiement d'une application Streamlit avec extraction CV par IA
### Dataset
 
**IBM HR Analytics Employee Attrition & Performance**
- 1 470 employés
- 35 variables (démographiques, professionnelles, comportementales)
- Variable cible : Attrition (Yes/No)
- Déséquilibre : 16,1% de départs
---
 
## Fonctionnalités
 
### Analyse de Biais
- Calcul automatique des métriques d'équité (Disparate Impact, Statistical Parity Difference)
- Identification des attributs sensibles biaisés (genre, situation familiale, âge)
- Correction par Reweighing avec validation quantitative
### Extraction de CV par IA
- Upload de CV PDF avec drag & drop
- Extraction automatique via Llama 3.3 70B (API Groq)
- Score de confiance (0-100%) de l'extraction
- Correction manuelle des données extraites
### Interface Interactive
- **Mode 1** : Analyse automatique de CV (PDF)
- **Mode 2** : Saisie manuelle de profil
- Prédiction en temps réel avec visualisations Plotly
- Recommandations graduées (A/B/C/D)
### Analyse Narrative par IA
- Explication contextuelle générée par Llama 3.3
- Identification des facteurs positifs et facteurs de risque
- Recommandations RH concrètes
---
 
## Utilisation
 
### Mode 1 : Analyse de CV (PDF)
1. Sélectionner le mode "Analyse CV (PDF)"
2. Uploader un CV PDF ou utiliser les CV exemples
3. Vérifier les données extraites automatiquement
4. (Optionnel) Corriger les erreurs
5. Consulter les résultats avec score, grade et analyse narrative
### Mode 2 : Saisie Manuelle
1. Sélectionner le mode "Saisie Manuelle"
2. Remplir le formulaire (profil personnel, formation, expérience, géographie, disponibilité)
3. Analyser le candidat
4. Consulter les mêmes résultats
### Interprétation des Résultats
 
| Score | Grade | Décision | Signification |
|-------|-------|----------|---------------|
| ≥80% | A | Fortement recommandé | Faible risque d'attrition |
| 60-79% | B | Recommandé | Risque modéré |
| 40-59% | C | À évaluer | Réserves à vérifier |
| <40% | D | Déconseillé | Risque élevé |
 
---
 
## Méthodologie
 
### 1. Prétraitement des Données
 
Chargement du dataset IBM HR Analytics, suppression des variables constantes et de l'identifiant technique, encodage des variables catégorielles (LabelEncoder), normalisation des features numériques (StandardScaler). Réduction de 35 variables à 8 variables finales retenues.
 
### 2. Gestion du Déséquilibre de Classes
 
Le dataset présente un fort déséquilibre avec seulement 16,1% de cas d'attrition. Quatre techniques testées :
- Pondération des classes (class_weight='balanced')
- SMOTE (Synthetic Minority Over-sampling)
- SMOTETomek (SMOTE + nettoyage)
- Balanced Random Forest (meilleur résultat : Recall=55%, F1=0.33)
### 3. Détection de Biais
 
Utilisation de la bibliothèque IBM AIF360 pour calculer les métriques d'équité avant mitigation :
 
| Attribut Protégé | Disparate Impact | Statistical Parity Diff | Diagnostic |
|------------------|------------------|-------------------------|------------|
| Genre | 0.87 | -0.022 | Biais modéré |
| Situation Familiale | 1.538 | 0.067 | Biais significatif |
| Âge | 1.738 | 0.081 | Biais significatif |
 
### 4. Mitigation des Biais
 
Application de la technique Reweighing (prétraitement) qui attribue des poids aux instances pour équilibrer la distribution des labels à travers les groupes protégés. Entraînement du modèle Random Forest avec les poids de rééquilibrage.
 
### 5. Sélection de Modèle
 
Comparaison de 7 classifieurs (Logistic Regression, Random Forest, SVM, Gaussian Naive Bayes, K-NN, Gradient Boosting, XGBoost). Random Forest retenu comme meilleur compromis avec optimisation des hyperparamètres via GridSearchCV.
 
**Modèle final** : RandomForestClassifier avec n_estimators=200, max_depth=20, class_weight='balanced_subsample'.
 
---
 
## Résultats
 
### Performances Globales (Ensemble de Test)
 
```
                  Precision  Recall  F1-Score  Support
Classe 0 (Reste)     0.85     0.98      0.91      247
Classe 1 (Départ)    0.50     0.11      0.18       47
Accuracy                                 0.84      294
```
 
### Matrice de Confusion
 
```
                 Prédiction
              Reste  Départ
Réalité Reste   242      5
       Départ    42      5
```
 
### Métriques d'Équité (Après Reweighing)
 
**Pour la Situation Familiale** :
 
| Métrique | Avant | Après | Seuil | Statut |
|----------|-------|-------|-------|--------|
| Disparate Impact | 1.538 | 1.029 | 0.8 - 1.2 | Conforme |
| Statistical Parity Diff | 0.067 | 0.005 | ±0.1 | Conforme |
| Accuracy | - | 78.46% | - | Maintenue |
 
**Pour le Genre** :
 
| Métrique | Avant | Après | Seuil | Statut |
|----------|-------|-------|-------|--------|
| Disparate Impact | 0.87 | 1.202 | 0.8 - 1.2 | Acceptable |
| Statistical Parity Diff | -0.022 | 0.056 | ±0.1 | Conforme |
| Accuracy | - | 71.43% | - | Maintenue |
 
### Observations
 
**Succès** :
- Réduction du biais de 75-95% selon les attributs
- Accuracy maintenue au-dessus de 71%
- Très faible taux de faux positifs (5/294)
- Conformité aux seuils d'équité standards
**Limites** :
- Recall classe minoritaire faible (11%)
- 89% des départs ne sont pas détectés
- Compromis équité-performance inhérent
- Biais de genre plus difficile à éliminer
---
 
## Technologies Utilisées
 
**Data Science** : Python 3.12, Pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost
 
**Équité & Biais** : IBM AIF360
 
**IA Générative** : Groq SDK, Llama 3.3 70B, pdfplumber
 
**Visualisation & Interface** : Streamlit, Plotly, Matplotlib, Seaborn
 
---
 
## Équipe
 
**Projet de Fin de Module** - Python pour la Science des Données  
Master d'Excellence en Intelligence Artificielle  
Faculté des Sciences Ben M'Sick - Université Hassan II de Casablanca
 
### Réalisé par :
- ALOUANI Hajar
- BOUSYF Wahiba
- CHAABAN Malika
- BOUSSAID Salama
### Encadrement :
- Pr. NOUH Said (Encadrant)
- Dr. Taoufik AMZIL (Co-Encadrant)
**Année Universitaire** : 2025-2026
 
---
 
## Références Principales
 
1. Mehrabi, N., et al. (2022). A Survey on Bias and Fairness in Machine Learning. ACM Computing Surveys, 54(6), 1-35.
2. Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.
3. Zemel, R., et al. (2013). Learning Fair Representations. ICML.
4. Raghavan, M., et al. (2020). Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices. FAT*.
5. Oladipupo, I. A., et al. (2026). Bias and Fairness in AI-Based Employee Attrition Prediction Using Random Forest. Innovations in Computing and Data Science, 2(1), 11-23.
6. Rigotti, C., & Fosch-Villaronga, E. (2024). Fairness, AI & Recruitment. Computer Law & Security Review, 53, 105966.
7. IBM AI Fairness 360 Toolkit - https://aif360.mybluemix.net/
---
 
## Licence
 
Ce projet est sous licence MIT.
 
---
 
## Remerciements
 
Nous remercions sincèrement Pr. NOUH Said et Dr. Taoufik AMZIL pour leur encadrement, la Faculté des Sciences Ben M'Sick pour l'infrastructure académique, IBM Research pour la bibliothèque AIF360, Groq pour l'accès API Llama 3.3, ainsi que nos familles et amis pour leur soutien constant.
 
---
 
## Citation
 
```bibtex
@mastersproject{alouani2026recrutia,
  title={Détection et mitigation des biais algorithmiques dans les modèles 
         de prédiction du turnover des ressources humaines},
  author={Alouani, Hajar and Bousyf, Wahiba and Chaaban, Malika and Boussaid, Salama},
  year={2026},
  school={Université Hassan II de Casablanca - Faculté des Sciences Ben M'Sick},
  type={Projet de Fin de Module},
  course={Python pour la Science des Données},
  program={Master d'Excellence en Intelligence Artificielle}
}
```
 
---
 
*Projet de Fin de Module - Python pour la Science des Données*  
*Master d'Excellence en Intelligence Artificielle - FSBM - UH2C*  
*Année Universitaire 2025-2026*
