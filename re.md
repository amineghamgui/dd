# You Only Watch Once (YOWO)
## Détection d’actions

### Fonctionnement
- **Backbone 2D**: Permet de détecter la personne.
- **Backbone 3D**: Permet de détecter l'action.
- **Attention**: Aggrégation des caractéristiques du backbone 3D avec celles du backbone 2D pour suivre les personnes et leurs actions.

### Modification du code pour l'adapter à votre machine
- **dataset/ava.py** :
  - Ligne 24 : Chemin d'accès vers train.csv
  - Ligne 27 : Chemin d'accès vers val.csv
  - Ligne 77 : Chemin d'accès vers les vidéos d'une seconde

- **evaluator/ava_evaluator.py** :
  - Ligne 40 : Chemin d'accès vers val.csv

- **train_finetuning_ava.py** :
  - Ligne 196 : Chemin d'accès vers les poids du modèle YOWO Nano entraîné sur Ava
  - Ligne 144 : Clé WandB

### Classes
1. "SL" (Action suspecte)
2. "NR" (Aucune action suspecte)

### Rôle des classes
- **dataset/ava.py** : Classe Dataset Pytorch
- **transform.py** : Transformation des données

- **evaluator/ava_eval_helper.py** : Classe complémentaire pour ava_evaluator
- **ava_evaluator.py** : Étapes de validation
- **cal_frame_mAP.py** : Calcul de la métrique mAP_frame
- **cal_video_mAP.py** : Calcul de la métrique mAP_video
- **utils.py** : Utilitaires divers

### Fonction de perte (Loss)
**Loss détection**
- **L1, smooth(x, y)** : 
  - Formule : 0.5(x − y) ^2 si |x − y| < 1, |x − y| − 0.5 sinon
  - Utilisation : Moins sensible aux valeurs aberrantes et aux gradients explosifs, pour la localisation des objets.

- **LMSE(x, y)** :
  - Formule : (x − y)^2
  - Utilisation : Calcul de l'erreur entre les scores de confiance prédits et réels.

**Loss classification**
- **Lfocal(x, y)** :
  - Formule : y(1−x)^γ log(x) + (1−y)x^γ log(1−x)
  - x : Prédictions de probabilité de classe
  - y : Étiquettes de classe réelles (0 ou 1)
  - γ : Facteur de modulation pour le déséquilibre de classe

