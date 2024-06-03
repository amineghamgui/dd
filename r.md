# Modèle X3D_S pré-entraîné sur Kinetics 400

## Données
- Vidéo de 1 seconde, chacune possède une seule annotation.
- Classes :
  0. "SL"
  1. "NR"

## Fonction de perte (Loss)
Utilisation de Focal Loss.

### Personnalisation du code en fonction de votre base de données

  - Ligne 58 : Chemin du fichier CSV contenant les chemins, les étiquettes, etc., des vidéos pour l'entraînement.
  - Ligne 61 : Chemin du fichier CSV contenant les chemins, les étiquettes, etc., des vidéos pour la validation.
  - ligne 88: Chemin des vidéos.

## Lancement d'entraînement
Pour lancer l'entraînement, suivez les étapes ci-dessous :

1. Naviguez dans le répertoire `x3d` :
   cd x3d
2. Installez les dépendances nécessaires :
   pip install -r requirements.txt 
3. Lancez le script d'entraînement :
   python trainX3D.py

