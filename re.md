# Modèle X3D_S pré-entraîné sur Kinetics 400

## Données
- Vidéo de 1 seconde, chacune possède une seule annotation.
- Classes :
  1. "SL"
  2. "NR"

## Fonction de perte (Loss)
Utilisation de Focal Loss.

## Lancement d'entraînement
Pour lancer l'entraînement, suivez les étapes ci-dessous :

1. Naviguez dans le répertoire `x3d` :
   cd x3d
2. Installez les dépendances nécessaires :
   pip install -r requirements.txt 
3. Lancez le script d'entraînement :
   python trainX3D.py
