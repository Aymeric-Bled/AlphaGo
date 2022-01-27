# Projet AlphaGo par Aymeric BLED et Rémi BARBOSA

Nous avons implémenté MCTS et le réseau de neurones en RL.
L'entraînement du réseau s'est fait par CNN_notebook.ipynb sur colab.

Nous avons 2 joueurs :
- myPlayerAymeric.py contient une version de MCTS fonctionnelle
- myPlayerRemi.py contient une version de MCTS fonctionnelle, ainsi que l'intégration du réseau de neurones (EN COURS).
Il est possible de choisir entre la version avec ou sans réseau en modifiant la variable ```use_nn```.

Pour lancer une partie :
```python namedGame.py myPlayerRemi.py myPlayerAymeric.py```
