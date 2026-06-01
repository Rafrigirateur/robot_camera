import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')  # Évite les erreurs d'affichage graphique


# === Étape 1 : Lecture et affichage de l'image ===
# -------------------------------------------------
# Lecture de l'image
img = cv.imread("tulipes.jpg")

# Vérification que l'image est bien chargée
if img is None:
    print("⚠️ Erreur : image introuvable ! Vérifiez le chemin du fichier.")
else:
    cv.namedWindow('image originale', cv.WINDOW_NORMAL)
    cv.imshow('image originale', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# === Étape 2 : Filtrage bilatéral ===
# ------------------------------------
# TODO : Appliquez un filtrage bilatéral à l’image.
# Quelle est la différence entre un flou bilatéral et un flou gaussien ?
# Quels sont les trois paramètres principaux du filtre ?
filtered = ...

cv.namedWindow('image filtrée', cv.WINDOW_NORMAL)
cv.imshow('image filtrée', filtered)
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 3 : Conversion en niveaux de gris ===
# -----------------------------------------------
# Conversion de l’image couleur en niveaux de gris
# TODO : utilisez cv.cvtColor avec le bon code de conversion.
gray = ...

cv.namedWindow('image grise', cv.WINDOW_NORMAL)
cv.imshow('image grise', gray)
cv.waitKey(0)
cv.destroyAllWindows()

# QUESTION : Comment est effectuée la conversion en niveaux de gris concrètement ?
# Donnez la formule mathématique qui combine les canaux R, G et B.


# === Étape 4 : Seuillage binaire simple ===
# ------------------------------------------
# TODO : Appliquez un seuillage simple avec une valeur de 150.
# Que signifie cette opération sur les pixels de l’image ?
th = ...

cv.namedWindow('image seuillée', cv.WINDOW_NORMAL)
cv.imshow('image seuillée', np.uint8(th))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 5 : Espaces de couleur ===
# ------------------------------------
# Conversion de l’image filtrée en espace HSV
# TODO : effectuez la conversion avec cv.cvtColor.
hsv = ...

# Calcul et affichage des histogrammes des trois canaux
colors = ('r', 'g', 'b')
labels = ('h', 's', 'v')

for i, col in enumerate(colors):
    # TODO : calculez l’histogramme du canal i
    hist = ...
    plt.plot(hist, color=col, label=labels[i])
    plt.xlim([0, 256])

plt.legend()
plt.show()
plt.savefig('fig1.pdf')

# QUESTION : Que représente la courbe sauvegardée dans fig1.pdf ?
# Quelle information donne-t-elle sur l’image ?


# === Étape 6 : Extraction et affichage des canaux ===
# ----------------------------------------------------
# TODO : extrayez les canaux H et S à partir de hsv.
h = ...
s = ...

cv.namedWindow('teinte (Hue)', cv.WINDOW_NORMAL)
cv.imshow('teinte (Hue)', np.uint8(h))
cv.waitKey(0)
cv.destroyAllWindows()

cv.namedWindow('saturation', cv.WINDOW_NORMAL)
cv.imshow('saturation', np.uint8(s))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 7 : Seuillage sur teinte et saturation ===
# ----------------------------------------------------
# TODO : Créez un masque pour isoler certaines couleurs.
# Exemple : détection de tons rouges à forte saturation.
thh = ...

cv.namedWindow('image seuillée H et S', cv.WINDOW_NORMAL)
cv.imshow('image seuillée H et S', np.uint8(thh))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 8 : Morphologie (érosion / ouverture) ===
# ---------------------------------------------------
# Création de l’élément structurant
# TODO : essayez différentes formes : RECT, ELLIPSE, CROSS
erosion_shape = ...
erosion_size = 15

element = cv.getStructuringElement(
    erosion_shape,
    (2 * erosion_size + 1, 2 * erosion_size + 1),
    (erosion_size, erosion_size)
)

# TODO : appliquez une opération morphologique (érosion ou ouverture)
erosion_dst = ...

cv.namedWindow('image morphologique', cv.WINDOW_NORMAL)
cv.imshow('image morphologique', np.uint8(erosion_dst * 255))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 9 : Composantes connexes ===
# --------------------------------------
# TODO : appliquez cv.connectedComponents() à l’image binaire.
retval, labels = ...

print("Nombre de composantes détectées :", ...)

# Parcours des composantes détectées
for i in range(...):
    idcc = ...
    if len(idcc[0]) > 1000:  # seuil minimal de taille
        cv.namedWindow('composante', cv.WINDOW_NORMAL)
        cv.imshow('composante', np.uint8((labels == i) * 255))

        # TODO : calculez et affichez la moyenne des coordonnées.
        print(...)

        cv.waitKey(500)
        cv.destroyAllWindows()

# Exercice supplémentaire pour ceux qui ont fini.
# Afficher la photo des tulipes avec une petite croix au milieu 
# de chaque tulipe détectée et le nombre de tulipes en bas.