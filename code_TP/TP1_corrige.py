import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')


# === Étape 1 : Lecture et affichage de l'image ===
img = cv.imread("tulipes.jpg")

if img is None:
    print("⚠️ Erreur : image introuvable !")
else:
    cv.namedWindow('image originale', cv.WINDOW_NORMAL)
    cv.imshow('image originale', img)
    cv.waitKey(0)
# 💬 imread lit une image sous forme de tableau NumPy (3 canaux BGR).


# === Étape 2 : Filtrage bilatéral ===
# (filtre non linéaire qui préserve les contours)
filtered = cv.bilateralFilter(img, d=20, sigmaColor=19, sigmaSpace=19)

cv.namedWindow('image filtrée', cv.WINDOW_NORMAL)
cv.imshow('image filtrée', filtered)
cv.waitKey(0)

# 💬 Contrairement au flou gaussien, le filtrage bilatéral floute l’image
# tout en conservant les bords nets (grâce à la pondération sur la distance ET la couleur).


# === Étape 3 : Conversion en niveaux de gris ===
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow('image grise', cv.WINDOW_NORMAL)
cv.imshow('image grise', gray)
cv.waitKey(0)

# 💬 Conversion : gray = 0.299*R + 0.587*G + 0.114*B
# Cette pondération correspond à la sensibilité de l’œil humain aux couleurs.


# === Étape 4 : Seuillage binaire simple ===
th = 255.0 * (gray > 150.0)

cv.namedWindow('image seuillée', cv.WINDOW_NORMAL)
cv.imshow('image seuillée', np.uint8(th))
cv.waitKey(0)

# 💬 Les pixels dont l’intensité est > 150 deviennent blancs (255), les autres noirs (0).


# === Étape 5 : Espaces de couleur ===
hsv = cv.cvtColor(filtered, cv.COLOR_BGR2HSV_FULL)

colors = ('r', 'g', 'b')
labels = ('h', 's', 'v')

for i, col in enumerate(colors):
    hist = cv.calcHist([hsv], [i], None, [256], [0, 256])
    plt.plot(hist, color=col, label=labels[i])
    plt.xlim([0, 256])

plt.legend()
plt.show()
plt.savefig('fig1.pdf')

# 💬 H = teinte (0–255), S = saturation (vivacité des couleurs), V = valeur (luminosité)
# Le graphe montre la distribution des pixels selon ces composantes.


# === Étape 6 : Extraction et affichage des canaux ===
h = hsv[:, :, 0]
s = hsv[:, :, 1]

cv.namedWindow('teinte (Hue)', cv.WINDOW_NORMAL)
cv.imshow('teinte (Hue)', np.uint8(h))
cv.waitKey(0)

cv.namedWindow('saturation', cv.WINDOW_NORMAL)
cv.imshow('saturation', np.uint8(s))
cv.waitKey(0)

# 💬 Hue = “couleur pure”, Saturation = “quantité de couleur”.


# === Étape 7 : Seuillage sur teinte et saturation ===
thh = 255 * (((h < 15.0) | (h > 245)) & (s > 200.0))

cv.namedWindow('image seuillée H et S', cv.WINDOW_NORMAL)
cv.imshow('image seuillée H et S', np.uint8(thh))
cv.waitKey(0)

# 💬 Cette formule isole les teintes proches du rouge
# (H faible ou proche de 255) et fortement saturées (S > 200).


# === Étape 8 : Morphologie (érosion / ouverture) ===
erosion_shape = cv.MORPH_RECT  # Essayez aussi MORPH_ELLIPSE ou MORPH_CROSS
erosion_size = 15

element = cv.getStructuringElement(
    erosion_shape,
    (2 * erosion_size + 1, 2 * erosion_size + 1),
    (erosion_size, erosion_size)
)

# Nettoyage du bruit avec une ouverture morphologique

erosion_dst = cv.morphologyEx(np.uint8(thh), cv.MORPH_OPEN, element)

cv.namedWindow('image morphologique', cv.WINDOW_NORMAL)
cv.imshow('image morphologique', np.uint8(erosion_dst))
cv.waitKey(0)
cv.destroyAllWindows()

# 💬 L’ouverture = érosion + dilatation, utile pour supprimer les petits points parasites.


# === Étape 9 : Composantes connexes ===
retval, labels = cv.connectedComponents(erosion_dst)
print("Nombre de composantes détectées :", retval)

for i in range(retval):
    idcc = np.where(labels == i)
    if len(idcc[0]) > 1000:
        cv.namedWindow('composante'+str(i), cv.WINDOW_NORMAL)
        cv.imshow('composante'+str(i), np.uint8((labels == i) * 255))
        print(np.mean(idcc, axis=1))
cv.waitKey(0)
cv.destroyAllWindows()
# 💬 connectedComponents renvoie :
#    - retval : nombre de composantes (y compris le fond)
#    - labels : image où chaque composante a un identifiant unique
# np.mean(idcc, axis=1) donne le centre approximatif de chaque région.
