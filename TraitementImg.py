import cv2
import numpy as np
import time
from Hardware.camera import Camera
from Hardware.moteur import Moteur
from Pid import PID

# Configurable : Mets False pour désactiver le retour vidéo en SSH
AFFICHAGE_ACTIF = False  

def main():
    global AFFICHAGE_ACTIF
    
    # 1. Configuration des composants
    largeur_image = 160
    hauteur_image = 120
    
    cam = Camera(camId=0, width=largeur_image, height=hauteur_image, fps=30)
    moteurs = Moteur()
    
    # Initialisation du PID (Coefficients à ajuster lors de tes tests !)
    # Règle d'abord Kp (ex: 0.4), laisse Ki à 0, et mets un poil de Kd (ex: 0.05)
    pid = PID(kp=0.30, ki=0.005, kd=0.15)
    
    # Vitesse de croisière du robot (sur 100)
    vitesse_base = 20 
    centre_vire = largeur_image // 2 
    
    print("Démarrage du Suiveur de Ligne. Ctrl+C pour arrêter.")
    time.sleep(1) # Laisse le temps à l'utilisateur de poser le robot au sol

    derniere_commande = 0

    try:
        while True:
            # 2. Capture d'image
            frame = cam.get_frame()
            if frame is None:
                print("Erreur : Impossible de lire la caméra.")
                break

            # 3. Traitement d'image HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low_b = np.array([0, 0, 0], dtype=np.uint8)
            high_b = np.array([180, 255, 50], dtype=np.uint8)
            mask = cv2.inRange(hsv, low_b, high_b)
            
            # Extraction des contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    erreur = cx - centre_vire
                    
                    commande = pid.calculer(erreur)
                    derniere_commande = commande  # <-- SAUVEGARDE DE LA COMMANDE ICI
                    
                    vitesse_gauche = vitesse_base + commande
                    vitesse_droite = vitesse_base - commande
                    moteurs.piloter(vitesse_gauche, vitesse_droite)
                    
                    print(f"Err: {erreur:3d} | Cmd: {commande:5.1f} | Moteurs: G:{vitesse_gauche:5.1f} D:{vitesse_droite:5.1f}")
                    
                    # Éléments de dessin pour le debug visuel
                    if AFFICHAGE_ACTIF:
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                        cv2.drawMarker(frame, (centre_vire, hauteur_image // 2), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
            else:
                # --- NOUVELLE STRATÉGIE DE PERTE DE LIGNE ---
                print("Ligne Perdue ! Recherche active...")
                # Au lieu de s'arrêter, le robot pivote sur lui-même dans la dernière direction connue
                if derniere_commande > 0:
                    # La ligne est sortie par la droite, on tourne fort à droite
                    moteurs.piloter(35, -35)
                else:
                    # La ligne est sortie par la gauche, on tourne fort à gauche
                    moteurs.piloter(-35, 35)

            # 6. Gestion de l'affichage sécurisée
            if AFFICHAGE_ACTIF:
                try:
                    cv2.imshow("Masque", mask)
                    cv2.imshow("Robot", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # Si OpenCV crash à cause de l'absence d'écran X11, on coupe l'affichage définitivement
                    print("Serveur graphique non détecté. Passage en mode headless automatisé.")
                    AFFICHAGE_ACTIF = False

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")
        
    finally:
        # Relâchement propre du matériel
        moteurs.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("Fermeture du programme.")

if __name__ == "__main__":
    # Si le script est exécuté directement, on lance la boucle principale
    main()