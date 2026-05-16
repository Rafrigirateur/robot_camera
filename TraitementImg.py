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
    
    cam = Camera(camId=1, width=largeur_image, height=hauteur_image, fps=30)
    moteurs = Moteur()
    
    # Initialisation du PID (Coefficients à ajuster lors de tes tests !)
    # Règle d'abord Kp (ex: 0.4), laisse Ki à 0, et mets un poil de Kd (ex: 0.05)
    pid = PID(kp=0.5, ki=0.0, kd=0.02)
    
    # Vitesse de croisière du robot (sur 100)
    vitesse_base = 35 
    centre_vire = largeur_image // 2 
    
    print("Démarrage du Suiveur de Ligne. Ctrl+C pour arrêter.")
    time.sleep(1) # Laisse le temps à l'utilisateur de poser le robot au sol

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
                # Le plus gros contour est considéré comme la ligne
                c = max(contours, key=cv2.contourArea)
                
                # Calcul du centre (Moments)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calcul de l'erreur (-80 à +80)
                    erreur = cx - centre_vire
                    
                    # 4. Calcul de la commande PID
                    commande = pid.calculer(erreur)
                    
                    # 5. Application aux moteurs (Direction différentielle)
                    vitesse_gauche = vitesse_base + commande
                    vitesse_droite = vitesse_base - commande
                    moteurs.piloter(vitesse_gauche, vitesse_droite)
                    
                    # Debug dans le terminal
                    print(f"Err: {erreur:3d} | Cmd: {commande:5.1f} | Moteurs: G:{vitesse_gauche:5.1f} D:{vitesse_droite:5.1f}")
                    
                    # Éléments de dessin pour le debug visuel
                    if AFFICHAGE_ACTIF:
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                        cv2.drawMarker(frame, (centre_vire, hauteur_image // 2), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
            else:
                # Sécurité : Si le robot perd la ligne, il s'arrête immédiatement
                print("Ligne Perdue ! Arrêt d'urgence.")
                moteurs.stop()
                pid.reset()

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