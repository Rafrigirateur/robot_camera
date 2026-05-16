import cv2
import numpy as np
from Hardware.camera import Camera
from Hardware.moteur import Moteur

def main():
    # 1. Initialisation des composants
    # On utilise la résolution légère (160x120) pour maximiser les FPS
    largeur_image = 160
    hauteur_image = 120
    
    cam = Camera(camId=1, width=largeur_image, height=hauteur_image, fps=30)
    moteurs = Moteur()
    
    # Le centre théorique de la caméra sur l'axe X
    centre_vire = largeur_image // 2 
    
    print("Démarrage du Suiveur de Ligne. Appuyez sur 'q' pour quitter.")

    try:
        while True:
            # 2. Récupération de l'image via ta classe Camera
            frame = cam.get_frame()
            if frame is None:
                print("Erreur de lecture caméra.")
                break

            # 3. Traitement d'image (Colorimétrie HSV)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Seuils pour la ligne noire
            low_b = np.array([0, 0, 0], dtype=np.uint8)
            high_b = np.array([180, 255, 50], dtype=np.uint8)
            mask = cv2.inRange(hsv, low_b, high_b)
            
            # Extraction des contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            erreur = 0  # Erreur par défaut si on perd la ligne
            
            if len(contours) > 0:
                # On prend le plus grand contour (la ligne)
                c = max(contours, key=cv2.contourArea)
                
                # 4. Calcul du centre de masse (Centroïde) du contour via les Moments
                M = cv2.moments(c)
                if M["m00"] != 0: # Évite la division par zéro sur des micro-contours bruités
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 5. Calcul de l'erreur pour le PID
                    erreur = cx - centre_vire
                    
                    # --- Section Visuelle (Optionnelle, pour debug) ---
                    # Dessiner le contour en vert
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                    # Dessiner le centre de la ligne (point bleu)
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                    # Dessiner le centre idéal de l'image (croix rouge)
                    cv2.drawMarker(frame, (centre_vire, hauteur_image // 2), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                
                # --- ZONE D'EXPLOITATION PID ---
                # C'est ici que tu injecteras ton calcul PID :
                # commande = PID.calculer(erreur)
                # moteurs.piloter(vitesse_base + commande, vitesse_base - commande)
                
                # Pour tester sans PID (juste un mini correcteur proportionnel basique) :
                # Kp = 0.5
                # correction = erreur * Kp
                # moteurs.piloter(30 + correction, 30 - correction)
                
                print(f"Position Ligne (Cx): {cx} | Erreur transmissible au PID: {erreur}")
            else:
                # Si on ne voit plus la ligne, on s'arrête de sécurité (ou stratégie de recherche)
                print("Ligne perdue !")
                moteurs.stop()

            # 6. Affichage des fenêtres de Debug
            cv2.imshow("Masque Noir", mask)
            cv2.imshow("Vue Robot", frame)
            
            # Sortie propre
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur.")
        
    finally:
        # Nettoyage propre de tous les périphériques
        moteurs.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("Robot arrêté proprement.")

if __name__ == "__main__":
    main()