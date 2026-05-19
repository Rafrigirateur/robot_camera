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
    
    seuil = 5

    cam = Camera(camId=0, width=largeur_image, height=hauteur_image, fps=30)
    moteurs = Moteur()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    enregistreur_video = cv2.VideoWriter('log_robot.avi', fourcc, 30.0, (largeur_image, hauteur_image))

    enregistreur_mask = cv2.VideoWriter('log_mask.avi', fourcc, 30.0, (largeur_image, hauteur_image), isColor=False)

    enregistreur_origine = cv2.VideoWriter('log_origine.avi', fourcc, 30.0, (largeur_image, hauteur_image))
    
    # Initialisation du PID (Coefficients à ajuster lors de tes tests !)
    # Règle d'abord Kp (ex: 0.4), laisse Ki à 0, et mets un poil de Kd (ex: 0.05)
    pid = PID(kp=0.12, ki=0.0, kd=0.05)
    
    # Vitesse de croisière du robot (sur 100)
    VITESSE_MAX = 40
    VITESSE_MIN = 20
    COEFF_FREINAGE = 1.8
    centre_vire = largeur_image // 2 

    i, j = np.indices((hauteur_image, largeur_image))
    
    print("Démarrage du Suiveur de Ligne. Ctrl+C pour arrêter.")
    #time.sleep(1) # Laisse le temps à l'utilisateur de poser le robot au sol

    derniere_commande = 0

    moteurs.piloter(40, 40)
    time.sleep(0.05)

    try:
        while True:
            # 2. Capture d'image
            frame = cam.get_frame()
            if frame is None:
                print("Erreur : Impossible de lire la caméra.")
                break


            frame_origine = frame.copy()

            #Flou gaussien pour réduire le bruit
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            

        

            # 3. Traitement d'image HSV
            """
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low_b = np.array([0, 0, 0], dtype=np.uint8)
            high_b = np.array([180, 255, 50], dtype=np.uint8)
            mask = cv2.inRange(hsv, low_b, high_b)
            """



            # 3. Traitement d'image (Niveaux de gris + Binarisation d'Otsu)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # La ligne noire devient blanche sur le masque grâce à THRESH_BINARY_INV
            seuil, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV ) #+ cv2.THRESH_OTSU

            mask[i + j < seuil] = 0
            mask[i + (largeur_image -1 - j) < seuil] = 0

            # On écrit le texte en blanc (255) en haut à gauche (x=10, y=20)
            texte_seuil = f"Seuil Otsu: {int(seuil)}"
            cv2.putText(mask, texte_seuil, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)


            #horizon = hauteur_image // 2
            #mask[0:horizon, :] = 0
            
            # Extraction des contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(c)
                est_un_croisement = w > (largeur_image * 0.70)

                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if est_un_croisement:
                        print("Croisement détecté ! On force le passage tout droit.")
                        # On ignore le PID et on trace droit pour traverser
                        commande = 0 
                        vitesse_base_dynamique = VITESSE_MAX
                        erreur = 0 # Pour l'affichage
                    else:
                        # --- TRAITEMENT NORMAL DU PID ---
                        erreur = cx - centre_vire
                        commande = pid.calculer(erreur)  # <-- UN SEUL APPEL ICI

                    # --- ON SUPPRIME LE DEUXIEME APPEL QUI ETAIT ICI ---

                    LIMITE_COMMANDE = 45
                    commande = max(min(commande, LIMITE_COMMANDE), -LIMITE_COMMANDE)

                    derniere_commande = commande  
                    
                    ralentissement = abs(erreur) * COEFF_FREINAGE
                    vitesse_base_dynamique = VITESSE_MAX - ralentissement

                    vitesse_base_dynamique = max(VITESSE_MIN, vitesse_base_dynamique)
                    
                    vitesse_gauche = vitesse_base_dynamique + commande
                    vitesse_droite = vitesse_base_dynamique - commande
                    moteurs.piloter(vitesse_gauche, vitesse_droite)                    
                    print(f"Err: {erreur:3d} | Base: {vitesse_base_dynamique:4.1f} | Cmd: {commande:5.1f} | Moteurs: G:{vitesse_gauche:5.1f} D:{vitesse_droite:5.1f}")
                    
                    # Éléments de dessin pour le debug visuel
                    #if AFFICHAGE_ACTIF:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                    cv2.drawMarker(frame, (centre_vire, hauteur_image // 2), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
            else:
                # --- NOUVELLE STRATÉGIE DE PERTE DE LIGNE ---
                print("Ligne Perdue ! Recherche active...")

                pid.reset()

                VITESSE_PIVOT = 22
                # Au lieu de s'arrêter, le robot pivote sur lui-même dans la dernière direction connue
                if derniere_commande > 0:
                    moteurs.piloter(VITESSE_PIVOT, -VITESSE_PIVOT)
                else:
                    moteurs.piloter(-VITESSE_PIVOT, VITESSE_PIVOT)

            enregistreur_video.write(frame)
            enregistreur_mask.write(mask)
            enregistreur_origine.write(frame_origine)

            # 6. Gestion de l'affichage sécurisée
            if AFFICHAGE_ACTIF:
                try:
                    cv2.imshow("Masque", mask)
                    cv2.imshow("Robot", frame)
                    cv2.imshow("Origine", frame_origine)
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
        enregistreur_origine.release() 
        
        # --- NOUVEAU : On sauvegarde la vidéo proprement ---
        enregistreur_video.release() 
        enregistreur_mask.release() 
        
        cv2.destroyAllWindows()
        print("Fermeture du programme et sauvegarde de la vidéo.")

if __name__ == "__main__":
    # Si le script est exécuté directement, on lance la boucle principale
    main()
    