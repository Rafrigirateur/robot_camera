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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    date_heure = time.strftime("%Y-%m-%d_%H-%M")
    nom_fichier = f"log_robot_{date_heure}.avi"
    
    enregistreur_video = cv2.VideoWriter(nom_fichier, fourcc, 30.0, (largeur_image * 2, hauteur_image * 2))
    print(f"La vidéo sera sauvegardée sous : {nom_fichier}")
    
    # Variables par défaut pour l'affichage au cas où la ligne est perdue dès le début
    erreur, commande, vitesse_gauche, vitesse_droite, vitesse_base_dynamique = 0, 0, 0, 0, 0
    cx, cy = 0, 0
    
    
    # Initialisation du PID (Coefficients à ajuster lors de tes tests !)
    # Règle d'abord Kp (ex: 0.4), laisse Ki à 0, et mets un poil de Kd (ex: 0.05)
    pid = PID(kp=0.12, ki=0.0, kd=0.05)
    
    # Vitesse de croisière du robot (sur 100)
    VITESSE_MAX = 45
    VITESSE_MIN = 15
    COEFF_FREINAGE = 1.6
    TOLERANCE_ERREUR = 15
    centre_vire = largeur_image // 2 


    SEUIL_PLAFOND = 2  # Marge en pixels depuis le haut
    COEFF_FREINAGE_Y = 1.2 # Force du freinage vertical (à ajuster)

    COEFF_FREINAGE_CY = 1.5
    
    print("Démarrage du Suiveur de Ligne. Ctrl+C pour arrêter.")
    time.sleep(1) # Laisse le temps à l'utilisateur de poser le robot au sol

    derniere_commande = 0

    moteurs.piloter(40, 40)
    time.sleep(0.2)

    try:
        while True:
            # 2. Capture d'image
            frame = cam.get_frame()
            if frame is None:
                print("Erreur : Impossible de lire la caméra.")
                break

            frame_originale = frame.copy()

            #Flou gaussien pour réduire le bruit
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # 3. Traitement d'image HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low_b = np.array([0, 0, 0], dtype=np.uint8)
            high_b = np.array([180, 255, 50], dtype=np.uint8)
            mask = cv2.inRange(hsv, low_b, high_b)

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

                    LIMITE_COMMANDE = 60
                    commande = max(min(commande, LIMITE_COMMANDE), -LIMITE_COMMANDE)

                    derniere_commande = commande  
                    
                    ralentissement = 0
                    
                    # 1. Freinage lié à l'erreur (Gauche/Droite)
                    if abs(erreur) > TOLERANCE_ERREUR:
                        ralentissement += (abs(erreur) - TOLERANCE_ERREUR) * COEFF_FREINAGE
                    if y > SEUIL_PLAFOND:
                        ralentissement += (y - SEUIL_PLAFOND) * COEFF_FREINAGE_Y
                    # Freinage lié au centre de masse (point bleu)
                    moitie_ecran_y = hauteur_image // 2
                    if cy > moitie_ecran_y:
                        # Plus le point bleu descend sous la moitié de l'écran, plus on freine fort
                        ralentissement += pow(cy - moitie_ecran_y, 2) * COEFF_FREINAGE_CY
                        
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

            # --- CRÉATION DE LA GRILLE D'AFFICHAGE 2x2 ---
            # 1. On crée le canevas noir global (320x240 pixels)
            frame_finale = np.zeros((hauteur_image * 2, largeur_image * 2, 3), dtype=np.uint8)
            
            # 2. On convertit le masque (1 canal) en image (3 canaux) pour l'assemblage
            mask_couleur = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 3. On colle les images dans leurs quadrants respectifs
            frame_finale[0:hauteur_image, 0:largeur_image] = frame_originale                   # Haut Gauche
            frame_finale[0:hauteur_image, largeur_image:largeur_image*2] = mask_couleur        # Haut Droite
            frame_finale[hauteur_image:hauteur_image*2, 0:largeur_image] = frame               # Bas Gauche (avec contours)
            
            couleur_bordure = (150, 150, 150) # Gris clair
            epaisseur = 2

            cv2.line(frame_finale, (largeur_image, 0), (largeur_image, hauteur_image * 2), couleur_bordure, epaisseur)
            cv2.line(frame_finale, (0, hauteur_image), (largeur_image * 2, hauteur_image), couleur_bordure, epaisseur)

            # 4. Textes et variables pour le quadrant Bas Droite (qui reste noir)
            texte_ligne1 = f"Err:{erreur:3d} | Cmd:{commande:3.0f}"
            texte_ligne2 = f"VG:{vitesse_gauche:3.0f} | VD:{vitesse_droite:3.0f}"
            texte_ligne3 = f"Base: {vitesse_base_dynamique:3.0f}"
            texte_ligne4 = f"cx:{cx:3d} | cy:{cy:3d}"
            texte_ligne5 = f"ralent:{ralentissement:3.0f}"
            
            # On décale les coordonnées d'écriture vers la droite et le bas
            decalage_x = largeur_image + 10
            cv2.putText(frame_finale, texte_ligne1, (decalage_x, hauteur_image + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_finale, texte_ligne2, (decalage_x, hauteur_image + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_finale, texte_ligne3, (decalage_x, hauteur_image + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame_finale, texte_ligne4, (decalage_x, hauteur_image + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
            cv2.putText(frame_finale, texte_ligne5, (decalage_x, hauteur_image + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # On enregistre la frame modifiée au lieu de l'originale
            enregistreur_video.write(frame_finale)

            # 6. Gestion de l'affichage sécurisée
            if AFFICHAGE_ACTIF:
                try:
                    cv2.imshow("Masque", mask)
                    cv2.imshow("Robot", frame_finale) # On affiche la nouvelle frame
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
        
        # --- NOUVEAU : On sauvegarde la vidéo proprement ---
        enregistreur_video.release() 
        
        cv2.destroyAllWindows()
        print("Fermeture du programme et sauvegarde de la vidéo.")

if __name__ == "__main__":
    # Si le script est exécuté directement, on lance la boucle principale
    main()
    