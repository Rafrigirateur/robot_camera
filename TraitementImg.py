import asyncio
import cv2
import numpy as np
import time
from Hardware.camera import Camera
from Hardware.moteur import Moteur
from Pid import PID
import threading

# Configurable : Mets False pour désactiver le retour vidéo en SSH
AFFICHAGE_ACTIF = False  

async def main():
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
    ralentissement = 0
    position_marquage = "Aucun"
    active = False
    
    # Initialisation du PID (Coefficients à ajuster lors de tes tests !)
    pid = PID(kp=0.25, ki=0.0, kd=0.10)
    
    # Vitesse de croisière du robot (sur 100)
    VITESSE_MAX = 35
    VITESSE_MIN = 15
    COEFF_FREINAGE = 1.6
    TOLERANCE_ERREUR = 15
    centre_vire = largeur_image // 2 

    SEUIL_PLAFOND = 2  # Marge en pixels depuis le haut
    COEFF_FREINAGE_Y = 1.2 # Force du freinage vertical (à ajuster)
    COEFF_FREINAGE_CY = 0.8
    
    score = 0
    before = False
    SEUIL_AIRE_MARQUAGE = 200

    print("Démarrage du Suiveur de Ligne. Ctrl+C pour arrêter.")
    time.sleep(1) # Laisse le temps à l'utilisateur de poser le robot au sol

    derniere_commande = 0
    temps_arret_prevu = None  # <-- Le chronomètre pour la fin du parcours

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

            # Flou gaussien pour réduire le bruit
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # 3. Traitement d'image HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low_b = np.array([0, 0, 0], dtype=np.uint8)
            high_b = np.array([180, 255, 70], dtype=np.uint8)
            mask = cv2.inRange(hsv, low_b, high_b)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Extraction des contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                c = contours[0]

                x, y, w, h = cv2.boundingRect(c)
                est_un_croisement = w > (largeur_image * 0.70)

                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if est_un_croisement:
                        print("Croisement détecté ! On force le passage tout droit.")
                        commande = 0 
                        vitesse_base_dynamique = VITESSE_MAX
                        erreur = 0 
                    else:
                        # --- TRAITEMENT NORMAL DU PID ---
                        erreur = cx - centre_vire
                        commande = pid.calculer(erreur)  

                    LIMITE_COMMANDE = 65
                    commande = max(min(commande, LIMITE_COMMANDE), -LIMITE_COMMANDE)

                    derniere_commande = commande  
                    
                    ralentissement = 0
                    
                    # 1. Freinage lié à l'erreur (Gauche/Droite)
                    if abs(erreur) > TOLERANCE_ERREUR:
                        ralentissement += (abs(erreur) - TOLERANCE_ERREUR) * COEFF_FREINAGE
                    if y > SEUIL_PLAFOND:
                        ralentissement += (y - SEUIL_PLAFOND) * COEFF_FREINAGE_Y
                    # 2. Freinage lié au centre de masse
                    moitie_ecran_y = hauteur_image // 2
                    if cy > moitie_ecran_y:
                        ralentissement += pow(cy - moitie_ecran_y, 2) * 0.01 * COEFF_FREINAGE_CY
                        
                    vitesse_base_dynamique = VITESSE_MAX - ralentissement
                    vitesse_base_dynamique = max(VITESSE_MIN, vitesse_base_dynamique)
                    
                    vitesse_gauche = vitesse_base_dynamique + commande
                    vitesse_droite = vitesse_base_dynamique - commande
                    moteurs.piloter(vitesse_gauche, vitesse_droite)                    
                    print(f"Err: {erreur:3d} | Base: {vitesse_base_dynamique:4.1f} | Cmd: {commande:5.1f} | Moteurs: G:{vitesse_gauche:5.1f} D:{vitesse_droite:5.1f}")
                    
                    # Éléments de dessin pour le debug visuel
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                    cv2.drawMarker(frame, (centre_vire, hauteur_image // 2), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

                    active = False
                position_marquage = "Aucun"
                
                # S'il y a au moins 2 contours, on analyse le deuxième
                if len(contours) > 1:
                    c_marquage = contours[1]
                    aire_marquage = cv2.contourArea(c_marquage)
                    
                    # On vérifie que ce n'est pas juste du bruit visuel
                    if aire_marquage > SEUIL_AIRE_MARQUAGE:
                        # Calcul du centre du marquage
                        M_marq = cv2.moments(c_marquage)
                        if M_marq["m00"] != 0:
                            cx_marq = int(M_marq["m10"] / M_marq["m00"])
                            cy_marq = int(M_marq["m01"] / M_marq["m00"])

                            # Configuration et test de l'ellipse
                            centre_x_ellipse = largeur_image // 2
                            centre_y_ellipse = hauteur_image // 2
                            rayon_x = largeur_image // 2 
                            rayon_y = hauteur_image // 2 

                            test_ellipse = ((cx_marq - centre_x_ellipse)**2) / (rayon_x**2) + ((cy_marq - centre_y_ellipse)**2) / (rayon_y**2)

                            if test_ellipse <= 1:
                                active = True  # Le marquage est validé  
                            
                            # Comparaison : le marquage est-il à gauche ou à droite de la ligne ?
                            if cx_marq < cx:
                                position_marquage = "Gauche"
                            else:
                                position_marquage = "Droite"
                                
                            x_m, y_m, w_m, h_m = cv2.boundingRect(c_marquage)
                            cv2.rectangle(frame, (x_m, y_m), (x_m + w_m, y_m + h_m), (255, 0, 0), 2)
                            cv2.circle(frame, (cx_marq, cy_marq), 3, (255, 0, 0), -1)

                if active and before == False:
                    score += 1
                    before = True
                    print(f"!!! Marquage détecté à {position_marquage} !!! Score : {score}")
                elif not active:
                    before = False
            else:
                pass # Si on perd la ligne, le code continue simplement

            # --- CRÉATION DE LA GRILLE D'AFFICHAGE 2x2 ---
            frame_finale = np.zeros((hauteur_image * 2, largeur_image * 2, 3), dtype=np.uint8)
            mask_couleur = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            frame_finale[0:hauteur_image, 0:largeur_image] = frame_originale                   
            frame_finale[0:hauteur_image, largeur_image:largeur_image*2] = mask_couleur        
            frame_finale[hauteur_image:hauteur_image*2, 0:largeur_image] = frame               
            
            couleur_bordure = (150, 150, 150)
            epaisseur = 2

            cv2.line(frame_finale, (largeur_image, 0), (largeur_image, hauteur_image * 2), couleur_bordure, epaisseur)
            cv2.line(frame_finale, (0, hauteur_image), (largeur_image * 2, hauteur_image), couleur_bordure, epaisseur)

            texte_ligne1 = f"Err:{erreur:3d} | Cmd:{commande:3.0f}"
            texte_ligne2 = f"VG:{vitesse_gauche:3.0f} | VD:{vitesse_droite:3.0f}"
            texte_ligne3 = f"Bs: {vitesse_base_dynamique:3.0f} | rltr:{ralentissement:3.0f}"
            texte_ligne4 = f"cx:{cx:3d} | cy:{cy:3d}"
            texte_ligne5 = f"pt:{score:3d} | posM:{position_marquage}"
            
            decalage_x = largeur_image + 10
            cv2.putText(frame_finale, texte_ligne1, (decalage_x, hauteur_image + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_finale, texte_ligne2, (decalage_x, hauteur_image + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_finale, texte_ligne3, (decalage_x, hauteur_image + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame_finale, texte_ligne4, (decalage_x, hauteur_image + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
            cv2.putText(frame_finale, texte_ligne5, (decalage_x, hauteur_image + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            enregistreur_video.write(frame_finale)

            # Gestion de l'affichage sécurisée
            if AFFICHAGE_ACTIF:
                try:
                    cv2.imshow("Masque", mask)
                    cv2.imshow("Robot", frame_finale) 
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("Serveur graphique non détecté. Passage en mode headless automatisé.")
                    AFFICHAGE_ACTIF = False
            
            # --- GESTION DE LA LIGNE D'ARRIVÉE ---
            # 1. On vient d'atteindre 4 points, on lance le chronomètre (une seule fois)
            if score >= 4 and temps_arret_prevu is None:
                print("Ligne d'arrivée détectée ! Poursuite du suivi de ligne pendant 0.5s...")
                temps_arret_prevu = time.time() + 0.5  # Heure actuelle + 0.5 seconde
                
            # 2. On vérifie en permanence si le temps supplémentaire est écoulé
            if temps_arret_prevu is not None and time.time() >= temps_arret_prevu:
                print(f"\n🎉 Objectif Atteint ! Score final : {score} point(s). Arrêt complet du robot.")
                moteurs.piloter(0, 0)
                break  # On sort de la boucle, le bloc 'finally' prend le relais

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")
        
    finally:
        # Sécurité pour être sûr que les moteurs s'arrêtent
        moteurs.piloter(0, 0)
        
        # Relâchement propre du matériel
        moteurs.cleanup()
        cam.release()
        
        enregistreur_video.release() 
        cv2.destroyAllWindows()
        print("Fermeture du programme et sauvegarde de la vidéo.")

if __name__ == "__main__":
    asyncio.run(main())