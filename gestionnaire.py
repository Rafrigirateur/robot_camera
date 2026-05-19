import RPi.GPIO as GPIO
import subprocess
import time
import signal

# Configuration de la broche (Changer si tu utilises une autre GPIO)
PIN_INTERRUPTEUR = 26 

# Initialisation du GPIO
GPIO.setmode(GPIO.BCM)
# On active la résistance de Pull-Up interne
GPIO.setup(PIN_INTERRUPTEUR, GPIO.IN, pull_up_down=GPIO.PUD_UP)

processus_robot = None

print("=== Gestionnaire de l'interrupteur prêt ===")
print("En attente de l'activation...")

try:
    while True:
        # Lecture de l'état (0 = Fermé/ON, 1 = Ouvert/OFF)
        etat_interrupteur = GPIO.input(PIN_INTERRUPTEUR)
        
        # CAS 1 : L'interrupteur est sur ON et le robot ne tourne pas encore
        if etat_interrupteur == GPIO.LOW and processus_robot is None:
            print("\n[ON] Interrupteur activé ! Lancement de TraitementImg...")
            
            # On lance le script dans un processus séparé
            processus_robot = subprocess.Popen(["python3", "TraitementImg.py"])
            
            # Anti-rebond basique pour éviter les déclenchements fous
            time.sleep(1) 
            
        # CAS 2 : L'interrupteur est remis sur OFF et le robot est en train de tourner
        elif etat_interrupteur == GPIO.HIGH and processus_robot is not None:
            print("\n[OFF] Interrupteur désactivé ! Arrêt propre du robot...")
            
            # On envoie un SIGINT (l'équivalent exact d'un Ctrl+C)
            processus_robot.send_signal(signal.SIGINT)
            
            # On attend que TraitementImg finisse son bloc 'finally' (sauvegarde vidéo, stop moteurs)
            processus_robot.wait() 
            
            print("Robot arrêté avec succès.")
            processus_robot = None
            time.sleep(1)

        # Petite pause pour ne pas saturer le processeur de la Pi
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nArrêt du gestionnaire par l'utilisateur (Ctrl+C).")

finally:
    # Sécurité : Si on coupe le gestionnaire alors que le robot tourne, on l'éteint
    if processus_robot is not None:
        print("Fermeture de secours du robot...")
        processus_robot.send_signal(signal.SIGINT)
        processus_robot.wait()
    
    GPIO.cleanup()
    print("GPIO nettoyés. Fin du programme.")