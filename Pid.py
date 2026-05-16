import time

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.derniere_erreur = 0
        self.somme_erreurs = 0
        self.dernier_temps = time.time()
        
    def calculer(self, erreur):
        temps_actuel = time.time()
        dt = temps_actuel - self.dernier_temps
        
        # Sécurité pour éviter la division par zéro
        if dt <= 0:
            dt = 0.001
            
        # 1. Terme Proportionnel
        P = self.kp * erreur
        
        # 2. Terme Intégral (avec anti-windup de base pour éviter l'emballement)
        self.somme_erreurs += erreur * dt
        self.somme_erreurs = max(min(self.somme_erreurs, 100), -100) 
        I = self.ki * self.somme_erreurs
        
        # 3. Terme Dérivé
        D = self.kd * ((erreur - self.derniere_erreur) / dt)
        
        # Sauvegarde pour le prochain cycle
        self.derniere_erreur = erreur
        self.dernier_temps = temps_actuel
        
        # La commande finale est la somme des 3 composantes
        return P + I + D
        
    def reset(self):
        self.derniere_erreur = 0
        self.somme_erreurs = 0
        self.dernier_temps = time.time()