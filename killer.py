import psutil
import time
from playsound import playsound

def check_and_kill_stockfish(interval_minutes, threshold_cpu_time):
    process_name = "stockfish"
    nb = 0

    while True:
        # Liste des processus en cours d'exécution
        for process in psutil.process_iter(['pid', 'name', 'cpu_times']):
            if process_name.lower() in process.info['name'].lower():
                cpu_time = process.info['cpu_times'].user + process.info['cpu_times'].system

                if cpu_time > threshold_cpu_time:
                    # Tuer le processus s'il a utilisé plus de temps CPU que le seuil
                    psutil.Process(process.info['pid']).terminate()
                    nb+=1
                    playsound('oyasumi.mp3')
                    print('𐐘💥╾━╤デ╦︻ඞා ', nb)

        # Attendre X minutes avant de vérifier à nouveau
        time.sleep(interval_minutes * 60)

# Paramètres du script
interval_minutes = 3  # Répétition toutes les X minutes
threshold_cpu_time = 600  # Seuil de temps CPU en secondes

# Appel de la fonction avec les paramètres
check_and_kill_stockfish(interval_minutes, threshold_cpu_time)
