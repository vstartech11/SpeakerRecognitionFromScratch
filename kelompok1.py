import subprocess


def run_neural_net():
    subprocess.run(["python", "neural_net.py"])

def run_evaluation():
    subprocess.run(["python", "evaluation.py"])

if __name__ == "__main__":
    run_train = input("Apakah Anda ingin menjalankan training? (y/n): ").strip().lower()
    if run_train == 'y':
        run_neural_net()
    
    run_eval = input("Apakah Anda ingin menjalankan evaluasi? (y/n): ").strip().lower()
    if run_eval == 'y':
        run_evaluation()