import subprocess

def get_jobs_to_cancel(user="jrabault"):
    """Récupère les JOBID des jobs appartenant à l'utilisateur donné."""
    result = subprocess.run(["squeue", "-o", "%.18i %.9P %.40j %.8u"], capture_output=True, text=True)
    jobs = []
    
    for line in result.stdout.splitlines()[1:]:  # Ignorer l'en-tête
        parts = line.split()
        if len(parts) >= 4 and parts[3] == user:
            jobs.append(parts[0])  # JOBID
    
    return jobs

def cancel_jobs(jobs):
    """Annule les jobs donnés en utilisant scancel."""
    for job in jobs:
        subprocess.run(["scancel", job])
        print(f"Job {job} annulé.")

def main():
    jobs = get_jobs_to_cancel()
    if jobs:
        cancel_jobs(jobs)
    else:
        print("Aucun job à annuler.")

if __name__ == "__main__":
    main()
