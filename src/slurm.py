import subprocess
import os


def show_this_job():
    job_id = os.getenv("SLURM_JOB_ID", None)

    if job_id is None:
        return None

    ret = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True)

    job_info = {}
    for line in ret.stdout.splitlines():
        line = line.strip()
        if line:
            # Handle multi-entry lines (like JobId=X JobName=Y)
            entries = line.split()
            for entry in entries:
                if "=" in entry:
                    # There are some entries that have multiple = signs, so this only splits once
                    key, value = entry.split("=", 1)
                    job_info[key] = value

    return job_info
