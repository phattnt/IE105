# Runs IN the kernel cell: starts run_inner.py detached (kernel stays free for keepalive cells), unless already running.
import subprocess, sys, os
busy = subprocess.run("pgrep -f '[r]un_inner.py|[t]rain_lines.py|[e]val_primevul.py'", shell=True, capture_output=True, text=True).stdout.split()
if busy:
    print("already running", busy, flush=True)
else:
    subprocess.Popen(f"nohup {sys.executable} /content/exp5/run_inner_conv2.py > /content/exp5/logs/run_inner_conv2.out 2>&1 &", shell=True, start_new_session=True)
    print("started detached", flush=True)
