import time, subprocess
for i in range(5):
    time.sleep(60)
    print(time.strftime("%H:%M"), subprocess.run("tail -c 120 /content/exp5/logs/conv2.log", shell=True, capture_output=True, text=True).stdout.replace("\n", " | ")[-120:], flush=True)
