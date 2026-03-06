import random
import time
import requests

url = "https://cnc-machine-8664d-default-rtdb.asia-southeast1.firebasedatabase.app/cnc_machine.json"

while True:

    data = {
        "rpm": random.randint(1100,1400),
        "temperature": random.randint(60,80),
        "vibration": round(random.uniform(1.0,3.0),2),
        "current": round(random.uniform(7.0,10.0),2),
        "sound": random.randint(40,70),
        "timestamp": int(time.time())
    }

    requests.patch(url, json=data)

    print("Sent:", data)

    time.sleep(5)