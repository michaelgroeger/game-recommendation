########################################################
# List some proxy adresses and check whether they work #
########################################################
import requests

adresses = [
    "185.199.229.156:7492",
    "185.199.228.220:7300",
    "185.199.231.45:8382",
    "188.74.210.207:6286",
    "188.74.183.10:8279",
    "188.74.210.21:6100",
    "45.155.68.129:8133",
    "154.95.36.199:6893",
    "45.94.47.66:8110",
    "144.168.217.88:8780",
]

# Makes get request to services to see of the proxy works. Will return ip adress if it worked.
for i in adresses:
    print(f"on {i}")
    proxies = {
        "http": f"http://ijnczdof:eg5jvtvir6hi@{i}",
        "https": f"http://ijnczdof:eg5jvtvir6hi@{i}",
    }

    url = "https://api.ipify.org"

    try:
        r = requests.get(url, proxies=proxies).content
        print(r)
    except:
        print("Proxy does not work")
