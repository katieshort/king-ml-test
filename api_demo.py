import requests

# URL of the Flask API
url = "http://127.0.0.1:5000/predict"


data = [
    {
        "id": "33553",
        "c1": "VFc",
        "c2": "aW9z",
        "c3": "cHQ",
        "c4": "aHVhd",
        "c5": True,
        "c6": "LTAzOjAw",
        "n1": 303.488539,
        "n2": 1085.434206,
        "n3": 90.097357,
        "n4": 21.113713,
        "n5": 404.121039,
        "n6": 3859250336.874662,
        "n7": 10.312303,
        "n8": 1112.885066,
        "n10": 0.409991,
        "n11": 3.289646,
        "n12": 0.08629,
    },
    {
        "id": "33553",
        "c1": "VFc",
        "c2": "aW9z",
        "c3": "cHQ",
        "c4": "aHVhd",
        "c5": True,
        "c6": "LTAzOjAw",
        "n1": 303.488539,
        "n2": 1085.434206,
        "n3": 90.097357,
        "n4": 21.113713,
        "n5": 404.121039,
        "n6": 3859250336.874662,
        "n7": 10.312303,
        "n8": 1112.885066,
        "n10": 0.409991,
        "n11": 3.289646,
        "n12": 0.08629,
    },
]

# Send a POST request
response = requests.post(url, json=data)

# Print the response from the server
print("Status Code:", response.status_code)
print("JSON Response:", response.json())
