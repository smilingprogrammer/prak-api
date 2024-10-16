import requests

url = "https://prakruti-api.herokuapp.com/predict"
payload = {
    "symptoms": ["symptom1", "symptom2"],
    "days": 3
}
response = requests.post(url, json=payload)
print(response.json())