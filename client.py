
import requests
# The URL for the predict route
url = 'http://127.0.0.1:5000/predict'
# Example input features
data = {
    'features': 'I had a good day'
}
# Send a POST request to the server
response = requests.post(url, json=data)
# Print the prediction result
print(response.text)