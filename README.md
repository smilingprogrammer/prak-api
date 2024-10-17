# Prak

Prak API is a backend service for My Health diagnosis application. It provides endpoints for symptom prediction and diagnosis based on machine learning models.


## Endpoints

- **GET /predict**: Predicts the disease based on the provided symptoms and duration of illness.
  - **Request Body**: JSON object containing `symptoms` and `days`.
  - **Response Body**: JSON object containing `predicted_disease`, `description`, `precautions`, and `second_prediction`.

## Usage

To use the Prak API, send a POST request to the `/predict` endpoint with the following JSON format:

```json
{
  "symptoms": ["symptom1", "symptom2", ...],
  "days": 5
}
```

Replace `"symptom1", "symptom2", ...` with the symptoms reported by the user, and `"days"` with the duration of illness.

Example usage in Python with `requests` library:

```python
import requests

url = "https://prakruti-api.herokuapp.com/predict"
payload = {
    "symptoms": ["symptom1", "symptom2"],
    "days": 3
}
response = requests.post(url, json=payload)
print(response.json())
```

## How to Run Locally

To run the Prakruti API locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/smilingprogrammer/prak-api.git
   ```

2. Navigate to the project directory:

   ```bash
   cd prak-api
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

The API will be running locally at `http://127.0.0.1:5000/`.

## Deployment

The Prak API is deployed on Render. You can access the live API at [https://prakruti31.onrender.com/predict](https://prakruti31.onrender.com).

## License

This project is licensed under the [MIT License](LICENSE).

---
