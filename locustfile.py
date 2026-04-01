from locust import HttpUser, task, between
import os

class PredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        image_path = "potato_dataset/Potato___Early_blight/26d0502d-5fb3-48fa-a86d-d1de92703285___RS_Early.B 6959.JPG"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                self.client.post("/predict", files={"file": f})

    @task(1)
    def health(self):
        self.client.get("/health")
