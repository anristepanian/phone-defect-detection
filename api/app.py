import zipfile
from PIL import Image
from flask import Flask, request, jsonify
import torch
from torchvision import transforms

from api.model import load_model, predict_tensor_batch
from pathlib import Path

image_size = 224
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "model" / "model.pth"  # path to a model


def create_app(testing=False):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # maximum content length is set to 50MB

    device = torch.device("cpu")

    if not testing:
        model, class_names = load_model(model_path, device)
    else:
        model = None
        class_names = ["dummy"]

    inference_transform = transforms.Compose([  # transforming the data
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def allowed_image(filename):  # allowed file formats
        return filename.lower().endswith((".jpg", ".jpeg", ".png"))

    @app.route("/health", methods=["GET"])  # route for the check
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    # route when user wants to make a prediction for one image only
    def predict_single():
        if testing:
            return jsonify({"prediction": "dummy"})

        if "image" not in request.files:
            return jsonify({"error: No image provided"}), 400

        file = request.files["image"]
        if not file or not allowed_image(file.filename):
            return jsonify({"error": "Invalid image"}), 400

        image = Image.open(file.stream).convert("RGB")
        tensor = inference_transform(image).unsqueeze(0).to(device)
        pred_idx = predict_tensor_batch(model, tensor)[0]

        return jsonify({"prediction": class_names[pred_idx]})

    @app.route("/predict_batch", methods=["POST"])
    # when user wants to make predictions for batch of images at once
    def predict_batch():
        if testing:
            return jsonify({"prediction": "dummy"})

        if "file" not in request.files:
            return jsonify({"error: No ZIP file provided"}), 400

        zip_file = request.files["file"]

        images = []
        filenames = []

        with zipfile.ZipFile(zip_file.stream) as z:
            for name in z.namelist():
                if allowed_image(name):
                    img = Image.open(z.open(name)).convert("RGB")
                    images.append(inference_transform(img))
                    filenames.append(name)

        if not images:
            return jsonify({"error: No valid images in ZIP"}), 400

        batch = torch.stack(images).to(device)
        preds = predict_tensor_batch(model, batch)

        results = {
            fname: class_names[pred]
            for fname, pred in zip(filenames, preds)
        }

        return jsonify(results)

    return app


app = create_app()

if __name__ == "__main__":
    app.run()
