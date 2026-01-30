import zipfile
from PIL import Image
from flask import Flask, request, jsonify

import torch
import torch.nn as nn
from torchvision import models, transforms

model_path = "../model/model.pth"  # accessing the model
image_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # we're running on CPU anyway

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # maximum content length is set to 50MB

checkpoint = torch.load(model_path, map_location=device)  # loading the model
class_names = checkpoint["class_names"]  # loading the outputs

model = models.resnet18(pretrained=False)  # loading the model structure
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

inference_transform = transforms.Compose([  # transforming the data
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_tensor_batch(batch_tensor):
    with torch.no_grad():  # we're not training the model, so no need to update the weights
        outputs = model(batch_tensor)
        preds = torch.argmax(outputs, dim=1)
# dim=1 is columns, so we chose the biggest logit from 4 columns (good, oil, scratch, stain) in every row
    return preds.cpu().numpy()


def allowed_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))  # allowed file formats


@app.route("/health", methods=["GET"])  # route for the check
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
# route when user wants to make a prediction for one image only
def predict_single():
    if "image" not in request.files:
        return jsonify({"error: No image provided"}), 400

    file = request.files["image"]

    if not allowed_image(file.filename):
        return jsonify({"error: Invalid file type"}), 400

    image = Image.open(file.stream).convert("RGB")
    tensor = inference_transform(image).unsqueeze(0).to(device)

    pred_idx = predict_tensor_batch(tensor)[0]

    return jsonify({
        "prediction": class_names[pred_idx]
    })


@app.route("/predict_batch", methods=["POST"])
# when user wants to make predictions for batch of images at once
def predict_batch():
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
    preds = predict_tensor_batch(batch)

    results = {
        fname: class_names[pred]
        for fname, pred in zip(filenames, preds)
    }

    return jsonify(results)


if __name__ == "__main__":
    app.run()
