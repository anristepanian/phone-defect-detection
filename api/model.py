import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)  # loading the model
    class_names = checkpoint["class_names"]  # loading the outputs

    model = models.resnet18(pretrained=False)  # loading the model structure
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, class_names


def predict_tensor_batch(model, batch_tensor):
    with torch.no_grad():  # we're not training the model, so no need to update the weights
        outputs = model(batch_tensor)
        preds = torch.argmax(outputs, dim=1)
# dim=1 is columns, so we chose the biggest logit from 4 columns (good, oil, scratch, stain) in every row
    return preds.cpu().numpy()
