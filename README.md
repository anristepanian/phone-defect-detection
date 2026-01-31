# Phone Defect Detection Model / Project

An online shopping platform for smartphones is growing rapidly. Despite the desirable development of the company, this also comes with new challenges. As one example, as more and more smartphones are ordered via the platform, the number of phones sent back for refund also increases. The workforce that is necessary to categorize the re-shipped smarphones has increased in the last month beyond a viable amount. As the newly employed data scientist of the company is assigned the task of developing a machine learning model that automatically classifies the refund items into categories based on pictures of the items. The model should run as a service that can be triggered in batches overnight. This project hopes to reduce the workforce and costs for manually sorting items. Instead, incoming smartphones will automatically be categorized daily.

## Criteria

- The phone defect detection model was built using `Python version 3.11.14`.
- The code was documented using the `Sphinx (reStructuredText)` style.
- The imperative programming was used to create the phone defect detection model and the Flask API.
- `PyTorch` framework was used in order to build and train the ML model.
- `Flask` framework was used in order to create a REST API.
- `Jupyter Notebook` was used as an IDE during creation and training of the model.
- `PyCharm` was used as an IDE during creation of a REST API.

## ML Model

When passed a photo (.jpg, .jpeg, .png) or a ZIP file containing photos of phones, the model categorizes them into 4 ctegories:
1. Good
   - The phone is in acceptable condition.
   - No visible defects or only negligible ones.
2. Oil
   - Oil, grease, or fingerprint smudges on the surface (often the screen or camera).
   - Typically caused by handling; usually removable but affects visual quality.
3. Scratch
   - Visible scratches on the screen, body, or camera lens.
   - Indicates physical wear or damage.
4. Stain
   - dead or stuck pixel clusters on the screen

## User Actions

- A user can either load a single smartphone image of next allowed formats - .jpg, .jpeg, .png, or load a ZIP file containing batch of images of allowed formats.

## Future Enhancements

- The dataset size should be increased. Because of small size of a training dataset, the phone defect detection model is still inaccurate. The system often misclassifies images uploaded by users.
- CT (continious training) pipeline should be implemented in order to automatically renew the training of the machine learning model over time.
- Add confidence & uncertanity + explainability & defect localization. For instance, the output should look like:
  ``` Command line
  Prediction: "Scratch"
  Confidence: 87%
  "Model predicted scratch because it focused on the top left corner of the phone"
  ```
- Web UI. Develop a simple and interactive Web UI instead of using apps like `Postman`.
- Mobile deployment (Android, IOS, etc.).
- Anomaly detection. If there are multiple defects on one image, or perhaps there is a defect which is not in the category.

## Installation

```commandline
pip install -r requirements.txt
```

## Usage

Start

```commandline
python api/app.py
```

then open any browser and paste http://127.0.0.1:5000/health, you should see next:
``` command line
{"status":"ok"}
```
this shows that application runs correctly.
Then open `Postman`, go to **Get data** and choose _POST_.
Now insert http://127.0.0.1:5000/predict if you want to make a prediction just on 1 image.
Or insert http://127.0.0.1:5000/predict_batch if you want to make a prediction on group of images.
Finally go to _Body_ and load an image or a group of images and click **Send**
That's it! You will see the prediction in the bottom.<br>
If you loaded a ZIP file with group of images you should get:
``` command line
{
    "1.jpg": "stain",
    "2.jpg": "oil",
    "3.jpg": "oil",
    "4.jpg": "oil"
}
```
Where you will see the name of an image and the prediction next to it.

## Authors

The project was developed by [Anri Stepanian](https://github.com/anristepanian)
