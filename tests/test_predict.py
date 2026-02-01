from api.app import create_app


def test_predict_dummy():
    app = create_app(testing=True)
    client = app.test_client()

    response = client.post("/predict")
    assert response.status_code == 200
