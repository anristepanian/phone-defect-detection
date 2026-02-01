from api.app import create_app


def test_health():
    app = create_app(testing=True)
    client = app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
