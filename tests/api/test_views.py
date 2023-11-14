from http import HTTPStatus

from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"


def test_health(
    client: TestClient,
) -> None:
    """
    Проверка доступа к /health
    """
    valid_token = "EHASMJLYDWHJKESU"

    with client:
        response = client.get("/health", headers={
            "Authorization": f"Bearer {valid_token}"})
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    """
    Проверка работы эндпоинта для модели
    """
    user_id = 123
    valid_token = "EHASMJLYDWHJKESU"
    path = GET_RECO_PATH.format(model_name="initial_recsys_model",
                                user_id=user_id)
    with client:
        response = client.get(path, headers={
            "Authorization": f"Bearer {valid_token}"})
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    client: TestClient,
) -> None:
    """
    Проверка наличия ошибки 404 при неправильном id пользователя
    """
    valid_token = "EHASMJLYDWHJKESU"
    user_id = 10 ** 10
    path = GET_RECO_PATH.format(model_name="initial_recsys_model",
                                user_id=user_id)
    with client:
        response = client.get(path, headers={
            "Authorization": f"Bearer {valid_token}"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_not_found_error(
    client: TestClient,
) -> None:
    """
    Предусмотреть, чтобы сервис возвращал 404 ошибку, если задано
    неверное имя модели (1 балл) + покрыть тестами (1 балл)
    """
    valid_token = "EHASMJLYDWHJKESU"
    incorrect_path = GET_RECO_PATH.format(model_name="bla_bla_model",
                                          user_id=321)

    correct_path = GET_RECO_PATH.format(model_name="initial_recsys_model",
                                        user_id=321)

    with client:
        bad_response = client.get(incorrect_path, headers={
            "Authorization": f"Bearer {valid_token}"})
        good_response = client.get(correct_path, headers={
            "Authorization": f"Bearer {valid_token}"})

    assert bad_response.status_code == HTTPStatus.NOT_FOUND
    assert good_response.status_code == HTTPStatus.OK


def test_authorization_with_any_token(
    client: TestClient,
) -> None:
    """
    Добавить аутентификацию (1 балл) + тесты на нее (1 балл)
    (Проверка правильного/неправильного токена)
    """
    valid_token = "EHASMJLYDWHJKESU"
    invalid_token = "BLABLATOKEN"

    with client:
        good_response = client.get("/reco/initial_recsys_model/432", headers={
            "Authorization": f"Bearer {valid_token}"})

        bad_response = client.get(
            "/reco/initial_recsys_model/432",
            headers={"Authorization": f"Bearer {invalid_token}"}
        )

    assert good_response.status_code == HTTPStatus.OK
    assert bad_response.status_code == HTTPStatus.UNAUTHORIZED


def test_authorization_with_empty_token(
    client: TestClient,
) -> None:
    """
    Добавить аутентификацию (1 балл) + тесты на нее (1 балл)
    (Проверка отсутствующего токена)
    """
    with client:
        response = client.get("/reco/initial_recsys_model/432")

    assert response.status_code == HTTPStatus.UNAUTHORIZED
