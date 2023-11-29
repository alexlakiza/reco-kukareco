import os
import pickle
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette import status

from service.api.exceptions import UserNotFoundError
from service.log import app_logger
from ..recsys_models.custom_unpickler import load
from ..recsys_models.userknn import get_online_recs_for_user, \
    get_offline_recs_for_user


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class UnauthorizedMessage(BaseModel):
    """
    Ответ при ошибке 401 при отсутствии токена или неправильном токене
    """
    detail: str = "Bearer token missing or unknown"
    description: str = "Вы не указали Bearer token или указали неверный"


class ModelNotFoundMessage(BaseModel):
    """
    Ответ при ошибке 404 при неправильном имени модели
    """
    detail: str = "Model is not found"
    description: str = "Вы ввели неправильное имя модели рекомендательной " \
                       "системы"


class SuccessMessage(BaseModel):
    detail: str = "Вы успешно достучались до /health"


router = APIRouter()

get_bearer_token = HTTPBearer(auto_error=False)

# Чтение пиклов для сдачи Лабы 3
# Пикл с инстансом PopularModel()
with open("service/recsys_models/popular_model_20231128.pkl", "rb") as f:
    pop_model = pickle.load(f)

# Пикл с rectools.Dataset для использования PopularModel
with open("service/recsys_models/dataset_for_pop_model_20231128.pkl",
          "rb") as f:
    dataset = pickle.load(f)

# Пикл с инстансом UserKnn()
MODEL_PATH = "service/recsys_models/knn_tfidf_model_20231127.pkl"
if os.path.exists(MODEL_PATH):
    knn_model = load(MODEL_PATH)
else:
    knn_model = None

# Пикл с заранее подготовленными рекомендацями
# для всех пользвоателей из датасета
with open("service/recsys_models/knn_tfidf_model_offline_recos_20231129.pkl",
          "rb") as f:
    offline_knn_tfidf_recs = pickle.load(f)

# Пикл с 10 самыми популярными айтемами, чтобы рекомендовать
# их пользователям, которых не было в датасете
# (было посчитано в knn_experiments.ipynb)
with open("service/recsys_models/10_most_popular_items_20231129.pkl",
          "rb") as f:
    top_10_popular = pickle.load(f)


async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if auth is None or (token := auth.credentials) not in ['EHASMJLYDWHJKESU']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UnauthorizedMessage().detail,
        )
    return token


@router.get(
    path="/health",
    tags=["Health"],
    responses={status.HTTP_200_OK: {"model": SuccessMessage}}
)
async def health(token: str = Depends(get_current_user)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        status.HTTP_200_OK: {"model": RecoResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": UnauthorizedMessage},
        status.HTTP_404_NOT_FOUND: {"model": ModelNotFoundMessage}}
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: str = Depends(get_current_user)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    reco = list(range(k_recs))

    # models
    if model_name == "initial_recsys_model":
        # Lab 1
        reco = [_ * 3 for _ in range(k_recs)]
    elif model_name == "knn_tfidf_online_model":
        # Lab 3 (Онлайн предсказания)
        reco = get_online_recs_for_user(knn_model=knn_model,
                                        pop_model=pop_model,
                                        pop_model_dataset=dataset,
                                        top_10_pop_items=top_10_popular,
                                        user_id=user_id)
    elif model_name == "knn_tfidf_offline_model":
        # Lab 3 (Оффлайн предсказания)
        reco = get_offline_recs_for_user(
            knn_pop_recs_for_all_users=offline_knn_tfidf_recs,
            top_10_pop_items=top_10_popular,
            user_id=user_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ModelNotFoundMessage().detail,
        )

    return RecoResponse(user_id=user_id, items=reco)


@router.get(path="/hello",
            tags=["Test"])
async def hello_world() -> str:
    return "Hello, world!"


def add_views(app: FastAPI) -> None:
    app.include_router(router)
