from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette import status

from service.api.exceptions import UserNotFoundError
from service.log import app_logger


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

    # Write your code here
    if model_name == "initial_recsys_model":
        reco = [_ * 3 for _ in range(k_recs)]
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
