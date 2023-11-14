from typing import List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from starlette import status

from service.api.exceptions import UserNotFoundError
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    # You can add your own logic to validate the token here
    if token != "EHASMJLYDWHJKESU":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@router.get(
    path="/health",
    tags=["Health"],
)
async def health(token: str = Depends(get_current_user)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request, model_name: str, user_id: int,
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
        raise HTTPException(status_code=404, detail="Incorrect model name")

    return RecoResponse(user_id=user_id, items=reco)


@router.get(path="/hello", tags=["test"])
async def hello_world() -> str:
    return "Hello, world!"


def add_views(app: FastAPI) -> None:
    app.include_router(router)
