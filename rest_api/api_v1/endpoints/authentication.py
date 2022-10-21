from datetime import timedelta

from axiom_client.client import Axiom
from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

from schemas.authenticator import Token
from services.authenticator import authenticate_user, create_access_token
from core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(settings.FAKE_USER_DB, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


class ServicesWrapper:
    def __init__(
            self,
            axiom_email: str,
            axiom_password: str
    ):
        self.axiom_email = axiom_email
        self.axiom_password = axiom_password
        self.client = Axiom(base_url="https://axiom.dev.ftech.ai")

    def auth_axiom(self):
        if self.axiom_email and self.axiom_password:
            if self.client.login(email=self.axiom_email, password=self.axiom_password):
                return True
        return False


