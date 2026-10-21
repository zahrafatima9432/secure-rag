"""FastAPI app. Run with:  uvicorn secure_rag.api:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg.errors import InsufficientPrivilege
from pydantic import BaseModel, Field

from .auth import AuthError, principal_from_token
from .cache import SemanticCache
from .config import get_settings
from .db import make_app_pool
from .models import make_embedder, make_generator, make_reranker
from .principal import Principal
from .service import RAGService, UnknownTenant

bearer = HTTPBearer(auto_error=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET must be set")
    pool = make_app_pool(settings)
    app.state.settings = settings
    app.state.service = RAGService(
        settings, pool, make_embedder(settings), make_generator(settings), make_reranker(settings),
        SemanticCache(settings.cache_threshold) if settings.cache_enabled else None,
    )
    yield
    pool.close()


app = FastAPI(title="secure-rag", lifespan=lifespan)


def current_principal(
    request: Request, creds: HTTPAuthorizationCredentials = Depends(bearer)
) -> Principal:
    try:
        return principal_from_token(creds.credentials, request.app.state.settings.jwt_secret)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}") from e


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    chunk_id: int


class QueryOut(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]
    cached: bool


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/me")
def me(p: Principal = Depends(current_principal)) -> dict:
    return {"user_id": p.user_id, "tenant_id": p.tenant_id, "groups": list(p.groups)}


@app.post("/query", response_model=QueryOut)
def query(body: QueryIn, request: Request, p: Principal = Depends(current_principal)) -> QueryOut:
    try:
        ans = request.app.state.service.answer(p, body.question, body.k)
    except UnknownTenant as e:
        raise HTTPException(status_code=403, detail="unknown tenant") from e
    except InsufficientPrivilege as e:
        # The audit tripwire fired (e.g. access revoked mid-request). Fail closed.
        raise HTTPException(status_code=409, detail="permissions changed during request; retry") from e
    return QueryOut(request_id=ans.request_id, answer=ans.answer,
                    sources=[Source(**s) for s in ans.sources], cached=ans.cached)
