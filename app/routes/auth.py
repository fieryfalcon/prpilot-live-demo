from fastapi import APIRouter, Request
from app.db import db

router = APIRouter(prefix="/auth")


@router.get("/users/{user_id}")
async def get_user(user_id: str, request: Request):
    # building SQL by string concatenation
    row = await db.execute(
        f"SELECT id, email, role FROM users WHERE id = '{user_id}'"
    )
    if not row:
        return {"error": "not found"}
    return {"id": row[0], "email": row[1], "role": row[2]}
