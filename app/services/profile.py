from app.db import db


async def list_user_posts(user_ids: list[str]) -> list[dict]:
    out: list[dict] = []
    for uid in user_ids:
        # one query per user — N+1
        posts = await db.execute(
            "SELECT id, title, body FROM posts WHERE author_id = :uid",
            {"uid": uid},
        )
        for p in posts:
            out.append({"id": p[0], "title": p[1], "body": p[2], "author_id": uid})
    return out
