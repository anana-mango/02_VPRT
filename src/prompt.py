from typing import List, Dict

def build_sql_messages(user_request: str, schema_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    schema_block = "\n\n".join(d["text"] for d in schema_docs)

    system = (
        "You are a Text-to-SQL assistant for SQLite.\n"
        "You MUST follow the provided schema context.\n"
        "Return ONLY valid JSON with keys: sql, assumptions.\n"
        "Rules:\n"
        "- Put SQL string in json.sql.\n"
        "- Prefer SELECT unless the user explicitly requests write operations.\n"
        "- Add LIMIT 50 if many rows might be returned.\n"
        "- Do NOT invent tables/columns not present in schema context.\n"
        "- SQLite dialect.\n"
    )

    user = (
        "### DATABASE SCHEMA (retrieved context)\n"
        f"{schema_block}\n\n"
        "### USER REQUEST (Korean)\n"
        f"{user_request}\n"
    )

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]