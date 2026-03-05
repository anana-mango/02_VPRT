sql_val = obj.get("sql")
assumptions_val = obj.get("assumptions")

# sql: 반드시 문자열이어야 함
if not isinstance(sql_val, str):
    raise ValueError(f"json.sql must be a string, got {type(sql_val)}")
sql = sql_val.strip()

# assumptions: str 또는 list[str] 또는 None 모두 허용
assumptions = ""
if isinstance(assumptions_val, str):
    assumptions = assumptions_val.strip()
elif isinstance(assumptions_val, list):
    # 리스트 안 요소들을 문자열로 합치기
    assumptions = "\n".join(str(x).strip() for x in assumptions_val if str(x).strip())
elif assumptions_val is None:
    assumptions = ""
else:
    # dict/int 등 예상 밖이면 그냥 문자열로 변환
    assumptions = str(assumptions_val).strip()