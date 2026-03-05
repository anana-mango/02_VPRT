from typing import Any, List, Tuple
from tabulate import tabulate

def render_table(cols: List[str], rows: List[Tuple[Any, ...]], max_rows: int = 50) -> str:
    shown = rows[:max_rows]
    if not cols:
        # SELECT가 아니거나(혹은 컬럼정보가 없는 경우) fallback
        return f"(no columns) rows={len(rows)}"

    return tabulate(shown, headers=cols, tablefmt="github")