from pathlib import Path

__all__ = ["scan_subsets", "str2nmb"]

def scan_subsets(
    root_dir: Path,
    included_keywords: list[str] | None = None,
    excluded_keywords: list[str] | None = None,
    depth: int = 3,
) -> list[Path]:
    excluded_keywords = excluded_keywords or []
    included_keywords = included_keywords or []

    dirs = Path(root_dir).resolve().glob("/".join(["*"] * depth))
    dirs = filter(
        lambda x: all([str(x).find(keyword) != -1 for keyword in included_keywords]),
        dirs,
    )
    dirs = filter(
        lambda x: all([str(x).find(keyword) == -1 for keyword in excluded_keywords]),
        dirs,
    )
    dirs = sorted(list(filter(lambda x: x.is_dir(), dirs)))
    return dirs

def str2nmb(string: str) -> float | int:
    string_filtered = "".join(c for c in string if c.isnumeric() or c in [".", "-"])
    return (
        -1
        if string_filtered == ""
        else (
            int(string_filtered)
            if string_filtered.find(".") == -1
            else float(string_filtered)
        )
    )
