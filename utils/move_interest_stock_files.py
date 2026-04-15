from pathlib import Path
import shutil
import re

"""
디렉토리 하나에 모든 파일이 있음 > 년/월/일 디렉토리에 적재시키는 스크립트
"""

# 정리할 대상 폴더
BASE_DIR = Path(r"F:\interest_stocks")


def extract_date_parts(filename: str):
    """
    파일명 앞의 YYYYMMDD 추출
    예: 20250908 금강공업 [014280].txt -> ('2025', '09', '08')
    """
    match = re.match(r"^(\d{4})(\d{2})(\d{2})", filename)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def organize_files(base_dir: Path):
    for item in base_dir.iterdir():
        if not item.is_file():
            continue

        date_parts = extract_date_parts(item.name)
        if not date_parts:
            print(f"건너뜀: {item.name}")
            continue

        year, month, day = date_parts
        target_dir = base_dir / year / month / day
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / item.name
        shutil.move(str(item), str(target_path))
        print(f"이동: {item.name} -> {target_path}")


if __name__ == "__main__":
    organize_files(BASE_DIR)