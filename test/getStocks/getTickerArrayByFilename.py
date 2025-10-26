import os
import re
from datetime import datetime

def extract_numbers_from_filenames(directory, isToday):
    numbers = []
    today = datetime.today().strftime('%Y%m%d')

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            if isToday:
                if not filename.startswith(today):
                    continue
            # [ 앞의 6자리 숫자 추출
            # match = re.search(r'\s(\d{6})\s*\[', filename)

            # 마지막 대괄호 안의 6자리 숫자 추출
            match = re.search(r'\[(\d{6})\]\.png$', filename)
            if match:
                    numbers.append(match.group(1))

    # 중복제거
    seen = set()
    uniq = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    return uniq


def extract_stock_code_from_filenames(directory):
    stock_codes = []

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # .png 떼고 strip
            name_only = os.path.splitext(filename)[0].strip()
            # 만약 ']'로 끝나면 스킵 (삼륭물산 [014970] 처럼)
            if name_only.endswith(']'):
                continue
            # ']' 뒤에 나오는 영어/숫자만 추출 (공백 포함 가능)
            match = re.search(r'\]\s*([A-Za-z0-9]+)$', name_only)
            if match:
                stock_codes.append(match.group(1))

    return stock_codes




directory = r'D:\kospi_stocks'  # 역슬래시 r''로 표기
directory = r'D:\5below20\g'  # 역슬래시 r''로 표기
extracted_numbers = extract_numbers_from_filenames(directory, False)
print("Extracted numbers:", extracted_numbers)


# directory = r'D:\sp500'  # 역슬래시 r''로 표기
# extracted_last_chars = extract_stock_code_from_filenames(directory)
# print("Extracted last chars:", extracted_last_chars)
