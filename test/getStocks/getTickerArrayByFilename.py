import os
import re

def extract_numbers_from_filenames(directory):
    numbers = []

    for filename in os.listdir(directory):
        # .png 파일만 처리
        if filename.endswith('.png'):
            # 정규식을 사용하여 두 번째 [ 앞의 6자리 숫자 추출
            match = re.search(r'\s(\d{6})\s*\[', filename)
            if match:
                numbers.append(match.group(1))

    return numbers


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


# extracted_numbers = extract_numbers_from_filenames(directory)
# print("Extracted numbers:", extracted_numbers)


extracted_last_chars = extract_stock_code_from_filenames(directory)
print("Extracted last chars:", extracted_last_chars)
