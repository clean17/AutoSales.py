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

# directory = 'D:\kosdaq_stocks'
directory = 'D:\kospi_stocks'

extracted_numbers = extract_numbers_from_filenames(directory)
print("Extracted numbers:", extracted_numbers)
