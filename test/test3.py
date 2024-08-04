import os
import matplotlib.pyplot as plt

# output_dir 설정
output_dir = '/images'

# 절대 경로로 변환
output_dir = os.path.abspath(output_dir)

# 디렉터리가 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 파일 이름 변수들 설정
ticker = 'AAPL'
stock_name = 'Apple'

# 이미지 저장 경로 설정
file_path = os.path.join(output_dir, f'{ticker}_{stock_name}.png')

# 디버깅 메시지 출력
print(f"Saving image to: {file_path}")

# 예제 그래프 생성
plt.plot([1, 2, 3], [4, 5, 6])

# 이미지 저장
try:
    plt.savefig(file_path)
    print(f"Image successfully saved to {file_path}")
except Exception as e:
    print(f"Error saving image: {e}")

# 그래프 닫기
plt.close()
