## 파이썬 가상환경

파이썬 가상환경에서 32비트를 별도로 사용한다면
1. 가상 환경을 생성 
```
python -m venv [가상환경이름]
```
2. 생성한 가상환경 디렉토리 이동후 `pyvenv.cfg`열기
3. home 부분 수정
```
home = C:\Python -> home = C:\Python32
```
4. 파이썬 인터프리터 변경 vs code에서 `Ctrl+Shift+P` 입력 후
Python Select Interpreter 설정에서 생성한 가상환경 선택
또는 윈도우에서
```
[가상환경이름]/Scripts/activate
```
Linux
```
source .venv/bin/activate
```

5. 패키지 설치