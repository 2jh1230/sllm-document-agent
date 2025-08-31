개인 문서 관리 AI 에이전트
로컬 대규모 언어 모델(LLM)을 사용하여 개인 컴퓨터의 문서를 관리하고, 문서 내용에 대해 질문하며, 파일 정리를 제안받을 수 있는 데스크톱 애플리케이션입니다.

주요 기능
로컬 LLM 구동: 인터넷 연결 없이 로컬 환경에서 AI 모델을 실행하여 개인정보를 보호합니다.

문서 기반 Q&A: 특정 문서(TXT, PDF, DOCX 등)를 선택하고 해당 문서의 내용에 대해 AI와 대화할 수 있습니다.

AI 파일 정리 제안: 지정된 폴더의 파일 목록을 분석하여 AI가 효율적인 정리 계획을 제안하고 실행할 수 있습니다.

직관적인 UI: Flet 프레임워크를 사용하여 사용하기 쉬운 그래픽 인터페이스를 제공합니다.

설치 및 실행
사전 준비: Python 3.10 이상, Git이 설치되어 있어야 합니다.

프로젝트 복제(Clone)

git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
cd YourRepositoryName/frontend

AI 모델 다운로드

이 프로젝트는 HyperCLOVAX-1.5B-model 모델을 사용합니다.

frontend 폴더 내에 models 라는 새 폴더를 만드세요.

다운로드한 모델 파일을 models/HyperCLOVAX-1.5B-model 경로에 위치시킵니다.

config.py 파일의 MODEL_PATH를 다음과 같이 수정합니다.

# config.py
MODEL_PATH = "models/HyperCLOVAX-1.5B-model"

가상환경 생성 및 활성화

# 가상환경 생성
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

필요한 라이브러리 설치

pip install -r requirements.txt

애플리케이션 실행

flet run main.py

사용 방법
앱이 실행되면 '폴더 선택' 버튼을 눌러 관리하고 싶은 문서가 있는 폴더를 선택합니다.

왼쪽 파일 탐색기에 선택한 폴더의 파일 목록이 나타납니다.

질문하고 싶은 문서 파일을 클릭하면 해당 파일이 초록색으로 활성화되며 로딩됩니다.

하단 입력창에 질문을 입력하고 전송 버튼을 누르면 AI가 문서 내용을 기반으로 답변합니다.

폴더 내 파일들을 정리하고 싶다면 'AI 파일 정리 제안' 버튼을 클릭하세요. AI가 제안하는 정리 계획을 보고 실행할 수 있습니다.
