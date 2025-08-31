# -*- coding: utf-8 -*-
"""
파일 시스템을 조작하는 LangChain 도구(@tool)와 관련 유틸리티 함수를 정의하는 파일입니다.
"""
import os
import shutil
import json
import re
from langchain.tools import tool

@tool
def list_files(directory: str) -> str:
    """
    지정된 디렉터리에 있는 파일 및 폴더 목록을 반환합니다. 
    입력(directory)은 파일 목록을 보고 싶은 폴더 경로여야 합니다.
    """
    if not os.path.isdir(directory):
        return f"오류: '{directory}'는 유효한 폴더가 아닙니다."
    try:
        files = os.listdir(directory)
        if not files:
            return f"'{directory}' 폴더는 비어 있습니다."
        return f"'{directory}' 폴더의 내용:\n" + "\n".join(files)
    except Exception as e:
        return f"오류 발생: {e}"

@tool
def move_file(source: str, destination: str) -> str:
    """
    파일이나 폴더를 한 위치에서 다른 위치로 이동시킵니다.
    입력(source)은 이동할 파일 또는 폴더의 경로여야 하고, 
    입력(destination)은 최종 목적지 경로여야 합니다.
    """
    try:
        shutil.move(source, destination)
        return f"성공: '{source}'를 '{destination}'(으)로 이동했습니다."
    except Exception as e:
        return f"오류 발생: {e}"

@tool
def create_folder(folder_name: str) -> str:
    """
    새로운 폴더(디렉터리)를 생성합니다.
    입력(folder_name)은 생성할 폴더의 이름이어야 합니다.
    """
    try:
        os.makedirs(folder_name, exist_ok=True)
        return f"성공: '{folder_name}' 폴더를 생성했습니다."
    except Exception as e:
        return f"오류 발생: {e}"

def clean_llm_response(response: str) -> str:
    """LLM 응답에서 불필요한 태그와 반복을 제거합니다."""
    # XML/HTML 태그 제거
    response = re.sub(r'<[^>]+>', '', response)
    
    # 특수 토큰 제거
    special_tokens = ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '[INST]', '[/INST]', 
                     '<s>', '</s>', '<bos>', '<eos>', '###', '**', '*']
    for token in special_tokens:
        response = response.replace(token, '')
    
    # 연속된 줄바꿈 정리
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # 연속된 공백 정리
    response = re.sub(r' {2,}', ' ', response)
    
    # JSON 부분만 추출
    json_start = response.find('[')
    json_end = response.rfind(']')
    
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return response[json_start:json_end+1]
    
    return response.strip()

def get_ai_organization_suggestion(directory: str, llm_pipeline) -> str:
    """
    주어진 폴더의 파일 목록을 분석하여, 파일을 정리할 수 있는 2가지 계획을 LLM을 통해 동적으로 제안합니다.
    """
    if not os.path.isdir(directory):
        return json.dumps({"error": f"'{directory}'는 유효한 폴더가 아닙니다."})

    file_list_str = list_files.invoke(directory)
    
    # 더 간단하고 명확한 프롬프트
    prompt = f"""아래 파일 목록을 보고 정리 계획 2개를 JSON 형식으로 제안하세요.

파일 목록:
{file_list_str}

응답 형식 (JSON만):
[
  {{
    "plan_description": "계획 설명",
    "commands": [
      {{"action": "create_folder", "folder_name": "{directory}/폴더명"}},
      {{"action": "move_file", "source": "{directory}/파일명", "destination": "{directory}/폴더명/파일명"}}
    ]
  }},
  {{
    "plan_description": "다른 계획 설명", 
    "commands": [...]
  }}
]

JSON만 출력하세요:"""
    
    try:
        response = llm_pipeline.invoke(prompt)
        
        # 응답 정리
        cleaned_response = clean_llm_response(response)
        
        # JSON 파싱 테스트
        parsed_json = json.loads(cleaned_response)
        
        # 유효성 검사
        if not isinstance(parsed_json, list) or len(parsed_json) == 0:
            raise ValueError("응답이 리스트 형태가 아닙니다.")
        
        for plan in parsed_json:
            if 'plan_description' not in plan or 'commands' not in plan:
                raise ValueError("계획에 필수 키가 없습니다.")
        
        return json.dumps(parsed_json, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON 파싱 오류: {e}"})
    except Exception as e:
        return json.dumps({"error": f"계획 생성 중 오류: {e}"})