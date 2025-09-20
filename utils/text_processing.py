# -*- coding: utf-8 -*-
"""
텍스트 처리를 위한 유틸리티 함수를 포함하는 모듈입니다.
LLM 응답 정제 등 공통으로 사용되는 텍스트 관련 로직을 여기에 정의합니다.
"""
import re

def post_process_response(response: str) -> str:
    """
    LLM의 응답을 후처리하여 불필요한 태그, 특수 토큰, 반복 등을 제거합니다.
    """
    if not isinstance(response, str):
        return ""

    # 1. XML/HTML 태그 제거
    response = re.sub(r'<[^>]+>', '', response)

    # 2. 특수 토큰 및 불필요한 키워드 제거
    unwanted_patterns = [r'<s>', r'</s>',
        r'\*\*', r'###', r'assistant', r'user:',
        r'AI:', r'인공지능:', r'답변:', r'응답:', r'Final Answer:'
    ]
    for pattern in unwanted_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)

    # 3. 연속된 특수문자 및 공백 정리
    response = re.sub(r'[\-=]{3,}', '', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r' {2,}', ' ', response)

    # 4. 문장 중복 제거 (더 정교하게)
    sentences = re.split(r'(?<=[.!?])\s+', response)
    unique_sentences = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # 너무 짧은 문장 제외
            # 문장의 핵심 부분만 비교 (공백, 구두점 제거)
            core = re.sub(r'[^\w가-힣]', '', sentence.lower())
            if core not in seen:
                seen.add(core)
                unique_sentences.append(sentence)

    # 5. 최종 정리
    result = ' '.join(unique_sentences)
    if result and not result.endswith(('.', '!', '?')):
        # 마지막 문장이 마침표로 끝나지 않으면 추가
        last_char_index = -1
        if len(response.strip()) > 0:
            last_char = response.strip()[-1]
            if last_char in ['.', '!', '?']:
                 result += last_char

    return result.strip()