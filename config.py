# -*- coding: utf-8 -*-
"""
애플리케이션의 모든 설정 값을 관리하는 파일입니다.
모델 경로, RAG 파라미터 등 주요 변수를 여기서 수정할 수 있습니다.
pathlib을 사용하여 운영체제에 독립적인 경로를 설정합니다.
"""
from pathlib import Path

# --- 기본 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent

# --- 모델 및 경로 설정 ---
MODEL_PATH = "D:/sllm_proj/HyperCLOVAX-1.5B-model"
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"

# --- RAG (검색 증강 생성) 설정 ---
# [수정] 청크 크기를 줄여 메모리 사용량 감소
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
# RAG 검색 시 가져올 문서 수 (정리 기능과 직접적 관련은 없으나 낮춰서 안정성 확보)
SEARCH_K = 1

# --- LLM (거대 언어 모델) 설정 ---
# [수정] 모델이 생성하는 최대 토큰 수를 줄여 메모리 사용량 크게 감소
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
TOP_P = 0.8
REPETITION_PENALTY = 1.2

# --- 지원되는 파일 확장자 ---
SUPPORTED_EXTENSIONS = [
    ".txt", ".md", ".py", ".json", ".xml", ".html", ".csv",
    ".pdf", ".docx", ".doc"
]