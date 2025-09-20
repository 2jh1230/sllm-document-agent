# -*- coding: utf-8 -*-
"""
파일 시스템을 조작하는 함수를 정의하는 서비스 모듈입니다.
이 함수들은 오케스트레이터에 의해 호출됩니다.
"""
import os
import shutil
from typing import List, Dict, Union

def list_files_in_directory(directory: str) -> List[str]:
    """지정된 디렉터리의 파일 및 폴더 목록을 반환합니다."""
    if not os.path.isdir(directory):
        raise ValueError(f"오류: '{directory}'는 유효한 폴더가 아닙니다.")
    return sorted(os.listdir(directory))

def create_directory(folder_path: str) -> str:
    """새로운 폴더(디렉터리)를 생성합니다."""
    try:
        os.makedirs(folder_path, exist_ok=True)
        return f"성공: '{folder_path}' 폴더를 생성했습니다."
    except Exception as e:
        return f"오류 발생: {e}"

def move_path(source: str, destination: str) -> str:
    """파일이나 폴더를 이동시킵니다."""
    try:
        shutil.move(source, destination)
        return f"성공: '{source}'를 '{destination}'(으)로 이동했습니다."
    except Exception as e:
        return f"오류 발생: {e}"

def execute_file_plan(base_directory: str, commands: List[Dict]) -> List[str]:
    """
    파일 정리 계획에 따라 여러 파일 시스템 명령을 실행합니다.
    경로는 항상 base_directory를 기준으로 합니다.
    """
    results = []
    for cmd in commands:
        action = cmd.get('action')
        try:
            if action == 'create_folder':
                folder_name = cmd.get('folder_name')
                if not folder_name:
                    results.append("오류: 'create_folder'에 'folder_name'이 없습니다.")
                    continue
                # 보안을 위해 경로 조작 방지
                full_path = os.path.abspath(os.path.join(base_directory, folder_name))
                if not full_path.startswith(os.path.abspath(base_directory)):
                    results.append(f"오류: 허용되지 않은 경로 접근 - {folder_name}")
                    continue
                results.append(create_directory(full_path))

            elif action == 'move_file':
                source = cmd.get('source')
                destination = cmd.get('destination')
                if not source or not destination:
                    results.append("오류: 'move_file'에 'source' 또는 'destination'이 없습니다.")
                    continue

                source_path = os.path.abspath(os.path.join(base_directory, source))
                dest_path = os.path.abspath(os.path.join(base_directory, destination))

                if not source_path.startswith(os.path.abspath(base_directory)) or \
                   not dest_path.startswith(os.path.abspath(base_directory)):
                    results.append(f"오류: 허용되지 않은 경로 접근 - {source} -> {destination}")
                    continue
                
                # 목적지 폴더가 없으면 생성
                dest_dir = os.path.dirname(dest_path)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)

                results.append(move_path(source_path, dest_path))
            else:
                results.append(f"알 수 없는 액션: {action}")
        except Exception as e:
            results.append(f"'{action}' 실행 중 오류: {e}")
    return results