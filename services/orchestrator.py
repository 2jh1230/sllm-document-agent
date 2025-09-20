# -*- coding: utf-8 -*-
"""
애플리케이션의 핵심 로직과 상태 관리를 담당하는 Orchestrator 클래스입니다.
UI 스레드와 백그라운드 작업 스레드 간의 통신을 중재합니다.
"""
import queue
from threading import Thread
import os

from services.ai_engine import AIEngine
from services.file_service import list_files_in_directory, execute_file_plan
import config

class Orchestrator:
    def __init__(self, task_queue: queue.Queue, ui_update_callback):
        self.task_queue = task_queue
        self.ui_update_callback = ui_update_callback
        
        self.ai_engine = AIEngine()
        
        # 애플리케이션의 중앙 상태 (UI는 이 상태를 참조하여 렌더링됨)
        self.state = {
            "status_text": "준비 완료! '폴더 선택' 버튼으로 시작하세요.",
            "is_loading": False,
            "chat_history": [],
            "file_tree": [],
            "current_directory": None,
            "selected_file": None,
            "is_chat_ready": False,
        }
        
        self.worker_thread = Thread(target=self._run, daemon=True)
        self.worker_thread.start()

    def _update_state(self, new_state: dict):
        """중앙 상태를 업데이트하고 UI에 변경을 알립니다."""
        self.state.update(new_state)
        self.ui_update_callback()

    def _run(self):
        """작업 큐를 계속 확인하고 들어오는 작업을 처리하는 메인 루프입니다."""
        while True:
            try:
                task = self.task_queue.get()
                task_type = task.get("type")
                payload = task.get("payload")

                if task_type == "SELECT_DIRECTORY":
                    self._handle_select_directory(payload)
                elif task_type == "SELECT_FILE":
                    self._handle_select_file(payload)
                elif task_type == "SEND_MESSAGE":
                    self._handle_send_message(payload)
                elif task_type == "SUGGEST_ORGANIZATION":
                    self._handle_suggest_organization()
                elif task_type == "EXECUTE_PLAN":
                    self._handle_execute_plan(payload)

            except Exception as e:
                print(f"오케스트레이터 오류: {e}")
                self._update_state({
                    "status_text": f"오류 발생: {e}",
                    "is_loading": False
                })

    def _get_file_tree_from_path(self, path: str) -> list:
        """주어진 경로로부터 UI에 표시할 파일 트리 구조를 생성합니다."""
        tree = []
        try:
            items = list_files_in_directory(path)
            for item in items:
                item_path = os.path.join(path, item)
                is_dir = os.path.isdir(item_path)
                # splitext의 결과는 ('파일이름', '.확장자') 형태
                ext = os.path.splitext(item)[1].lower() if not is_dir else ''
                is_supported = ext in config.SUPPORTED_EXTENSIONS
                
                tree.append({
                    "name": item,
                    "path": item_path,
                    "is_dir": is_dir,
                    "is_supported": not is_dir and is_supported
                })
        except Exception as e:
            self._update_state({"status_text": f"폴더 읽기 오류: {e}"})
        return tree

    def _handle_select_directory(self, path: str):
        self._update_state({
            "is_loading": True,
            "status_text": f"폴더 로딩 중: {path}",
            "current_directory": path,
            "selected_file": None,
            "is_chat_ready": False,
            "chat_history": []
        })
        file_tree = self._get_file_tree_from_path(path)
        self._update_state({
            "file_tree": file_tree,
            "is_loading": False,
            "status_text": f"폴더 선택됨: {path}"
        })

    def _handle_select_file(self, file_path: str):
        file_name = os.path.basename(file_path)
        self._update_state({
            "is_loading": True,
            "status_text": f"'{file_name}' 문서 로딩 중...",
            "selected_file": file_path,
            "is_chat_ready": False
        })
        
        success = self.ai_engine.setup_rag_for_file(file_path)
        
        if success:
            self._update_state({
                "is_loading": False,
                "status_text": f"'{file_name}' 문서 준비 완료! 질문하세요.",
                "is_chat_ready": True
            })
        else:
            self._update_state({
                "is_loading": False,
                "status_text": f"'{file_name}' 문서 로드 실패.",
                "selected_file": None,
                "is_chat_ready": False
            })

    def _handle_send_message(self, query: str):
        self.state["chat_history"].append({"sender": "user", "message": query})
        self._update_state({
            "is_loading": True,
            "status_text": "AI가 답변을 생성 중입니다..."
        })

        ai_response = ""
        self.state["chat_history"].append({"sender": "ai", "message": ai_response})
        
        for chunk in self.ai_engine.run_rag_chat(query):
            ai_response += chunk
            self.state["chat_history"][-1]["message"] = ai_response
            self.ui_update_callback() # 스트리밍을 위해 직접 콜백 호출

        self._update_state({
            "is_loading": False,
            "status_text": "답변 생성 완료. 다음 질문을 입력하세요."
        })

    def _handle_suggest_organization(self):
        if not self.state["current_directory"]:
            self.state["chat_history"].append({"sender": "ai", "message": "먼저 '폴더 선택'으로 관리할 폴더를 지정해주세요."})
            self._update_state({})
            return

        self._update_state({"is_loading": True, "status_text": "AI가 파일 정리 계획을 제안합니다..."})
        self.state["chat_history"].append({"sender": "ai", "message": "AI가 파일 정리 계획을 분석 중입니다. 잠시만 기다려주세요..."})
        
        file_list = [item['name'] for item in self.state['file_tree'] if not item['is_dir']]
        suggestions = self.ai_engine.get_organization_suggestion(self.state["current_directory"], file_list)
        
        if "error" in suggestions:
            self.state["chat_history"][-1]["message"] = f"오류: {suggestions['error']}"
        else:
            self.state["chat_history"][-1] = {"sender": "ai", "message": "다음과 같은 정리 계획을 제안합니다:", "plans": suggestions.get('plans', [])}

        self._update_state({
            "is_loading": False,
            "status_text": "파일 정리 계획 제안 완료."
        })

    def _handle_execute_plan(self, plan: dict):
        self._update_state({"is_loading": True, "status_text": "정리 계획을 실행 중입니다..."})
        
        commands = plan.get('commands', [])
        results = execute_file_plan(self.state["current_directory"], commands)
        
        result_message = "계획 실행 결과:\n" + "\n".join(results)
        self.state["chat_history"].append({"sender": "ai", "message": result_message})
        
        # 파일 트리 새로고침
        file_tree = self._get_file_tree_from_path(self.state["current_directory"])
        self._update_state({
            "file_tree": file_tree,
            "is_loading": False,
            "status_text": "파일 정리 완료."
        })