# -*- coding: utf-8 -*-
"""
Flet을 사용한 UI 레이아웃과 이벤트 핸들러를 정의하는 파일입니다.
"""
import flet as ft
import os
import json
import time
from threading import Thread

from agent.core import AIAgent
from tools.file_system import get_ai_organization_suggestion, create_folder, move_file

class AppUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.agent = AIAgent()
        self.current_directory = None
        self.selected_file = None
        self._initialize_controls()

    def _initialize_controls(self):
        """UI에 사용될 모든 컨트롤(위젯)을 초기화합니다."""
        self.status_text = ft.Text("AI 모델을 로딩 중입니다...")
        self.progress_ring = ft.ProgressRing(visible=True)
        self.chat_history = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.user_input = ft.TextField(hint_text="문서를 선택하고 질문하세요.", expand=True, disabled=True, on_submit=self._send_message_click)
        self.send_button = ft.IconButton(icon=ft.Icons.SEND, disabled=True, on_click=self._send_message_click)
        self.file_tree = ft.Column(scroll=ft.ScrollMode.ADAPTIVE, expand=True)
        self.selected_file_text = ft.Text("선택된 파일: 없음", size=12, color="grey")
        # mode_selector는 더 이상 필요 없으므로 삭제합니다.
        self.file_picker = ft.FilePicker(on_result=self._on_directory_selected)
        self.page.overlay.append(self.file_picker)

    def build(self):
        """페이지에 표시될 최종 UI 레이아웃을 구성하고 반환합니다."""
        return ft.Row(
            [
                ft.Container(
                    content=ft.Column([
                        ft.Text("파일 탐색기", weight=ft.FontWeight.BOLD), 
                        ft.Divider(), 
                        self.selected_file_text,
                        ft.Divider(),
                        self.file_tree
                    ], expand=True),
                    width=300, padding=10, border=ft.border.all(1, ft.Colors.BLACK26), border_radius=5
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.ElevatedButton("폴더 선택", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: self.file_picker.get_directory_path(dialog_title="관리할 폴더를 선택하세요")),
                            ft.ElevatedButton("AI 파일 정리 제안", icon=ft.Icons.AUTO_FIX_HIGH, on_click=self._suggest_organization_click),
                        ]),
                        ft.Row([self.status_text, self.progress_ring]),
                        # mode_selector를 UI에서 제거합니다.
                        ft.Container(content=self.chat_history, border=ft.border.all(1, ft.Colors.BLACK26), border_radius=5, padding=10, expand=True),
                        ft.Row(controls=[self.user_input, self.send_button])
                    ]),
                    padding=10, expand=True
                )
            ],
            expand=True
        )

    def _update_chat(self, message: str, is_user: bool):
        """채팅 내역을 업데이트하는 UI 로직입니다."""
        icon = ft.Icons.PERSON if is_user else ft.Icons.ASSISTANT
        color = "black" if is_user else "blue"
        self.chat_history.controls.append(ft.Row([ft.Icon(name=icon, color=color), ft.Text(message, selectable=True, expand=True, overflow=ft.TextOverflow.VISIBLE)]))
        self.page.update()

    def _on_file_click(self, file_path: str, file_name: str):
        """파일이 클릭되었을 때의 콜백 함수입니다."""
        print(f"파일 클릭됨: {file_path}")  # 디버깅용
        
        if os.path.isfile(file_path):
            # 지원되는 파일 형식 확인
            supported_extensions = ['.txt', '.md', '.pdf', '.docx', '.doc', '.py', '.json', '.xml', '.html']
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if file_ext in supported_extensions:
                self.selected_file = file_path
                self.selected_file_text.value = f"선택된 파일: {file_name}"
                self.selected_file_text.color = "green"
                
                # 파일 트리 업데이트 (선택 표시)
                self._update_file_tree(self.current_directory)
                
                # 선택된 파일로 RAG 업데이트
                self.user_input.disabled = True
                self.send_button.disabled = True
                self.status_text.value = f"'{file_name}' 문서를 로딩 중..."
                self.page.update()
                
                def update_agent_file():
                    self.agent.update_single_file(file_path)
                    self.status_text.value = f"'{file_name}' 문서 준비 완료!"
                    self.user_input.disabled = False
                    self.send_button.disabled = False
                    self.user_input.hint_text = f"'{file_name}'에 대해 질문하세요."
                    self.page.update()
                
                Thread(target=update_agent_file).start()
            else:
                self.selected_file_text.value = f"지원되지 않는 파일 형식: {file_name}"
                self.selected_file_text.color = "red"
                self.page.update()
        else:
            print(f"파일이 존재하지 않음: {file_path}")  # 디버깅용

    def _update_file_tree(self, path: str):
        """파일 탐색기 UI를 업데이트합니다."""
        self.file_tree.controls.clear()
        if path and os.path.isdir(path):
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                
                if os.path.isdir(item_path):
                    # 폴더
                    self.file_tree.controls.append(
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(name=ft.Icons.FOLDER, color="orange", size=16), 
                                ft.Text(item, color="orange")
                            ]),
                            padding=5
                        )
                    )
                else:
                    # 파일 - 클릭 가능하게 만들기
                    file_ext = os.path.splitext(item)[1].lower()
                    supported_extensions = ['.txt', '.md', '.pdf', '.docx', '.doc', '.py', '.json', '.xml', '.html']
                    
                    if file_ext in supported_extensions:
                        # 지원되는 파일 - 클릭 가능 (ElevatedButton 사용)
                        file_button = ft.ElevatedButton(
                            content=ft.Row([
                                ft.Icon(name=ft.Icons.INSERT_DRIVE_FILE, color="white", size=16), 
                                ft.Text(item, color="white", size=12)
                            ], tight=True),
                            bgcolor=ft.Colors.GREEN if self.selected_file == item_path else ft.Colors.BLUE,
                            color="white",
                            height=35,
                            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5)),
                            on_click=lambda e, fp=item_path, fn=item: self._on_file_click(fp, fn)
                        )
                        self.file_tree.controls.append(file_button)
                    else:
                        # 지원되지 않는 파일
                        self.file_tree.controls.append(
                            ft.Container(
                                content=ft.Row([
                                    ft.Icon(name=ft.Icons.INSERT_DRIVE_FILE, color="grey", size=16), 
                                    ft.Text(item, color="grey")
                                ]),
                                padding=5
                            )
                        )
        self.page.update()

    def _on_directory_selected(self, e: ft.FilePickerResultEvent):
        """폴더가 선택되었을 때의 콜백 함수입니다."""
        if not e.path: return
        
        self.current_directory = e.path
        self.selected_file = None
        self.selected_file_text.value = "선택된 파일: 없음"
        self.selected_file_text.color = "grey"
        self.status_text.value = f"선택된 폴더: {e.path}"
        self.user_input.hint_text = "파일을 선택하고 질문하세요."
        self.user_input.disabled = True
        self.send_button.disabled = True
        
        # 폴더 구조 업데이트
        self._update_file_tree(e.path)
        
        # 에이전트 경로 업데이트 (파일 관리용)
        def update_agent_path():
            self.agent.update_doc_path(e.path)
            self.page.update()
        
        Thread(target=update_agent_path).start()

    def _execute_plan(self, commands: list):
        """AI가 제안한 정리 계획을 실행합니다."""
        results = [create_folder(cmd['folder_name']) if cmd['action'] == 'create_folder' else move_file(cmd['source'], cmd['destination']) for cmd in commands]
        self._update_chat("AI 정리 계획을 실행했습니다:\n" + "\n".join(results), is_user=False)
        if self.current_directory:
            self._update_file_tree(self.current_directory)

    def _suggest_organization_click(self, e):
        """'AI 파일 정리 제안' 버튼 클릭 이벤트 핸들러입니다."""
        if not self.current_directory:
            self._update_chat("먼저 '폴더 선택'으로 관리할 폴더를 지정해주세요.", is_user=False)
            return
        
        self._update_chat("AI가 파일 정리 계획을 제안합니다. 잠시만 기다려주세요...", is_user=False)
        self.progress_ring.visible = True
        self.page.update()
        
        def task():
            suggestions_json_str = get_ai_organization_suggestion(self.current_directory, self.agent.llm_pipeline)
            self.progress_ring.visible = False
            try:
                suggestions = json.loads(suggestions_json_str)
                if "error" in suggestions:
                    self._update_chat(f"오류: {suggestions['error']}", False)
                    return
                
                self.chat_history.controls.append(ft.Divider())
                for plan in suggestions:
                    description = plan.get('plan_description', '이름 없는 계획')
                    commands = plan.get('commands', [])
                    self.chat_history.controls.append(ft.Row([ft.Icon(ft.Icons.LIGHTBULB), ft.Text(f"제안: {description}", weight=ft.FontWeight.BOLD)]))
                    self.chat_history.controls.append(ft.ElevatedButton(text="이 계획 실행하기", icon=ft.Icons.PLAY_ARROW, on_click=lambda _, cmds=commands: self._execute_plan(cmds)))
                self.chat_history.controls.append(ft.Divider())
            except json.JSONDecodeError:
                self._update_chat("오류: AI가 잘못된 형식의 정리 계획을 생성했습니다.", False)
            
            self.page.update()

        Thread(target=task).start()

    def _run_ai_task(self, query: str):
        """사용자 입력을 받아 AI와 채팅하는 백그라운드 작업입니다."""
        # 파일 관리 모드가 제거되었으므로, 문서 질문(chat) 기능만 실행합니다.
        if not self.selected_file:
            self._update_chat("먼저 파일 탐색기에서 문서를 선택해주세요.", is_user=False)
            self.user_input.disabled = False
            self.send_button.disabled = False
            self.progress_ring.visible = False
            self.page.update()
            return
            
        ai_message_text = ft.Text(
            "", 
            selectable=True, 
            expand=True,
            overflow=ft.TextOverflow.VISIBLE
        )
        self.chat_history.controls.append(
            ft.Row([ft.Icon(name=ft.Icons.ASSISTANT, color="blue"), ai_message_text])
        )
        self.page.update()

        response_stream = self.agent.run_single_file_chat(query)
        for chunk in response_stream:
            ai_message_text.value += chunk
            self.page.update()
            time.sleep(0.02)
        
        self.user_input.disabled = False
        self.send_button.disabled = False
        self.progress_ring.visible = False
        if self.current_directory:
            self._update_file_tree(self.current_directory)
        self.page.update()

    def _send_message_click(self, e):
        """전송 버튼 클릭 또는 Enter 입력 시 호출되는 이벤트 핸들러입니다."""
        query = self.user_input.value
        if not query: return
        self.user_input.value = ""
        self.user_input.disabled = True
        self.send_button.disabled = True
        self.progress_ring.visible = True
        self._update_chat(query, is_user=True)
        self.page.update()
        Thread(target=self._run_ai_task, args=(query,)).start()

    def start_agent_initialization(self):
        """앱 시작 시 AI 에이전트를 초기화하는 백그라운드 작업을 시작합니다."""
        def task():
            self.agent.initialize()
            self.status_text.value = "준비 완료! '폴더 선택' 버튼으로 시작하세요."
            self.progress_ring.visible = False
            self.page.update()
        
        Thread(target=task).start()

