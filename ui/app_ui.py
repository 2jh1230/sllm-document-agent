# -*- coding: utf-8 -*-
"""
Flet을 사용한 UI 레이아웃과 이벤트 핸들러를 정의하는 파일입니다.
이 클래스는 상태를 소유하지 않으며, 오케스트레이터로부터 받은 상태를 렌더링하고
사용자 입력을 오케스트레이터의 작업 큐로 보냅니다.
"""
import flet as ft
import queue

class AppUI:
    def __init__(self, page: ft.Page, task_queue: queue.Queue, initial_state: dict):
        self.page = page
        self.task_queue = task_queue
        self.state = initial_state
        
        self._initialize_controls()
        self.page.on_resize = self.on_resize

    def _initialize_controls(self):
        """UI에 사용될 모든 컨트롤(위젯)을 초기화합니다."""
        self.status_text = ft.Text()
        self.progress_bar = ft.ProgressBar(visible=False)
        self.chat_history = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.user_input = ft.TextField(hint_text="분석할 파일을 선택하세요.", expand=True, on_submit=self._send_message_click)
        self.send_button = ft.IconButton(icon=ft.Icons.SEND, on_click=self._send_message_click)
        self.file_tree = ft.Column(scroll=ft.ScrollMode.ADAPTIVE, expand=True)
        self.selected_file_text = ft.Text(size=12, color="grey")
        
        self.file_picker = ft.FilePicker(on_result=self._on_directory_selected)
        self.page.overlay.append(self.file_picker)

    def build(self):
        """페이지에 표시될 최종 UI 레이아웃을 구성하고 반환합니다."""
        # 좌측 파일 탐색기 및 제어 버튼 영역
        left_panel = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("파일 탐색기", weight=ft.FontWeight.BOLD),
                    self.selected_file_text,
                    ft.Divider(),
                    self.file_tree,
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(
                                "폴더 선택",
                                icon=ft.Icons.FOLDER_OPEN,
                                on_click=lambda _: self.file_picker.get_directory_path(
                                    dialog_title="관리할 폴더를 선택하세요"
                                ),
                            ),
                            ft.ElevatedButton(
                                "AI 파일 정리 제안",
                                icon=ft.Icons.AUTO_FIX_HIGH,
                                on_click=self._suggest_organization_click,
                            ),
                        ]
                    )
                ]
            ),
            width=350,
            padding=10,
            border=ft.border.all(1, ft.Colors.BLACK26), # 수정됨
            border_radius=5
        )

        # 우측 채팅 및 상태 표시 영역
        right_panel = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        [self.status_text, self.progress_bar],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                    ),
                    ft.Container(
                        content=self.chat_history,
                        border=ft.border.all(1, ft.Colors.BLACK26), # 수정됨
                        border_radius=5,
                        padding=10,
                        expand=True
                    ),
                    ft.Row(
                        controls=[self.user_input, self.send_button]
                    )
                ]
            ),
            padding=10,
            expand=True
        )

        # 최종 레이아웃 반환
        return ft.Row(
            controls=[
                left_panel,
                ft.VerticalDivider(),
                right_panel
            ],
            expand=True
        )

    def update_ui_from_state(self):
        """오케스트레이터의 중앙 상태를 기반으로 전체 UI를 업데이트합니다."""
        self.status_text.value = self.state.get("status_text", "")
        self.progress_bar.visible = self.state.get("is_loading", False)
        
        # 파일 트리 업데이트
        self.file_tree.controls.clear()
        current_selected_file = self.state.get("selected_file")
        if self.state.get("current_directory"):
            for item in self.state.get("file_tree", []):
                is_selected = not item["is_dir"] and item["path"] == current_selected_file
                self.file_tree.controls.append(self._create_file_tree_item(item, is_selected))
        
        # 선택된 파일 텍스트 업데이트
        if current_selected_file:
            self.selected_file_text.value = f"선택: {current_selected_file}"
            self.selected_file_text.max_lines=1
            self.selected_file_text.overflow=ft.TextOverflow.ELLIPSIS
            self.selected_file_text.tooltip=current_selected_file
            self.selected_file_text.color = "green"
        else:
            self.selected_file_text.value = "선택된 파일: 없음"
            self.selected_file_text.color = "grey"

        # 채팅 내역 업데이트
        self.chat_history.controls.clear()
        for entry in self.state.get("chat_history", []):
            self.chat_history.controls.append(self._create_chat_message(entry))
        
        # 입력 필드 및 버튼 상태 업데이트
        is_chat_ready = self.state.get("is_chat_ready", False)
        is_loading = self.state.get("is_loading", False)
        self.user_input.disabled = not is_chat_ready or is_loading
        self.send_button.disabled = not is_chat_ready or is_loading
        if is_chat_ready and not is_loading:
            self.user_input.hint_text = "선택된 파일에 대해 질문하세요."
        
        self.page.update()

    def _create_file_tree_item(self, item, is_selected):
        """파일 트리 목록의 개별 항목을 생성합니다."""
        icon = ft.Icons.FOLDER if item["is_dir"] else ft.Icons.INSERT_DRIVE_FILE
        color = "orange" if item["is_dir"] else ("green" if is_selected else ("blue" if item["is_supported"] else "grey"))
        
        content = ft.Row(
            controls=[
                ft.Icon(name=icon, color=color, size=16),
                ft.Text(value=item["name"], color=color, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, tooltip=item["name"])
            ]
        )
        
        if item["is_dir"]:
            return ft.Container(content=content, padding=5)
        else:
            return ft.TextButton(
                content=content,
                on_click=lambda _, p=item["path"]: self._on_file_click(p),
                disabled=not item["is_supported"],
                style=ft.ButtonStyle(
                    bgcolor=ft.Colors.GREEN_100 if is_selected else None, # 수정됨
                    padding=ft.padding.only(left=5)
                )
            )

    def _create_chat_message(self, entry):
        """채팅 메시지 위젯을 생성합니다."""
        sender = entry.get("sender")
        message = entry.get("message", "")
        plans = entry.get("plans")

        if sender == "user":
            return ft.Row(
                controls=[
                    ft.Icon(name=ft.Icons.PERSON, color="blue"),
                    ft.Container(
                        content=ft.Text(message, selectable=True),
                        padding=10,
                        border_radius=10,
                        bgcolor=ft.Colors.BLUE_100, # 수정됨
                    )
                ],
                alignment=ft.MainAxisAlignment.START
            )
        else: # AI
            ai_message = ft.Row(
                controls=[
                    ft.Icon(name=ft.Icons.ASSISTANT, color="green"),
                    ft.Container(
                        content=ft.Text(message, selectable=True),
                        padding=10,
                    )
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
            
            content_column = [ai_message]
            
            if plans:
                content_column.append(ft.Divider())
                for i, plan in enumerate(plans):
                    plan_desc = plan.get('plan_description', f'계획 {i+1}')
                    plan_text = ft.Text(f"제안 {i+1}: {plan_desc}", weight=ft.FontWeight.BOLD)
                    
                    execute_button = ft.ElevatedButton(
                        text="이 계획 실행하기",
                        icon=ft.Icons.PLAY_ARROW,
                        on_click=lambda _, p=plan: self._execute_plan_click(p),
                        tooltip="이 제안에 따라 파일 정리를 시작합니다."
                    )
                    content_column.append(ft.Row([plan_text, execute_button], alignment=ft.MainAxisAlignment.SPACE_BETWEEN))
                content_column.append(ft.Divider())
                
            return ft.Column(controls=content_column)

    # --- 이벤트 핸들러 (오케스트레이터에 작업 요청) ---
    def _on_directory_selected(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.task_queue.put({"type": "SELECT_DIRECTORY", "payload": e.path})

    def _on_file_click(self, file_path: str):
        self.task_queue.put({"type": "SELECT_FILE", "payload": file_path})

    def _send_message_click(self, e):
        query = self.user_input.value
        if query:
            self.user_input.value = ""
            self.task_queue.put({"type": "SEND_MESSAGE", "payload": query})

    def _suggest_organization_click(self, e):
        self.task_queue.put({"type": "SUGGEST_ORGANIZATION"})

    def _execute_plan_click(self, plan: dict):
        self.task_queue.put({"type": "EXECUTE_PLAN", "payload": plan})
        
    def on_resize(self, e):
        # 창 크기 변경 시 UI 업데이트 (텍스트 잘림 등 재계산)
        self.page.update()