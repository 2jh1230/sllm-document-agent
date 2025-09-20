# -*- coding: utf-8 -*-
"""
애플리케이션을 실행하는 메인 파일입니다.
오케스트레이터와 UI를 초기화하고 연결합니다.
"""
import flet as ft
import queue
import time
from threading import Thread

from ui.app_ui import AppUI
from services.orchestrator import Orchestrator

def main(page: ft.Page):
    """Flet 앱의 메인 함수"""
    page.title = "개인 문서 관리 AI 에이전트 (리팩토링 버전)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1200
    page.window_height = 800
    page.window_min_width = 800
    page.window_min_height = 600

    # UI와 오케스트레이터 간의 통신을 위한 큐 생성
    task_queue = queue.Queue()

    # UI 업데이트 콜백 함수 정의
    def ui_update_callback():
        # Flet은 메인 스레드에서만 UI 업데이트를 허용하므로,
        # page.update()를 직접 호출하는 대신 이벤트를 사용합니다.
        # 
        # [수정] .send()를 .send_all()로 변경했습니다.
        page.pubsub.send_all("update_ui")

    # 오케스트레이터 초기화
    orchestrator = Orchestrator(task_queue, ui_update_callback)

    # UI 초기화
    app_ui = AppUI(page, task_queue, orchestrator.state)

    def on_ui_update(e):
        app_ui.update_ui_from_state()

    page.pubsub.subscribe(on_ui_update)

    # 페이지에 UI 빌드 및 추가
    page.add(app_ui.build())
    
    # 초기 UI 렌더링
    app_ui.update_ui_from_state()

if __name__ == "__main__":
    ft.app(target=main)