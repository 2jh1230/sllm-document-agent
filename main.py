# -*- coding: utf-8 -*-
"""
애플리케이션을 실행하는 메인 파일입니다.
"""
import flet as ft
from ui.app_layout import AppUI

def main(page: ft.Page):
    """Flet 앱의 메인 함수"""
    page.title = "개인 문서 관리 AI 에이전트 (리팩토링 버전)"
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # UI 클래스를 인스턴스화하고 페이지에 추가합니다.
    app_ui = AppUI(page)
    page.add(app_ui.build())
    
    # UI가 표시된 후 백그라운드에서 AI 에이전트 초기화를 시작합니다.
    app_ui.start_agent_initialization()

if __name__ == "__main__":
    # Flet 앱을 실행합니다.
    ft.app(target=main)

