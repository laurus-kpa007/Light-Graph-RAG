"""
WebUI 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.webui import create_app
from src.utils import Config


def main():
    """WebUI 실행"""
    print("="*60)
    print("Light GraphRAG - 사내 규정 질의응답 시스템")
    print("="*60)

    # 설정 로드
    config = Config.default()

    print(f"\n설정 정보:")
    print(f"  - LLM 모델: {config.llm.llm_model}")
    print(f"  - 임베딩 모델: {config.llm.embedding_model}")
    print(f"  - Ollama 주소: {config.llm.ollama_host}")
    print(f"  - 인덱스 디렉토리: {config.paths.index_dir}")

    # 앱 생성 및 실행
    print(f"\nWebUI 시작 중...")
    app = create_app(config)

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
