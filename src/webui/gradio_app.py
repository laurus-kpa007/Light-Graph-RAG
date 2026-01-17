"""
Gradio 웹 인터페이스
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, List
import asyncio

from ..preprocessing import DocumentConverter, MarkdownParser, Chunker, ChunkConfig
from ..rag import LightRAGWrapper, Indexer, Searcher, SearchConfig
from ..utils import Config


class GradioApp:
    """Gradio 웹 인터페이스"""

    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: 설정 객체
        """
        self.config = config or Config.default()

        # 모듈 초기화
        self.converter = DocumentConverter()
        self.parser = MarkdownParser()
        self.chunker = Chunker(ChunkConfig(
            max_chunk_size=self.config.rag.chunk_size,
            overlap_size=self.config.rag.chunk_overlap,
            preserve_tables=self.config.rag.preserve_tables
        ))

        # RAG 초기화
        self.rag = LightRAGWrapper(
            working_dir=str(self.config.paths.index_dir),
            llm_model=self.config.llm.llm_model,
            embedding_model=self.config.llm.embedding_model,
            ollama_host=self.config.llm.ollama_host
        )

        self.indexer = Indexer(self.rag, batch_size=10)
        self.searcher = Searcher(
            self.rag,
            config=SearchConfig(
                search_mode=self.config.rag.search_mode,
                top_k=self.config.rag.top_k
            )
        )

    def handle_query(
        self,
        query: str,
        search_mode: str
    ) -> Tuple[str, str]:
        """질의 처리"""
        if not query.strip():
            return "질문을 입력해주세요.", ""

        try:
            result = self.searcher.search(query, mode=search_mode)

            answer = result.answer
            metadata = f"검색 모드: {result.search_mode}\n소요 시간: {result.latency_ms:.2f}ms"

            return answer, metadata

        except Exception as e:
            return f"검색 중 오류 발생: {str(e)}", ""

    def handle_upload(
        self,
        files: List[gr.File],
        progress=gr.Progress()
    ) -> str:
        """파일 업로드 및 인덱싱"""
        if not files:
            return "파일을 선택해주세요."

        try:
            all_chunks = []
            total_files = len(files)

            for i, file in enumerate(files):
                progress((i + 1) / total_files, desc=f"처리 중: {file.name}")

                # 파일 변환
                file_path = Path(file.name)

                # 임시 파일 처리
                if hasattr(file, 'name') and Path(file.name).exists():
                    markdown = self.converter.convert_file(str(file.name))
                else:
                    continue

                # 파싱 및 청크 분할
                doc = self.parser.parse(markdown)
                chunks = self.chunker.chunk_document(doc)

                # 메타데이터 추가
                for chunk in chunks:
                    chunk.metadata["source_file"] = file_path.name

                all_chunks.extend(chunks)

            # 인덱싱
            progress(0.9, desc="인덱싱 중...")
            result = self.indexer.index_chunks(all_chunks)

            return f"""인덱싱 완료!

총 파일 수: {total_files}개
총 청크 수: {result.total_count}개
성공: {result.success_count}개
실패: {result.failed_count}개
소요 시간: {result.duration_seconds:.2f}초
"""

        except Exception as e:
            return f"인덱싱 중 오류 발생: {str(e)}"

    def get_system_status(self) -> Tuple[str, str]:
        """시스템 상태 조회"""
        try:
            stats = self.rag.get_index_stats()

            # 인덱스 통계
            stats_text = f"""
**인덱스 정보**
- 작업 디렉토리: {stats.get('working_dir', 'N/A')}
- 인덱스 크기: {stats.get('total_size_mb', 0):.2f} MB
- 파일 수: {len(stats.get('index_files', []))}개

**모델 정보**
- LLM: {stats.get('llm_model', 'N/A')}
- 임베딩: {stats.get('embedding_model', 'N/A')}
- 한국어 프롬프트: {'활성화' if stats.get('korean_prompts_enabled') else '비활성화'}
"""

            # Ollama 상태
            ollama_status = "연결 확인 중..."

            return stats_text, ollama_status

        except Exception as e:
            return f"상태 조회 실패: {str(e)}", "연결 실패"

    def create_app(self) -> gr.Blocks:
        """Gradio 앱 생성"""
        with gr.Blocks(
            title="사내규정 Q&A 시스템",
            theme=gr.themes.Soft()
        ) as app:
            gr.Markdown("# 📚 사내 규정 질의응답 시스템")
            gr.Markdown("Light GraphRAG 기반 한국어 문서 검색")

            with gr.Tabs():
                # 탭 1: 질의응답
                with gr.Tab("💬 질의응답"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            query_input = gr.Textbox(
                                label="질문을 입력하세요",
                                placeholder="예: 연차 사용 규정은 어떻게 되나요?",
                                lines=3
                            )

                            search_mode = gr.Radio(
                                choices=["hybrid", "local", "global", "naive"],
                                value="hybrid",
                                label="검색 모드",
                                info="hybrid 모드 권장 (그래프 + 벡터 융합)"
                            )

                            submit_btn = gr.Button("🔍 검색", variant="primary", size="lg")

                        with gr.Column(scale=3):
                            answer_output = gr.Textbox(
                                label="답변",
                                lines=15,
                                interactive=False
                            )

                            metadata_output = gr.Textbox(
                                label="메타데이터",
                                lines=2,
                                interactive=False
                            )

                    submit_btn.click(
                        fn=self.handle_query,
                        inputs=[query_input, search_mode],
                        outputs=[answer_output, metadata_output]
                    )

                    # 예제 질문
                    gr.Examples(
                        examples=[
                            ["연차 사용 규정은?", "hybrid"],
                            ["휴가 신청 방법은?", "hybrid"],
                            ["누구에게 적용되나요?", "hybrid"],
                        ],
                        inputs=[query_input, search_mode]
                    )

                # 탭 2: 문서 관리
                with gr.Tab("📁 문서 관리"):
                    gr.Markdown("### 문서 업로드 및 인덱싱")

                    file_upload = gr.Files(
                        label="문서 파일 업로드 (.docx 또는 .md)",
                        file_types=[".docx", ".md"],
                        file_count="multiple"
                    )

                    index_btn = gr.Button("📥 인덱싱 시작", variant="primary")

                    status_output = gr.Textbox(
                        label="상태",
                        lines=10,
                        interactive=False
                    )

                    index_btn.click(
                        fn=self.handle_upload,
                        inputs=[file_upload],
                        outputs=[status_output]
                    )

                # 탭 3: 시스템 상태
                with gr.Tab("⚙️ 시스템 상태"):
                    gr.Markdown("### 시스템 정보")

                    refresh_btn = gr.Button("🔄 새로고침")

                    with gr.Row():
                        stats_output = gr.Markdown(label="인덱스 통계")
                        ollama_output = gr.Textbox(
                            label="Ollama 상태",
                            lines=3,
                            interactive=False
                        )

                    refresh_btn.click(
                        fn=self.get_system_status,
                        outputs=[stats_output, ollama_output]
                    )

            gr.Markdown("""
---
**Light GraphRAG v1.0** | 로컬 LLM 기반 한국어 문서 검색 시스템
""")

        return app


def create_app(config: Optional[Config] = None) -> gr.Blocks:
    """Gradio 앱 생성 (헬퍼 함수)"""
    app_instance = GradioApp(config)
    return app_instance.create_app()


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
