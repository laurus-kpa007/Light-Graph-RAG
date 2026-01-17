"""
인덱싱 모듈

청크 데이터를 LightRAG 인덱스에 저장
"""

import asyncio
from typing import List
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

from ..preprocessing.chunker import Chunk
from .lightrag_wrapper import LightRAGWrapper


@dataclass
class IndexingResult:
    """인덱싱 결과"""
    total_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    errors: List[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    started_at: datetime = None
    completed_at: datetime = None


class Indexer:
    """문서 인덱싱 관리자"""

    def __init__(
        self,
        rag_wrapper: LightRAGWrapper,
        batch_size: int = 10,
        show_progress: bool = True
    ):
        """
        Args:
            rag_wrapper: LightRAG 래퍼 인스턴스
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부
        """
        self.rag = rag_wrapper
        self.batch_size = batch_size
        self.show_progress = show_progress

    async def index_chunks_async(
        self,
        chunks: List[Chunk]
    ) -> IndexingResult:
        """
        청크 리스트를 인덱싱 (비동기)

        Args:
            chunks: 청크 리스트

        Returns:
            IndexingResult: 인덱싱 결과
        """
        result = IndexingResult(
            total_count=len(chunks),
            started_at=datetime.now()
        )

        # 진행률 표시
        iterator = tqdm(chunks, desc="인덱싱") if self.show_progress else chunks

        # 배치 처리
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            for chunk in batch:
                try:
                    # 청크 텍스트 생성 (메타데이터 포함)
                    chunk_text = self._format_chunk_for_indexing(chunk)

                    # 인덱싱
                    success = await self.rag.insert(chunk_text)

                    if success:
                        result.success_count += 1
                    else:
                        result.failed_count += 1
                        result.errors.append({
                            "chunk_id": chunk.id,
                            "error": "인덱싱 실패"
                        })

                except Exception as e:
                    result.failed_count += 1
                    result.errors.append({
                        "chunk_id": chunk.id,
                        "error": str(e)
                    })

                # 진행률 업데이트
                if self.show_progress and hasattr(iterator, 'update'):
                    iterator.update(1)

        result.completed_at = datetime.now()
        result.duration_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()

        return result

    def index_chunks(self, chunks: List[Chunk]) -> IndexingResult:
        """
        청크 리스트를 인덱싱 (동기)

        Args:
            chunks: 청크 리스트

        Returns:
            IndexingResult: 인덱싱 결과
        """
        return asyncio.run(self.index_chunks_async(chunks))

    def _format_chunk_for_indexing(self, chunk: Chunk) -> str:
        """
        청크를 인덱싱용 텍스트로 포맷팅

        메타데이터를 포함하여 더 나은 검색 결과 제공
        """
        # 기본 텍스트
        text_parts = [chunk.content]

        # 메타데이터 추가 (선택적)
        if chunk.source_section:
            text_parts.append(f"\n\n[출처: {chunk.source_section}]")

        if chunk.chunk_type == "table":
            text_parts.append("[표 데이터 포함]")

        return "\n".join(text_parts)


if __name__ == "__main__":
    from ..preprocessing import DocumentConverter, MarkdownParser, Chunker

    async def test():
        # 테스트 데이터 준비
        test_markdown = """
# 연차 규정

## 제1조 (목적)
이 규정은 회사의 연차 휴가에 관한 사항을 규정함을 목적으로 한다.

## 제2조 (연차 일수)
정규직 직원에게는 1년 근무 시 15일의 연차가 부여된다.
        """

        parser = MarkdownParser()
        doc = parser.parse(test_markdown)

        chunker = Chunker()
        chunks = chunker.chunk_document(doc)

        print(f"총 {len(chunks)}개 청크 생성")

        # LightRAG 초기화
        rag = LightRAGWrapper(working_dir="./test_index")

        # 인덱싱
        indexer = Indexer(rag, batch_size=5)
        result = indexer.index_chunks(chunks)

        print(f"\n인덱싱 완료:")
        print(f"  - 전체: {result.total_count}")
        print(f"  - 성공: {result.success_count}")
        print(f"  - 실패: {result.failed_count}")
        print(f"  - 소요 시간: {result.duration_seconds:.2f}초")

    asyncio.run(test())
