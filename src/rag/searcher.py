"""
검색 모듈

하이브리드 검색 수행 및 결과 융합
"""

import asyncio
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..preprocessing.chunker import Chunk
from .lightrag_wrapper import LightRAGWrapper


@dataclass
class SearchConfig:
    """검색 설정"""
    search_mode: str = "hybrid"  # naive, local, global, hybrid
    top_k: int = 10
    min_relevance_score: float = 0.5
    include_metadata: bool = True


@dataclass
class SearchResult:
    """검색 결과"""
    query: str
    answer: str
    context: str = ""
    search_mode: str = "hybrid"
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class Searcher:
    """하이브리드 검색 엔진"""

    def __init__(
        self,
        rag_wrapper: LightRAGWrapper,
        config: Optional[SearchConfig] = None
    ):
        """
        Args:
            rag_wrapper: LightRAG 래퍼 인스턴스
            config: 검색 설정
        """
        self.rag = rag_wrapper
        self.config = config or SearchConfig()

    async def search_async(
        self,
        query: str,
        mode: Optional[str] = None,
        only_context: bool = False
    ) -> SearchResult:
        """
        하이브리드 검색 수행 (비동기)

        Args:
            query: 검색 질의
            mode: 검색 모드 (설정 우선, 미지정 시 config 사용)
            only_context: 컨텍스트만 반환할지 여부

        Returns:
            SearchResult: 검색 결과
        """
        start_time = datetime.now()
        search_mode = mode or self.config.search_mode

        try:
            # LightRAG 검색 수행
            result_text = await self.rag.query(
                question=query,
                mode=search_mode,
                only_need_context=only_context
            )

            # 검색 결과 생성
            if only_context:
                answer = ""
                context = result_text
            else:
                answer = result_text
                context = ""

            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000

            return SearchResult(
                query=query,
                answer=answer,
                context=context,
                search_mode=search_mode,
                latency_ms=latency,
                metadata={
                    "timestamp": end_time.isoformat(),
                    "only_context": only_context
                }
            )

        except Exception as e:
            return SearchResult(
                query=query,
                answer=f"검색 중 오류 발생: {str(e)}",
                search_mode=search_mode,
                metadata={"error": str(e)}
            )

    def search(
        self,
        query: str,
        mode: Optional[str] = None,
        only_context: bool = False
    ) -> SearchResult:
        """
        하이브리드 검색 수행 (동기)

        Args:
            query: 검색 질의
            mode: 검색 모드
            only_context: 컨텍스트만 반환할지 여부

        Returns:
            SearchResult: 검색 결과
        """
        return asyncio.run(
            self.search_async(query, mode, only_context)
        )

    async def batch_search_async(
        self,
        queries: List[str],
        mode: Optional[str] = None
    ) -> List[SearchResult]:
        """
        배치 검색 (비동기)

        Args:
            queries: 질문 리스트
            mode: 검색 모드

        Returns:
            List[SearchResult]: 검색 결과 리스트
        """
        tasks = [
            self.search_async(query, mode)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

    def batch_search(
        self,
        queries: List[str],
        mode: Optional[str] = None
    ) -> List[SearchResult]:
        """
        배치 검색 (동기)

        Args:
            queries: 질문 리스트
            mode: 검색 모드

        Returns:
            List[SearchResult]: 검색 결과 리스트
        """
        return asyncio.run(
            self.batch_search_async(queries, mode)
        )


if __name__ == "__main__":
    from .lightrag_wrapper import LightRAGWrapper

    async def test():
        # LightRAG 초기화
        rag = LightRAGWrapper(working_dir="./test_index")

        # 테스트 데이터 삽입
        test_text = """
        제1조 (목적) 이 규정은 회사의 연차 휴가에 관한 사항을 규정함을 목적으로 한다.
        제2조 (적용범위) 정규직 전체 직원에게 적용한다.
        제3조 (연차 일수) 연차는 1년 근무 시 15일이 부여된다.
        제4조 (신청 방법) 연차는 최소 3일 전에 신청해야 한다.
        """

        print("인덱싱 중...")
        await rag.insert(test_text)

        # 검색기 초기화
        searcher = Searcher(
            rag,
            config=SearchConfig(
                search_mode="hybrid",
                top_k=5
            )
        )

        # 단일 검색 테스트
        print("\n=== 단일 검색 ===")
        result = searcher.search("연차는 며칠인가요?")
        print(f"질문: {result.query}")
        print(f"답변: {result.answer}")
        print(f"소요 시간: {result.latency_ms:.2f}ms")

        # 배치 검색 테스트
        print("\n=== 배치 검색 ===")
        queries = [
            "연차 신청은 언제 해야 하나요?",
            "누구에게 적용되나요?",
            "규정의 목적은 무엇인가요?"
        ]

        results = searcher.batch_search(queries)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 질문: {result.query}")
            print(f"   답변: {result.answer[:100]}...")
            print(f"   소요 시간: {result.latency_ms:.2f}ms")

    asyncio.run(test())
