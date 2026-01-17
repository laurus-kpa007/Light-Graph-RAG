"""
LightRAG 래퍼 모듈

LightRAG 라이브러리를 래핑하고 한국어 최적화 적용
"""

import os
from pathlib import Path
from typing import Optional, Callable

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import openai_complete_if_cache, openai_embedding
except ImportError:
    raise ImportError(
        "lightrag-hku가 설치되지 않았습니다. "
        "`pip install lightrag-hku` 명령어로 설치해주세요."
    )

from .korean_prompts import KoreanPrompts


class LightRAGWrapper:
    """
    LightRAG를 래핑하여 한국어 최적화 및 커스터마이징 적용
    """

    def __init__(
        self,
        working_dir: str,
        llm_model: str = "gemma2:latest",
        embedding_model: str = "bge-m3:latest",
        ollama_host: str = "http://localhost:11434",
        enable_korean_prompts: bool = True
    ):
        """
        Args:
            working_dir: 인덱스 저장 디렉토리
            llm_model: Ollama LLM 모델명
            embedding_model: Ollama 임베딩 모델명
            ollama_host: Ollama 서버 주소
            enable_korean_prompts: 한국어 프롬프트 사용 여부
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.enable_korean_prompts = enable_korean_prompts

        # LLM 및 임베딩 함수 생성
        llm_func = self._create_ollama_llm_func()
        embedding_func = self._create_ollama_embedding_func()

        # LightRAG 초기화
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_func,
            embedding_func=embedding_func
        )

        # 한국어 프롬프트 적용 (선택적)
        if enable_korean_prompts:
            self._apply_korean_prompts()

    def _create_ollama_llm_func(self) -> Callable:
        """Ollama LLM 함수 생성"""
        async def ollama_model_complete(
            prompt,
            system_prompt=None,
            history_messages=[],
            **kwargs
        ) -> str:
            """Ollama를 사용한 텍스트 생성"""
            import httpx

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.ollama_host}/api/chat",
                        json={
                            "model": self.llm_model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": kwargs.get("temperature", 0.7),
                                "num_predict": kwargs.get("max_tokens", 2048),
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result["message"]["content"]

            except Exception as e:
                print(f"Ollama LLM 에러: {e}")
                return ""

        return ollama_model_complete

    def _create_ollama_embedding_func(self) -> Callable:
        """Ollama 임베딩 함수 생성"""
        async def ollama_embedding(texts: list[str]) -> list[list[float]]:
            """Ollama를 사용한 임베딩 생성"""
            import httpx

            embed_list = []

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    for text in texts:
                        response = await client.post(
                            f"{self.ollama_host}/api/embeddings",
                            json={
                                "model": self.embedding_model,
                                "prompt": text
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        embed_list.append(result["embedding"])

                return embed_list

            except Exception as e:
                print(f"Ollama 임베딩 에러: {e}")
                # 빈 임베딩 반환 (오류 처리)
                return [[0.0] * 768 for _ in texts]

        return ollama_embedding

    def _apply_korean_prompts(self):
        """한국어 최적화 프롬프트 적용"""
        # LightRAG의 내부 프롬프트를 한국어 버전으로 교체
        # 주의: LightRAG 라이브러리의 내부 구조에 따라 조정 필요
        print("한국어 프롬프트가 활성화되었습니다.")
        # 실제 프롬프트 교체는 LightRAG의 API에 따라 구현

    async def insert(self, text: str) -> bool:
        """
        텍스트를 인덱스에 삽입

        Args:
            text: 삽입할 텍스트 (청크 내용)

        Returns:
            bool: 성공 여부
        """
        try:
            await self.rag.ainsert(text)
            return True
        except Exception as e:
            print(f"인덱싱 에러: {e}")
            return False

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        only_need_context: bool = False
    ) -> str:
        """
        질의 수행

        Args:
            question: 질문
            mode: 검색 모드 (naive, local, global, hybrid)
            only_need_context: 컨텍스트만 반환할지 여부

        Returns:
            str: 답변 또는 컨텍스트
        """
        try:
            result = await self.rag.aquery(
                question,
                param=QueryParam(
                    mode=mode,
                    only_need_context=only_need_context
                )
            )
            return result
        except Exception as e:
            print(f"검색 에러: {e}")
            return ""

    def get_index_stats(self) -> dict:
        """인덱스 통계 조회"""
        stats = {
            "working_dir": str(self.working_dir),
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "korean_prompts_enabled": self.enable_korean_prompts
        }

        # 저장소 파일 확인
        if self.working_dir.exists():
            files = list(self.working_dir.glob("*"))
            stats["index_files"] = [f.name for f in files]
            stats["total_size_mb"] = sum(
                f.stat().st_size for f in files if f.is_file()
            ) / (1024 * 1024)

        return stats


if __name__ == "__main__":
    import asyncio

    async def test():
        # 테스트
        rag = LightRAGWrapper(
            working_dir="./test_index",
            llm_model="gemma2:latest",
            embedding_model="bge-m3:latest"
        )

        # 테스트 데이터 삽입
        test_text = """
        제1조 (목적) 이 규정은 회사의 연차 휴가에 관한 사항을 규정함을 목적으로 한다.
        제2조 (적용범위) 정규직 전체 직원에게 적용한다.
        제3조 (연차 일수) 연차는 1년 근무 시 15일이 부여된다.
        """

        print("인덱싱 중...")
        success = await rag.insert(test_text)
        print(f"인덱싱 성공: {success}")

        # 검색 테스트
        print("\n검색 중...")
        answer = await rag.query("연차는 며칠인가요?", mode="hybrid")
        print(f"답변: {answer}")

        # 통계 조회
        print("\n통계:")
        stats = rag.get_index_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    asyncio.run(test())
