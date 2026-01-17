"""
Ollama 클라이언트

Ollama API와 통신하는 클라이언트
"""

import httpx
from typing import List, Dict, Optional, AsyncIterator


class OllamaClient:
    """Ollama API 클라이언트"""

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 120.0):
        """
        Args:
            host: Ollama 서버 주소
            timeout: 요청 타임아웃 (초)
        """
        self.host = host.rstrip('/')
        self.timeout = timeout

    async def generate_async(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """
        텍스트 생성 (비동기)

        Args:
            model: 모델명 (예: gemma2:latest)
            prompt: 입력 프롬프트
            system: 시스템 프롬프트
            temperature: 생성 다양성 (0.0 ~ 1.0)
            max_tokens: 최대 토큰 수
            stream: 스트리밍 여부

        Returns:
            str: 생성된 텍스트
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]

    async def embed_async(
        self,
        model: str,
        texts: List[str]
    ) -> List[List[float]]:
        """
        텍스트 임베딩 생성 (비동기)

        Args:
            model: 임베딩 모델명 (예: bge-m3:latest)
            texts: 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        embeddings = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                response = await client.post(
                    f"{self.host}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])

        return embeddings

    async def list_models_async(self) -> List[Dict]:
        """사용 가능한 모델 목록 조회 (비동기)"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])

    async def check_health_async(self) -> bool:
        """Ollama 서버 상태 확인 (비동기)"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except:
            return False

    async def pull_model_async(self, model: str) -> bool:
        """모델 다운로드 (비동기)"""
        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(
                    f"{self.host}/api/pull",
                    json={"name": model}
                )
                response.raise_for_status()
                return True
        except Exception as e:
            print(f"모델 다운로드 실패: {e}")
            return False


if __name__ == "__main__":
    import asyncio

    async def test():
        client = OllamaClient()

        # 서버 상태 확인
        print("Ollama 서버 확인 중...")
        is_healthy = await client.check_health_async()
        print(f"서버 상태: {'정상' if is_healthy else '연결 실패'}")

        if not is_healthy:
            print("Ollama 서버가 실행 중인지 확인해주세요.")
            return

        # 모델 목록 조회
        print("\n사용 가능한 모델:")
        models = await client.list_models_async()
        for model in models:
            print(f"  - {model['name']}")

        # 텍스트 생성 테스트 (gemma2 모델이 있는 경우)
        if any("gemma2" in m["name"] for m in models):
            print("\n텍스트 생성 테스트:")
            response = await client.generate_async(
                model="gemma2:latest",
                prompt="안녕하세요를 영어로 번역해주세요.",
                temperature=0.7
            )
            print(f"응답: {response}")

        # 임베딩 테스트 (bge-m3 모델이 있는 경우)
        if any("bge-m3" in m["name"] for m in models):
            print("\n임베딩 생성 테스트:")
            embeddings = await client.embed_async(
                model="bge-m3:latest",
                texts=["안녕하세요", "Hello"]
            )
            print(f"임베딩 차원: {len(embeddings[0])}")
            print(f"첫 번째 임베딩 (처음 5개 값): {embeddings[0][:5]}")

    asyncio.run(test())
