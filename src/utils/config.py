"""
설정 관리 모듈
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class PathConfig:
    """경로 설정"""
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    index_dir: Path


@dataclass
class LLMConfig:
    """LLM 설정"""
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "gemma2:latest"
    embedding_model: str = "bge-m3:latest"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 120.0


@dataclass
class RAGConfig:
    """RAG 설정"""
    search_mode: str = "hybrid"
    top_k: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 50
    preserve_tables: bool = True


@dataclass
class Config:
    """전역 설정"""
    paths: PathConfig
    llm: LLMConfig
    rag: RAGConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 경로 설정
        project_root = Path(config_dict.get('project_root', '.'))
        paths = PathConfig(
            project_root=project_root,
            data_dir=project_root / "data",
            raw_dir=project_root / "data" / "raw",
            processed_dir=project_root / "data" / "processed",
            index_dir=project_root / "data" / "index"
        )

        # LLM 설정
        llm_dict = config_dict.get('llm', {})
        llm = LLMConfig(**llm_dict)

        # RAG 설정
        rag_dict = config_dict.get('rag', {})
        rag = RAGConfig(**rag_dict)

        return cls(paths=paths, llm=llm, rag=rag)

    @classmethod
    def default(cls) -> 'Config':
        """기본 설정 생성"""
        project_root = Path.cwd()

        paths = PathConfig(
            project_root=project_root,
            data_dir=project_root / "data",
            raw_dir=project_root / "data" / "raw",
            processed_dir=project_root / "data" / "processed",
            index_dir=project_root / "data" / "index"
        )

        return cls(
            paths=paths,
            llm=LLMConfig(),
            rag=RAGConfig()
        )


def load_config(config_path: Optional[str] = None) -> Config:
    """설정 로드"""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config.default()


if __name__ == "__main__":
    config = Config.default()
    print(f"프로젝트 루트: {config.paths.project_root}")
    print(f"LLM 모델: {config.llm.llm_model}")
    print(f"검색 모드: {config.rag.search_mode}")
