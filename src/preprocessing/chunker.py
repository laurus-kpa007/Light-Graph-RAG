"""
청크 분할 모듈

구조화된 문서를 의미 단위의 청크로 분할
"""

import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .markdown_parser import MarkdownDocument, Section


@dataclass
class ChunkConfig:
    """청크 분할 설정"""
    min_chunk_size: int = 100  # 최소 청크 크기 (문자)
    max_chunk_size: int = 1000  # 최대 청크 크기 (문자)
    overlap_size: int = 50  # 오버랩 크기 (문자)
    preserve_tables: bool = True  # 표 보존 여부
    split_by_header: bool = True  # 헤더 기준 분할 여부


@dataclass
class Chunk:
    """청크 데이터 클래스"""
    id: str  # 고유 ID
    content: str  # 청크 내용
    metadata: Dict = field(default_factory=dict)  # 메타데이터
    source_section: str = ""  # 원본 섹션 경로
    chunk_type: str = "text"  # text, table, mixed
    chunk_index: int = 0  # 청크 인덱스
    char_count: int = 0  # 문자 수

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.content)


class Chunker:
    """의미 단위 청크 분할기"""

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()

    def chunk_document(self, doc: MarkdownDocument) -> List[Chunk]:
        """
        문서를 청크로 분할

        Args:
            doc: 파싱된 마크다운 문서

        Returns:
            List[Chunk]: 청크 리스트
        """
        all_chunks = []
        global_chunk_index = 0

        for section in doc.sections:
            chunks = self._chunk_section_recursive(section, "")

            # 청크 인덱스 및 ID 할당
            for chunk in chunks:
                chunk.chunk_index = global_chunk_index
                chunk.id = self._create_chunk_id(chunk.source_section, global_chunk_index)
                global_chunk_index += 1

            all_chunks.extend(chunks)

        return all_chunks

    def _chunk_section_recursive(
        self,
        section: Section,
        parent_path: str
    ) -> List[Chunk]:
        """재귀적으로 섹션을 청크로 분할"""
        chunks = []

        # 현재 섹션 경로
        current_path = f"{parent_path} > {section.title}" if parent_path else section.title

        # 현재 섹션의 청크 생성
        section_chunks = self._chunk_section(section, current_path)
        chunks.extend(section_chunks)

        # 하위 섹션 재귀 처리
        for subsection in section.subsections:
            sub_chunks = self._chunk_section_recursive(subsection, current_path)
            chunks.extend(sub_chunks)

        return chunks

    def _chunk_section(self, section: Section, section_path: str) -> List[Chunk]:
        """단일 섹션을 청크로 분할"""
        if not section.content.strip():
            return []

        content_size = len(section.content)

        # 표 포함 여부 확인
        if section.has_table and self.config.preserve_tables:
            return self._chunk_section_with_tables(section, section_path)

        # 크기에 따른 분할 전략
        if content_size < self.config.min_chunk_size:
            # 작은 섹션: 단일 청크
            return [self._create_chunk(
                content=section.content,
                section_path=section_path,
                chunk_type="text"
            )]

        elif content_size <= self.config.max_chunk_size:
            # 적정 크기: 단일 청크
            return [self._create_chunk(
                content=section.content,
                section_path=section_path,
                chunk_type="text"
            )]

        else:
            # 큰 섹션: 다중 청크 분할 (오버랩 적용)
            return self._split_long_text(
                section.content,
                section_path
            )

    def _chunk_section_with_tables(
        self,
        section: Section,
        section_path: str
    ) -> List[Chunk]:
        """표가 포함된 섹션 처리"""
        chunks = []
        content = section.content

        # 표와 일반 텍스트 분리
        lines = content.split('\n')
        current_text = []
        in_table = False
        table_lines = []

        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                if not in_table:
                    # 표 시작 전 텍스트 처리
                    if current_text:
                        text_content = '\n'.join(current_text).strip()
                        if len(text_content) >= self.config.min_chunk_size:
                            chunks.append(self._create_chunk(
                                content=text_content,
                                section_path=section_path,
                                chunk_type="text"
                            ))
                        current_text = []
                    in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    # 표 종료 - 표를 별도 청크로
                    table_content = '\n'.join(table_lines).strip()
                    chunks.append(self._create_chunk(
                        content=table_content,
                        section_path=section_path,
                        chunk_type="table",
                        metadata={"has_table": True}
                    ))
                    table_lines = []
                    in_table = False
                current_text.append(line)

        # 마지막 표 처리
        if table_lines:
            table_content = '\n'.join(table_lines).strip()
            chunks.append(self._create_chunk(
                content=table_content,
                section_path=section_path,
                chunk_type="table",
                metadata={"has_table": True}
            ))

        # 마지막 텍스트 처리
        if current_text:
            text_content = '\n'.join(current_text).strip()
            if len(text_content) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(
                    content=text_content,
                    section_path=section_path,
                    chunk_type="text"
                ))

        return chunks

    def _split_long_text(
        self,
        text: str,
        section_path: str
    ) -> List[Chunk]:
        """긴 텍스트를 최대 크기 기준으로 분할 (오버랩 적용)"""
        chunks = []
        text_length = len(text)
        start = 0

        while start < text_length:
            # 청크 끝 위치 계산
            end = start + self.config.max_chunk_size

            if end >= text_length:
                # 마지막 청크
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append(self._create_chunk(
                        content=chunk_text,
                        section_path=section_path,
                        chunk_type="text"
                    ))
                break

            # 문장 경계에서 분할 (마침표, 줄바꿈 등)
            chunk_text = text[start:end]

            # 마침표나 줄바꿈 찾기
            boundary_chars = ['. ', '.\n', '\n\n']
            best_boundary = -1

            for boundary in boundary_chars:
                pos = chunk_text.rfind(boundary)
                if pos > self.config.max_chunk_size * 0.7:  # 최소 70% 이상 사용
                    best_boundary = pos + len(boundary)
                    break

            if best_boundary > 0:
                end = start + best_boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    content=chunk_text,
                    section_path=section_path,
                    chunk_type="text"
                ))

            # 다음 시작 위치 (오버랩 적용)
            start = end - self.config.overlap_size

        return chunks

    def _create_chunk(
        self,
        content: str,
        section_path: str,
        chunk_type: str = "text",
        metadata: Dict = None
    ) -> Chunk:
        """청크 객체 생성"""
        chunk_metadata = {
            "section_path": section_path,
            "chunk_type": chunk_type,
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }

        return Chunk(
            id="",  # 나중에 할당
            content=content,
            metadata=chunk_metadata,
            source_section=section_path,
            chunk_type=chunk_type,
            char_count=len(content)
        )

    def _create_chunk_id(self, section_path: str, index: int) -> str:
        """청크 고유 ID 생성"""
        # 섹션 경로와 인덱스를 조합하여 해시 생성
        content = f"{section_path}_{index}"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return f"chunk_{hash_obj.hexdigest()[:12]}"


if __name__ == "__main__":
    # 테스트 코드
    from .markdown_parser import MarkdownParser

    test_markdown = """
# 제1장 총칙

## 제1조 (목적)

이 규정은 회사의 운영에 필요한 사항을 규정함을 목적으로 한다. 모든 임직원은 이 규정을 준수해야 하며, 위반 시 징계 대상이 될 수 있다.

## 제2조 (적용범위)

다음 표는 적용 대상을 나타낸다:

| 구분 | 대상 | 비고 |
|------|------|------|
| 정규직 | 전체 | - |
| 계약직 | 일부 | 별도 규정 |
| 인턴 | 해당 없음 | - |

위 표에 명시된 대로 적용한다.

# 제2장 인사

## 제3조 (채용)

채용 절차는 다음과 같다. 1) 서류 심사, 2) 면접 전형, 3) 최종 합격 통보. 각 단계별로 세부 기준이 적용되며, 이는 인사팀에서 관리한다.
"""

    parser = MarkdownParser()
    doc = parser.parse(test_markdown)

    chunker = Chunker(ChunkConfig(
        min_chunk_size=50,
        max_chunk_size=200,
        overlap_size=20
    ))

    chunks = chunker.chunk_document(doc)

    print(f"총 {len(chunks)}개 청크 생성\n")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 청크 {i+1} ({chunk.chunk_type}) ---")
        print(f"ID: {chunk.id}")
        print(f"섹션: {chunk.source_section}")
        print(f"길이: {chunk.char_count}자")
        print(f"내용: {chunk.content[:100]}...")
