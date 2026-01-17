"""
마크다운 파싱 모듈

마크다운 텍스트를 구조화된 문서 객체로 변환
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Section:
    """문서의 한 섹션을 표현"""
    level: int  # 헤더 레벨 (1-6)
    title: str  # 섹션 제목
    content: str  # 섹션 내용
    subsections: List['Section'] = field(default_factory=list)
    has_table: bool = False
    table_data: List[Dict] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0

    def get_full_path(self) -> str:
        """섹션의 전체 경로 반환 (예: "1장 > 1.1절 > 1.1.1항")"""
        return self.title


@dataclass
class MarkdownDocument:
    """마크다운 문서를 표현하는 데이터 클래스"""
    sections: List[Section] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    raw_text: str = ""

    def get_total_sections(self) -> int:
        """전체 섹션 수 반환"""
        count = len(self.sections)
        for section in self.sections:
            count += self._count_subsections(section)
        return count

    def _count_subsections(self, section: Section) -> int:
        """재귀적으로 하위 섹션 카운트"""
        count = len(section.subsections)
        for subsection in section.subsections:
            count += self._count_subsections(subsection)
        return count


class ParsingError(Exception):
    """파싱 에러"""
    pass


class MarkdownParser:
    """마크다운 텍스트를 파싱하여 구조화"""

    def __init__(self):
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'\|.+\|.+\|')

    def parse(self, markdown_text: str) -> MarkdownDocument:
        """
        마크다운 텍스트를 파싱

        Args:
            markdown_text: 마크다운 형식의 텍스트

        Returns:
            MarkdownDocument: 구조화된 문서 객체

        Raises:
            ParsingError: 파싱 실패 시
        """
        if not markdown_text or not markdown_text.strip():
            return MarkdownDocument(raw_text=markdown_text)

        try:
            # 섹션 추출
            sections = self._extract_sections(markdown_text)

            # 계층 구조 구축
            hierarchical_sections = self._build_hierarchy(sections)

            # 문서 객체 생성
            doc = MarkdownDocument(
                sections=hierarchical_sections,
                raw_text=markdown_text,
                metadata={
                    "total_sections": len(sections),
                    "has_tables": any(s.has_table for s in sections),
                    "max_header_level": max([s.level for s in sections]) if sections else 0
                }
            )

            return doc

        except Exception as e:
            raise ParsingError(f"마크다운 파싱 실패: {str(e)}")

    def _extract_sections(self, text: str) -> List[Section]:
        """헤더를 기준으로 섹션 추출"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        line_num = 0

        for i, line in enumerate(lines):
            # 헤더 체크
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # 이전 섹션 저장
                if current_section:
                    content_text = '\n'.join(current_content).strip()
                    current_section.content = content_text
                    current_section.line_end = i - 1
                    current_section.has_table = self._has_table_in_text(content_text)
                    if current_section.has_table:
                        current_section.table_data = self._parse_tables(content_text)
                    sections.append(current_section)

                # 새 섹션 시작
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = Section(
                    level=level,
                    title=title,
                    content="",
                    line_start=i
                )
                current_content = []

            else:
                # 헤더가 아니면 현재 섹션의 내용에 추가
                if current_section:
                    current_content.append(line)
                else:
                    # 문서 시작 부분에 헤더가 없는 경우
                    if not sections and line.strip():
                        # 암시적 섹션 생성
                        current_section = Section(
                            level=1,
                            title="(제목 없음)",
                            content="",
                            line_start=i
                        )
                        current_content = [line]

        # 마지막 섹션 저장
        if current_section:
            content_text = '\n'.join(current_content).strip()
            current_section.content = content_text
            current_section.line_end = len(lines) - 1
            current_section.has_table = self._has_table_in_text(content_text)
            if current_section.has_table:
                current_section.table_data = self._parse_tables(content_text)
            sections.append(current_section)

        return sections

    def _build_hierarchy(self, sections: List[Section]) -> List[Section]:
        """섹션 간 계층 구조 구축"""
        if not sections:
            return []

        root_sections = []
        stack = []  # (level, section) 스택

        for section in sections:
            # 현재 섹션보다 레벨이 높거나 같은 것들을 스택에서 제거
            while stack and stack[-1][0] >= section.level:
                stack.pop()

            if not stack:
                # 최상위 섹션
                root_sections.append(section)
            else:
                # 하위 섹션으로 추가
                parent_section = stack[-1][1]
                parent_section.subsections.append(section)

            stack.append((section.level, section))

        return root_sections

    def _has_table_in_text(self, text: str) -> bool:
        """텍스트에 표가 있는지 확인"""
        return bool(self.table_pattern.search(text))

    def _parse_tables(self, text: str) -> List[Dict]:
        """마크다운 표 파싱"""
        tables = []
        lines = text.split('\n')
        in_table = False
        current_table_lines = []

        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                in_table = True
                current_table_lines.append(line)
            else:
                if in_table and current_table_lines:
                    # 표 종료
                    table_data = self._parse_single_table(current_table_lines)
                    if table_data:
                        tables.append(table_data)
                    current_table_lines = []
                in_table = False

        # 마지막 표 처리
        if current_table_lines:
            table_data = self._parse_single_table(current_table_lines)
            if table_data:
                tables.append(table_data)

        return tables

    def _parse_single_table(self, table_lines: List[str]) -> Dict:
        """단일 표 파싱"""
        if len(table_lines) < 2:
            return None

        # 헤더 파싱
        header_line = table_lines[0].strip('|').strip()
        headers = [h.strip() for h in header_line.split('|')]

        # 데이터 파싱 (구분선 제외)
        rows = []
        for line in table_lines[2:]:  # 0: 헤더, 1: 구분선, 2~: 데이터
            if not line.strip() or set(line.replace('|', '').strip()) == {'-', ':'}:
                continue
            row_data = line.strip('|').strip()
            cells = [c.strip() for c in row_data.split('|')]
            if cells:
                rows.append(cells)

        return {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers)
        }


if __name__ == "__main__":
    # 테스트 코드
    test_markdown = """
# 제1장 총칙

## 제1조 (목적)

이 규정은 회사의 운영에 필요한 사항을 규정함을 목적으로 한다.

## 제2조 (적용범위)

| 구분 | 대상 | 비고 |
|------|------|------|
| 정규직 | 전체 | - |
| 계약직 | 일부 | 별도 규정 |

# 제2장 인사

## 제3조 (채용)

채용에 관한 사항은 다음과 같다.
"""

    parser = MarkdownParser()
    doc = parser.parse(test_markdown)

    print(f"총 섹션 수: {doc.get_total_sections()}")
    print(f"최상위 섹션 수: {len(doc.sections)}")

    for section in doc.sections:
        print(f"\n{'#' * section.level} {section.title}")
        print(f"  - 내용 길이: {len(section.content)}")
        print(f"  - 표 포함: {section.has_table}")
        print(f"  - 하위 섹션: {len(section.subsections)}")
