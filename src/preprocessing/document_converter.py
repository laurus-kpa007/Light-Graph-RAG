"""
문서 변환 모듈

.docx 및 .md 파일을 마크다운으로 변환
"""

import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass

try:
    from markitdown import MarkItDown
except ImportError:
    raise ImportError(
        "markitdown이 설치되지 않았습니다. "
        "`pip install markitdown` 명령어로 설치해주세요."
    )


@dataclass
class ConversionResult:
    """문서 변환 결과"""
    original_path: str
    filename: str
    file_format: str  # "docx" or "md"
    markdown_path: str = None
    markdown_content: str = None
    converted_at: datetime = None
    file_size: int = 0
    has_tables: bool = False
    header_count: int = 0
    encoding: str = "utf-8"
    conversion_required: bool = True
    status: str = "success"  # success, failed, skipped
    error_message: str = None


class UnsupportedFormatError(Exception):
    """지원하지 않는 파일 형식 에러"""
    pass


class MarkdownConversionError(Exception):
    """마크다운 변환 실패 에러"""
    pass


class DocumentConverter:
    """
    .docx 및 .md 파일을 마크다운으로 변환하는 통합 클래스

    - .docx: markitdown 라이브러리를 활용하여 표 구조를 유지하면서 변환
    - .md: 파일 내용을 그대로 로드
    """

    def __init__(self, preserve_tables: bool = True):
        """
        Args:
            preserve_tables: 표 구조 유지 여부 (기본값: True)
        """
        self.markitdown_converter = MarkItDown()
        self.preserve_tables = preserve_tables
        self.supported_formats = ['.docx', '.md']

    def convert_file(self, file_path: str) -> str:
        """
        단일 파일을 마크다운으로 변환 또는 로드

        Args:
            file_path: 파일 경로 (.docx 또는 .md)

        Returns:
            str: 마크다운 텍스트

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            UnsupportedFormatError: 지원하지 않는 파일 형식
            MarkdownConversionError: 변환 실패 시
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        ext = file_path.suffix.lower()

        if ext == '.docx':
            return self._convert_docx(str(file_path))
        elif ext == '.md':
            return self._load_markdown(str(file_path))
        else:
            raise UnsupportedFormatError(
                f"지원하지 않는 형식: {ext}. "
                f"지원 형식: {', '.join(self.supported_formats)}"
            )

    def convert_directory(
        self,
        dir_path: str,
        output_dir: str = None,
        recursive: bool = False
    ) -> List[ConversionResult]:
        """
        디렉토리 내 모든 .docx 및 .md 파일을 일괄 변환

        Args:
            dir_path: 파일들이 있는 디렉토리 경로
            output_dir: 변환된 마크다운 저장 경로 (선택)
            recursive: 하위 디렉토리 포함 여부

        Returns:
            List[ConversionResult]: 변환 결과 메타데이터 리스트

        Raises:
            NotADirectoryError: 디렉토리가 아닐 때
        """
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise NotADirectoryError(f"디렉토리가 아닙니다: {dir_path}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # 파일 패턴
        pattern = "**/*" if recursive else "*"

        # .docx 및 .md 파일 찾기
        for ext in self.supported_formats:
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    result = self._convert_single_file(
                        file_path,
                        output_dir
                    )
                    results.append(result)

        return results

    def _convert_single_file(
        self,
        file_path: Path,
        output_dir: Path = None
    ) -> ConversionResult:
        """단일 파일 변환 및 결과 생성"""
        try:
            # 변환
            markdown_content = self.convert_file(str(file_path))

            # 출력 경로 설정
            markdown_path = None
            if output_dir:
                markdown_path = output_dir / f"{file_path.stem}.md"
                markdown_path.write_text(markdown_content, encoding='utf-8')

            # 메타데이터 추출
            has_tables = self._has_tables(markdown_content)
            header_count = self._count_headers(markdown_content)

            return ConversionResult(
                original_path=str(file_path),
                filename=file_path.name,
                file_format=file_path.suffix.lower().replace('.', ''),
                markdown_path=str(markdown_path) if markdown_path else None,
                markdown_content=markdown_content,
                converted_at=datetime.now(),
                file_size=file_path.stat().st_size,
                has_tables=has_tables,
                header_count=header_count,
                conversion_required=(file_path.suffix.lower() == '.docx'),
                status="success"
            )

        except Exception as e:
            return ConversionResult(
                original_path=str(file_path),
                filename=file_path.name,
                file_format=file_path.suffix.lower().replace('.', ''),
                converted_at=datetime.now(),
                status="failed",
                error_message=str(e)
            )

    def _convert_docx(self, docx_path: str) -> str:
        """
        .docx 파일을 마크다운으로 변환

        Args:
            docx_path: .docx 파일 경로

        Returns:
            str: 변환된 마크다운 텍스트
        """
        try:
            result = self.markitdown_converter.convert(docx_path)
            markdown_text = result.text_content

            # 후처리
            markdown_text = self._post_process_markdown(markdown_text)

            return markdown_text

        except Exception as e:
            raise MarkdownConversionError(
                f".docx 변환 실패: {docx_path}\n오류: {str(e)}"
            )

    def _load_markdown(self, md_path: str) -> str:
        """
        .md 파일을 로드

        Args:
            md_path: .md 파일 경로

        Returns:
            str: 마크다운 텍스트
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()

            # 후처리
            markdown_text = self._post_process_markdown(markdown_text)

            return markdown_text

        except UnicodeDecodeError:
            # UTF-8 실패 시 다른 인코딩 시도
            try:
                with open(md_path, 'r', encoding='cp949') as f:
                    markdown_text = f.read()
                return self._post_process_markdown(markdown_text)
            except Exception as e:
                raise MarkdownConversionError(
                    f".md 로드 실패: {md_path}\n오류: {str(e)}"
                )
        except Exception as e:
            raise MarkdownConversionError(
                f".md 로드 실패: {md_path}\n오류: {str(e)}"
            )

    def _post_process_markdown(self, markdown_text: str) -> str:
        """
        변환된 마크다운 후처리

        - 불필요한 공백 제거
        - 표 포맷팅 정규화
        - 헤더 계층 구조 검증
        """
        # 연속된 빈 줄을 최대 2개로 제한
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)

        # 라인 끝 공백 제거
        lines = [line.rstrip() for line in markdown_text.split('\n')]
        markdown_text = '\n'.join(lines)

        # 문서 시작/끝 공백 제거
        markdown_text = markdown_text.strip()

        return markdown_text

    def _has_tables(self, markdown_text: str) -> bool:
        """마크다운 텍스트에 표가 있는지 확인"""
        # 마크다운 표 패턴: | ... | ... |
        table_pattern = r'\|.+\|.+\|'
        return bool(re.search(table_pattern, markdown_text))

    def _count_headers(self, markdown_text: str) -> int:
        """마크다운 헤더 개수 세기"""
        # # 으로 시작하는 라인 카운트
        header_pattern = r'^#{1,6}\s+.+'
        return len(re.findall(header_pattern, markdown_text, re.MULTILINE))


if __name__ == "__main__":
    # 테스트 코드
    converter = DocumentConverter()

    # 테스트 파일 경로 (실제 경로로 변경 필요)
    test_file = "test.docx"  # 또는 "test.md"

    try:
        markdown = converter.convert_file(test_file)
        print(f"변환 성공!\n\n{markdown[:500]}...")
    except FileNotFoundError:
        print(f"테스트 파일이 없습니다: {test_file}")
    except Exception as e:
        print(f"에러: {e}")
