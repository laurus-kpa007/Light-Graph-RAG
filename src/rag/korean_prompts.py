"""
한국어 최적화 프롬프트

LightRAG에서 사용할 한국어 엔티티 추출 및 답변 생성 프롬프트
"""


class KoreanPrompts:
    """한국어 최적화 프롬프트 모음"""

    # 엔티티 추출 프롬프트
    ENTITY_EXTRACTION = """당신은 한국어 문서에서 핵심 개념과 엔티티를 추출하는 전문가입니다.

주어진 텍스트에서 다음 유형의 엔티티를 추출하세요:
1. **조직**: 회사, 부서, 팀, 위원회 등
2. **인물**: 직책, 역할, 담당자 등
3. **개념**: 규정명, 정책명, 제도명, 용어 등
4. **날짜/기간**: 구체적인 날짜, 기간, 시한 정보
5. **수치**: 금액, 일수, 비율, 기준 등 구체적 수치
6. **절차**: 업무 프로세스, 단계, 방법 등

텍스트:
---
{input_text}
---

출력 형식 (JSON):
{{
  "entities": [
    {{
      "name": "엔티티명",
      "type": "조직|인물|개념|날짜|수치|절차",
      "description": "간단한 설명 (선택)"
    }}
  ]
}}

주의사항:
- 모든 출력은 한국어로 작성
- 동일한 의미를 가진 엔티티는 하나로 통합
- 엔티티명은 문서에 나온 원문 그대로 사용
- JSON 형식을 정확히 준수
"""

    # 관계 추출 프롬프트
    RELATIONSHIP_EXTRACTION = """다음 엔티티들 사이의 관계를 파악하여 추출하세요.

엔티티 목록:
{entities}

텍스트:
---
{input_text}
---

관계 유형:
- 소속: A는 B에 소속됨
- 규정: A는 B에 의해 규정됨
- 절차: A 이후 B가 진행됨
- 조건: A인 경우 B가 적용됨
- 기준: A는 B를 기준으로 함
- 참조: A는 B를 참조함
- 담당: A는 B를 담당함
- 승인: A는 B의 승인이 필요함

출력 형식 (JSON):
{{
  "relationships": [
    {{
      "source": "엔티티1",
      "relation": "관계 유형",
      "target": "엔티티2",
      "context": "관계에 대한 간단한 설명"
    }}
  ]
}}

주의사항:
- 텍스트에 명시적으로 드러난 관계만 추출
- 추론이나 가정은 배제
- JSON 형식을 정확히 준수
"""

    # 답변 생성 프롬프트
    ANSWER_GENERATION = """당신은 사내 규정 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 정확하게 답변하세요.

질문:
{question}

관련 컨텍스트:
---
{context}
---

답변 작성 지침:
1. 컨텍스트에 근거하여 정확하게 답변
2. 규정명, 조항, 수치 등을 명시하여 신뢰성 제공
3. 모호한 경우 "명확하지 않습니다"라고 표시
4. 컨텍스트에 없는 내용은 추측하지 말것
5. 답변은 간결하고 명확하게 (3-5문장 권장)
6. 가능한 경우 단계별로 설명
7. 관련 규정이나 근거를 함께 제시

답변:
"""

    # 요약 생성 프롬프트
    SUMMARY_GENERATION = """다음 텍스트의 핵심 내용을 간결하게 요약하세요.

텍스트:
---
{input_text}
---

요약 지침:
- 1-2문장으로 핵심만 요약
- 중요한 수치, 날짜, 조건 등은 포함
- 불필요한 수식어 제거
- 한국어로 작성

요약:
"""

    # 키워드 추출 프롬프트
    KEYWORD_EXTRACTION = """다음 텍스트에서 핵심 키워드를 추출하세요.

텍스트:
---
{input_text}
---

키워드 추출 지침:
- 3-7개의 핵심 키워드 추출
- 명사 위주로 추출
- 문서의 주제를 대표하는 단어 선택
- 쉼표로 구분하여 나열

키워드:
"""

    @classmethod
    def get_entity_extraction_prompt(cls, text: str) -> str:
        """엔티티 추출 프롬프트 생성"""
        return cls.ENTITY_EXTRACTION.format(input_text=text)

    @classmethod
    def get_relationship_extraction_prompt(cls, text: str, entities: str) -> str:
        """관계 추출 프롬프트 생성"""
        return cls.RELATIONSHIP_EXTRACTION.format(
            input_text=text,
            entities=entities
        )

    @classmethod
    def get_answer_generation_prompt(cls, question: str, context: str) -> str:
        """답변 생성 프롬프트 생성"""
        return cls.ANSWER_GENERATION.format(
            question=question,
            context=context
        )

    @classmethod
    def get_summary_prompt(cls, text: str) -> str:
        """요약 생성 프롬프트 생성"""
        return cls.SUMMARY_GENERATION.format(input_text=text)

    @classmethod
    def get_keyword_extraction_prompt(cls, text: str) -> str:
        """키워드 추출 프롬프트 생성"""
        return cls.KEYWORD_EXTRACTION.format(input_text=text)


if __name__ == "__main__":
    # 테스트
    test_text = """
    제1조 (목적) 이 규정은 회사의 연차 휴가에 관한 사항을 규정함을 목적으로 한다.
    제2조 (적용범위) 정규직 전체 직원에게 적용한다. 연차는 15일이 부여된다.
    """

    print("=== 엔티티 추출 프롬프트 ===")
    print(KoreanPrompts.get_entity_extraction_prompt(test_text))

    print("\n=== 답변 생성 프롬프트 ===")
    print(KoreanPrompts.get_answer_generation_prompt(
        "연차는 며칠인가요?",
        test_text
    ))
