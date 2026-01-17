# Light GraphRAG - 사내 규정 질의응답 시스템

로컬 LLM(Ollama + Gemma3)을 활용한 사내 규정 문서 질의응답 시스템

## 프로젝트 개요

사내 규정 문서(.docx, .md)에 대한 자연어 질의응답이 가능한 Light GraphRAG 시스템입니다. 모든 데이터는 로컬에서 처리되어 보안이 보장됩니다.

### 주요 특징

- **다양한 입력 형식**: Word 문서(.docx)와 마크다운(.md) 모두 지원
- **로컬 LLM**: Ollama 기반 Gemma3 모델 사용 (외부 API 호출 없음)
- **한국어 최적화**: 한국어 엔티티 추출에 최적화된 프롬프트
- **하이브리드 검색**: 그래프 검색 + 벡터 검색 융합
- **표 구조 유지**: Word 문서의 표 구조를 그대로 유지
- **웹 인터페이스**: Gradio 기반 직관적인 UI

## 기술 스택

- **LLM**: Ollama + Gemma3
- **Embedding**: Ollama + BGE-M3
- **RAG 프레임워크**: lightrag-hku
- **문서 변환**: markitdown
- **Web UI**: Gradio
- **언어**: Python 3.10+

## 프로젝트 구조

```
Light-Graph-RAG/
├── docs/                           # 설계 문서
│   ├── 01_시스템_아키텍처_설계.md
│   ├── 02_모듈별_상세_설계.md
│   ├── 03_데이터_플로우_설계.md
│   └── 04_API_인터페이스_명세.md
│
├── src/                            # 소스 코드
│   ├── preprocessing/              # 전처리 모듈 ✅
│   ├── rag/                        # RAG 엔진 모듈 ✅
│   ├── llm/                        # LLM 인터페이스 모듈 ✅
│   ├── webui/                      # Web UI 모듈 ✅
│   └── utils/                      # 유틸리티 모듈 ✅
│
├── scripts/                        # 실행 스크립트
│   └── run_webui.py               # WebUI 실행 스크립트
│
├── data/                           # 데이터 디렉토리
│   ├── raw/                        # 원본 .docx 파일
│   ├── processed/                  # 전처리된 마크다운
│   └── index/                      # LightRAG 인덱스 저장소
│
└── README.md
```

## 문서

### 설계 문서
상세한 설계는 [docs](./docs/) 폴더를 참조하세요:

1. [시스템 아키텍처 설계](./docs/01_시스템_아키텍처_설계.md) - 전체 시스템 구조 및 컴포넌트
2. [모듈별 상세 설계](./docs/02_모듈별_상세_설계.md) - 각 모듈의 클래스 및 메서드 명세
3. [데이터 플로우 설계](./docs/03_데이터_플로우_설계.md) - 인덱싱 및 검색 파이프라인
4. [API 인터페이스 명세](./docs/04_API_인터페이스_명세.md) - API 사용법 및 데이터 모델
5. **[로컬 환경 구성 가이드](./docs/05_로컬_환경_구성_가이드.md)** ⭐ - Windows/macOS 설치 및 실행 가이드

## 핵심 설계 결정사항

### 1. 한국어 프롬프트 최적화

LightRAG의 기본 프롬프트를 한국어에 최적화된 프롬프트로 교체:
- 엔티티 추출 프롬프트
- 관계 추출 프롬프트
- 답변 생성 프롬프트

### 2. 하이브리드 검색

그래프 검색과 벡터 검색을 병렬로 수행 후 RRF 알고리즘으로 융합:
- 그래프 검색: 엔티티 및 관계 기반
- 벡터 검색: 의미적 유사도 기반
- RRF 융합: 최적의 검색 결과 제공

### 3. 청크 분할 전략

마크다운 헤더를 기준으로 의미 단위 분할:
- 표 구조는 분할하지 않음 (하나의 청크로 유지)
- 최소 100자 ~ 최대 1000자
- 50자 오버랩으로 문맥 연속성 보장

## 빠른 시작

### 1. 사전 준비
- Python 3.10 이상
- Git
- [Ollama](https://ollama.com/download) 설치

### 2. 프로젝트 클론
```bash
git clone https://github.com/laurus-kpa007/Light-Graph-RAG.git
cd Light-Graph-RAG
```

### 3. 가상환경 및 의존성 설치

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Ollama 모델 다운로드
```bash
ollama pull gemma2:latest
ollama pull bge-m3:latest
```

### 5. WebUI 실행
```bash
python scripts/run_webui.py
```

### 6. 브라우저 접속
```
http://localhost:7860
```

**자세한 설치 가이드는 [로컬 환경 구성 가이드](./docs/05_로컬_환경_구성_가이드.md)를 참조하세요.**

## 성능 지표 (예상)

- **전처리**: ~0.5초/파일 (docx → markdown)
- **인덱싱**: ~2.5초/청크 (배치 처리 시 개선)
- **검색**: ~3.5초 (컨텍스트 검색 + 답변 생성)

## 시스템 요구사항

- Python 3.10 이상
- GPU 메모리 8GB 이상 (권장)
- RAM 16GB 이상 (권장)
- SSD 권장

## 주요 기능

### 문서 처리
- ✅ .docx 및 .md 파일 자동 변환
- ✅ 표 구조 유지
- ✅ 헤더 기반 계층적 청크 분할

### 검색 및 답변
- ✅ 하이브리드 검색 (그래프 + 벡터)
- ✅ 한국어 최적화 프롬프트
- ✅ 컨텍스트 기반 정확한 답변

### 사용자 인터페이스
- ✅ Gradio 기반 웹 UI
- ✅ 파일 업로드 및 인덱싱
- ✅ 실시간 검색 및 답변
- ✅ 시스템 상태 모니터링

## 로드맵

- [x] 시스템 아키텍처 설계
- [x] 모듈별 상세 설계
- [x] 데이터 플로우 설계
- [x] API 인터페이스 명세
- [x] 전처리 모듈 구현
- [x] RAG 엔진 모듈 구현
- [x] Web UI 구현
- [ ] 단위 테스트 작성
- [ ] 성능 최적화
- [ ] Docker 배포 지원

## 라이선스

MIT License

## 작성자

Light GraphRAG Team

---

**버전**: 1.0.0
**작성일**: 2026-01-17
