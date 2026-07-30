# Codebase RAG 어시스턴트 — 쿼리 서빙 컨테이너.
# 지식원(문서·코드·git)은 별도 repo(호스트)에 있으므로 인덱싱은 호스트에서 수행하고,
# 컨테이너는 그 결과인 chroma_db 를 마운트해 검색·응답만 담당한다.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# chromadb·sentence-transformers 의 네이티브 확장 빌드에 필요한 도구
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# torch 는 CPU 전용 휠로 먼저 설치(기본 CUDA 휠은 수 GB) → 이미지 용량 절감
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app/ ./app/
COPY web/ ./web/

EXPOSE 8123

# 임베딩 모델은 첫 기동 시 HuggingFace 에서 받아 HF_HOME 에 캐시(compose 볼륨으로 유지).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8123"]
