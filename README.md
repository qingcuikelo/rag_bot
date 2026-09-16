# 星驰科技 RAG 智能客服

## 项目概念

面向星驰科技（智能门锁、摄像头、门铃、网关、门窗传感器）的 **RAG 智能客服**，
实现「问得准、答得对、说得清」。


完整设计与实现契约见 [`docs/RAG系统开发文档.md`](docs/RAG系统开发文档.md)。

## 启动方法

### 1. 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 生成质量评测如需 RAGAS，请另建独立环境安装 `requirements-eval.txt`（其依赖与主环境冲突）。

### 2. 配置

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写模型连接配置：

```dotenv
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://xxx
OPENAI_API_KEY=sk-***
LLM_MODEL=<对话模型>
EMBED_PROVIDER=openai
EMBED_MODEL=<向量模型>
RERANK_MODEL=<重排模型>
```

### 3. 构建索引

```powershell
python scripts/build_index.py
```

产出：`storage/chroma`（向量）、`storage/bm25`（稀疏索引）、`storage/xingchi.db`（结构化库）、
`storage/data_quality_report.json`（数据质量报告）、`storage/manifest.json`（索引版本）。

### 4. 启动服务

```powershell
python scripts/serve.py            # 默认 http://127.0.0.1:8000
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"智能门锁整机保修几年？\"}"
```

接口：`POST /v1/chat`、`POST /v1/chat/stream`（SSE 流式）、`POST /v1/chat/resume`（转人工恢复）、
`GET /v1/health`、`GET /v1/metrics`、`POST /v1/feedback`。

### 5. 测试与评测（可选）

```bash
pytest -q                                        # 单元 + 集成测试
ruff check . ; mypy                              # lint / 类型检查
python scripts/run_eval.py --mode hybrid --rerank   # 检索评测门禁
python scripts/run_eval.py --generation --split dev # 生成评测（LLM-as-Judge）
python scripts/load_test.py                      # 并发压测（需先启动服务）
```
