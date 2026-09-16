# 星驰科技 RAG 智能客服（Xingchi RAG Customer Service）

以 `data/` 目录资料为唯一知识底座，基于 **LangChain + LangGraph** 构建的检索增强生成客服系统，
目标为「检索召回率高、回答准确率高、最终回复清晰明确」。

完整设计与实现契约见 [`docs/RAG系统开发文档.md`](docs/RAG系统开发文档.md)（v3.0）。

## 目录结构

```
src/xingchi_rag/     # 业务代码
  config.py          # 配置（pydantic-settings）
  logging.py         # 日志 + PII 脱敏
  providers/         # LLM / Embeddings 工厂
  ingestion/         # 加载、治理、切分、事实卡、索引流水线
  sql/               # SQLite DDL / 入库 / SQL Agent
  retrieval/         # 混合检索 / 重排
  graph/             # LangGraph 状态图
  generation/        # 生成链与 Prompt
  api/               # FastAPI 服务
configs/             # 治理规则、别名表、版本化 Prompt
eval/                # 金标集与评测指标
tests/               # 单元 / 集成测试
scripts/             # build_index / run_eval / serve
```

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # 然后自行填写密钥，切勿提交
```

## 运行命令

```bash
python scripts/build_index.py   # 构建/重建索引
python scripts/run_eval.py      # 检索评测门禁（--rerank 可选）
python scripts/run_eval.py --generation --split dev   # 生成评测（LLM-as-Judge）
python scripts/serve.py         # 启动 API
python scripts/load_test.py     # 并发压测（需先启动 serve）
pytest -q                       # 单元 + 集成测试
ruff check .                    # lint
mypy                            # 类型检查
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat` | 问答（一次性） |
| POST | `/v1/chat/stream` | 问答（SSE：route → token → citations → done） |
| POST | `/v1/chat/resume` | 人工介入后恢复被中断会话（转人工） |
| GET | `/v1/health` | 健康检查（索引版本 / Provider） |
| GET | `/v1/metrics` | 运行指标（延迟均值 / 计数） |
| POST | `/v1/feedback` | 用户反馈 |

## 评测依赖说明

生成质量评测默认使用内置 LLM-as-Judge，无需额外依赖。如需 RAGAS，请使用**独立环境**
安装 `requirements-eval.txt`（其 `instructor` 依赖锁定 `openai<2`，与主环境冲突）。

## 隐私与安全

- `data/`、`.env`、`storage/` 已在 `.gitignore` 中排除，**禁止提交**；
- 客户信息（PII）不入向量库，SQLite 中手机号以加盐 SHA-256 存储；
- 日志与 API 输出统一经 PII 脱敏；Prompt 注入请求被识别并拒答。
