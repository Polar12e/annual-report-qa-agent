# 年报问答 Agent 交接文档

> 交接日期：2026-11-18  
> 仓库：`https://github.com/Polar12e/annual-report-qa-agent`  
> 分支：`main`  
> 交接基线 commit：`c89b37c`（`feat: add line-aware hit windows to search_report`）  
> 交接基线状态：158 个测试通过；离线检索评测 PASS；DeepSeek `deepseek-flash` 已完成真实调用验证。  
> 说明：`HANDOVER.md` 是本次交接新增文件，尚未提交。

---

## 1. 项目是什么

一个面向上市公司年报的 **RAG + 工具调用 + 离线评测** 项目：

- 语料：美的集团 2024 年 452 页年报（418,330 字符）；
- 确定性数据：`data/financial_facts.csv` 中的 12 条已核验财务事实；
- 检索：BM25 / 概念重写 / LSA 向量 / RRF 混合，四路可切换；
- 工具：5 个 JSON Schema 工具，Agent 只能通过工具取数；
- LLM：OpenAI 兼容端点，当前默认 DeepSeek `deepseek-flash`；
- 评测：
  - `evals/run_eval.py`：检索与数值的离线评测；
  - `evals/agent_eval.py`：走完整 `agent.run` 的端到端评测。

详细设计、实验和踩坑记录在：

- `README.md`：使用说明、数据口径、评测结果；
- `docs/rag-notes.md`：RAG 原理 → 代码映射 → 实测 → 失败模式。

---

## 2. 当前验收状态

### 测试

```bash
python -m unittest discover -s tests
```

结果：

```text
Ran 158 tests
OK
```

### 离线检索/数值评测

```bash
python -m evals.run_eval
```

结果：

```text
result: PASS
numeric 11/11
```

### Agent 端到端评测

6 题小集：

```bash
python -m evals.agent_eval \
  --ids q01,q03,q09,q13,h04,pa01 \
  --live --allow-external
```

结果：`6/6 PASS`。

32 题首轮（标签修正前）：

| split | 通过 |
|---|---|
| dev | 14/14 |
| holdout | 4/4 |
| para | 5/8 |
| para_holdout | 5/6 |
| 合计 | 28/32 |

首轮 4 个失败都在 `para` / `para_holdout`：

- `pa06`：外汇风险附注 280/436 是重复披露；已补 `expected_pages`，单独重跑 PASS；
- `pa08`：股东质押问题引用 301 页摘要表；已补 143/301，单独重跑 PASS；
- `pa07`：答案正确但一次没写页码，重跑引用 220 页后 PASS；
- `ph04`：“收购别人多花了多少钱”口径不明确，仍可能 `max_steps`；**未解决**。

修正标签后尚未做完整 32 题重跑；交接后优先做一次稳定重跑。

---

## 3. 快速开始

### Linux/macOS

```bash
cd annual-report-qa-agent
python3 -m venv .venv
source .venv/bin/activate
pip install numpy openai pypdf

python -m unittest discover -s tests -v
python -m evals.run_eval
python -m src.cli --company 000333.SZ --year 2024
python -m src.agent "2023年营业成本为什么重述？"
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy openai pypdf
```

### `.env` 加载

项目**不会自动加载 `.env`**。复制示例并填入自己的凭据：

```bash
cp .env.example .env
chmod 600 .env
# 编辑 .env：填写 DEEPSEEK_API_KEY，保留 FIN_AGENT_PROVIDER=deepseek
set -a
source .env
set +a
```

`.env` 已被 `.gitignore` 忽略，不要提交。`.env.example` 中的 key 必须保持为空。

---

## 4. 代码结构

```text
src/
├─ contracts/financial.py              领域契约：FinancialFact / MetricResult
├─ data/csv_store.py                   CSV 读取、校验、缓存
├─ capabilities/financial_analysis/    三个指标计算
├─ retrieval.py                        切块、BM25、页级聚合、命中窗口
├─ embedding.py                        Embedder 协议、LSA、OpenAI 兼容端点、缓存
├─ lexicon.py                          口语 → 会计科目的领域词表
├─ policy.py                           数据边界钩子（端点/截断/审计）
├─ tools.py                            Agent 工具层（JSON Schema + 调度器）
├─ agent.py                            最小 Agent 循环
├─ llm_config.py                       LLM provider/model/凭据/超时/重试配置
├─ llm_policy.py                       OpenAI 兼容 function calling 策略
└─ cli.py                              命令行入口

evals/
├─ run_eval.py                         检索与数值评测
├─ agent_eval.py                       Agent 端到端评测（默认 dry-run）
└─ questions.json                      32 道评测题

scripts/
├─ extract_pages.py                    PDF → corpus/pages.jsonl
└─ check_llm.py                        LLM 配置/连通性检查（默认不联网）
```

### 依赖方向

```text
entrypoints (cli / agent / evals)
        ↓
agent / tools
        ↓
capabilities
        ↓
contracts
```

检索、CSV、LLM、embedding 都是适配器；领域契约不依赖它们。

---

## 5. 关键实现

### 5.1 检索

四个通道：

| 通道 | 说明 |
|---|---|
| `bm25` | 中文 bigram BM25，报表页加分，比率词扩展 |
| `expanded` | BM25 + OOV 触发的概念重写（默认） |
| `dense` | Embedder 余弦；离线 LSA 实测差，默认权重 0 |
| `hybrid` | 加权 RRF，默认稠密权重 0 |

当前默认：`expanded`，`k1=1.5`、`b=0.75`、`chunk=360/80`。

### 5.2 页级上下文 + 命中窗口（当前版本）

- `Chunk` 带 `line_start` / `line_end`；
- `SearchHit` 带行范围和 `matched_terms`；
- `search_report` 返回：
  - `pdf_page`
  - `page_heading`
  - `line_start` / `line_end`
  - `truncated`
  - `matched_terms`
  - `text`：页标题附近 + chunk 行范围 ±2 行，可跨出最高分 chunk；
- `read_page` 仍然返回整页；
- `DataPolicy.max_snippet_chars` 默认 900。

q01 的 199 页 top1 现在能同时返回：

```text
销售费用  (2,928,031) -
营业成本  2,928,031 -
```

### 5.3 确定性财务工具

- `query_financials`：返回事实、页码、record_id；
- `calculate_metrics`：返回公式、状态、输入 record_id、`input_pages`；
- 金额用 `Decimal`，不由 LLM 计算；
- `calculate_metrics` 已带来源页，指标答案可以直接引用。

### 5.4 LLM 接入

- 配置：`src/llm_config.py`；
- 策略：`src/llm_policy.py`；
- 默认 provider `deepseek`，模型 `deepseek-flash`，思考模式默认关闭；
- 支持 OpenAI 兼容端点、`FIN_AGENT_PROVIDER=local`（Ollama）；
- `OpenAIPolicy` 做公网端点拒绝；embedding 路径尚未接入 `DataPolicy`；
- 运行：

```bash
export DEEPSEEK_API_KEY='sk-...'
export FIN_AGENT_PROVIDER='deepseek'
export FIN_AGENT_MODEL='deepseek-flash'
python -m src.agent "2023年营业成本为什么重述？" --policy llm --allow-external
```

---

## 6. 数据与语料

### 财务事实

- `data/financial_facts.csv`：12 条，2023/2024 各 6 项；
- `revenue` 取“营业收入”，不是“营业总收入”；
- `cost_of_revenue` 以正数入库，原表括号列示；
- `net_income` 指归母净利润；
- 2023 营业成本重列为 `276,409,404` 千元，质保费重分类 `2,928,031` 千元，标 `adjusted=true`；
- 来源 PDF SHA256：`99950e0093665865783f30292796e013e00d995311bd5c6ee45a1c40fc4268e8`；
- 官方披露 URL 尚未独立核验。

### 语料

- `corpus/pages.jsonl`：452 页，418,330 字符；
- 20 页无文本层（PDF 305–322、451–452 等）；
- 约 127 对近重复页（年报正文与独立财务报表/审计报告重叠）；
- 可用 `scripts/extract_pages.py` 从固定 PDF 复现；
- 20 页无文本层目前跳过，需 OCR 才能覆盖。

---

## 7. 评测

### 7.1 检索评测

```bash
python -m evals.run_eval
```

指标：hit@1 / hit@3 / MRR / 数值校验。当前结果见 README。

### 7.2 Agent 评测

```bash
# dry-run，不调用 LLM
python -m evals.agent_eval --ids q01,q03,q09,q13,h04,pa01

# 真实调用
python -m evals.agent_eval --ids q01,q03,q09,q13,h04,pa01 --live --allow-external

# holdout
python -m evals.agent_eval --all --split holdout --live --allow-external

# 全量（会花 token）
python -m evals.agent_eval --all --live --allow-external --max-steps 10 --json-out ./agent_eval_32.json
```

评测检查：

- `stopped == final`；
- 工具 observation 无错误；
- 答案包含期望数值（有 `check` 时）；
- 答案引用的页码在 `expected_pages` 中。

### 7.3 评测题分层

- `dev` 14：调参/回归；
- `holdout` 4：调参后未看过；
- `para` 8：口语诊断；
- `para_holdout` 6：概念表定稿后新写。

不要用 `para_holdout` 反复调参；它是对外报告的留出集。

---

## 8. 已知问题 / 风险

### 高优先级

1. **`ph04` 题目歧义**  
   “收购别人多花了多少钱”可能指商誉/对价/溢价；模型可能 `max_steps`。  
   建议：改题目口径，或补 `expected_pages`（157/229/153）并明确“指商誉”。

2. **Agent 运行有采样波动**  
   `OpenAIPolicy` 没有显式 `temperature`；同一题多次运行可能走不同工具路径。  
   建议：增加 `FIN_AGENT_TEMPERATURE=0`，在评测中固定。

3. **页级评测不等于可回答**  
   `run_eval` 只看 expected page；`agent_eval` 才开始看引用和最终答案。  
   下一步可加 `answer_span_hit`。

4. **重复页**  
   280/436、301/143 等是重复/补充披露；标签已补，但检索仍可能返回重复页。  
   需要 canonical page 映射。

### 中优先级

5. **20 页无文本层**  
   PDF 305–322、451–452 不可检索；需要 OCR。

6. **embedding 路径未接入 policy**  
   `OpenAICompatibleEmbedder` 没有调用 `DataPolicy.ensure_endpoint`；接外部 embedding 前必须补。

7. **embedding cache key 不完整**  
   LSA 的 `min_df/max_df_ratio/seed` 和 API `base_url` 不在 key 中；调参/换端点可能复用旧向量。

8. **多报告支持不足**  
   `SearchHit` 只有 `page`，没有 `report_id`；接第二家公司前要改主键和工具参数。

9. **检索语义缺口**  
   `pa02/pa04/pa08/ph02/ph06` 等纯语义问题仍依赖词法；LSA 不是语义模型，dense 默认关闭。

### 低优先级

10. 没有 `build-system`，`pip install .` 不可用；
11. CI 只测 Python 3.11/3.12，不跑真实 LLM；
12. 没有 license、依赖锁文件；
13. `scripts/check_llm.py` 和 `agent_eval` 不在 CI 中。

---

## 9. 设计决策记录

| 决策 | 原因 |
|---|---|
| 财务数字用 `Decimal` + 工具 | 不让 LLM 算钱 |
| `bm25` / `expanded` 作为默认 | 在这份小语料上实测强于 LSA |
| LSA `dense` 默认权重 0 | 离线向量实测更差，不能为了“有向量”而开启 |
| RRF 只看名次 | BM25 分数和余弦不可直接比较 |
| `search_report` 返回命中窗口 | 单 chunk 可能缺少答案行 |
| 不做 LLM 查询改写 | 当前收益主要来自标签/引用/题目歧义；额外改写增加成本 |
| 保留 `read_page` | 窗口用于回答，整页用于深核 |
| `max_snippet_chars=900` | 需要容纳页标题 + 行窗口 |

---

## 10. 交接后优先事项

### P0

- [ ] 确认 `main` / `origin/main` 和当前 commit 一致；
- [ ] 跑 `python -m unittest discover -s tests`（158）；
- [ ] 跑 `python -m evals.run_eval`（PASS）；
- [ ] 配 `.env`，跑 `python scripts/check_llm.py --live --allow-external`；
- [ ] 稳定重跑 32 题，记录结果；
- [ ] 处理 `ph04` 题目口径。

### P1

- [ ] 给 LLM 加 `temperature` 配置，评测用 0；
- [ ] 把 agent eval 的 gate 拆成 answer/citation/stopped；
- [ ] 补负向用例：2022 年、未知公司、检索不到；
- [x] `MetricResult.input_pages` 已有测试（`test_metric_results_include_source_pages`）；可补文档；
- [ ] 补 `answer_span_hit`；
- [ ] 处理 embedding policy 和 cache key。

### P2

- [ ] 真实 embedding A/B（bge-m3）；
- [ ] 表格感知窗口 / 多窗口；
- [ ] OCR 无文本层页面；
- [ ] 多报告 `(report_id, page)`；
- [ ] 打包、许可证、依赖锁、CI 扩展。

---

## 11. 交接检查清单

```text
[ ] git status 干净
[ ] HEAD == origin/main
[ ] .env 存在且被忽略，未提交
[ ] .env.example 中 key 为空
[ ] 158 tests OK
[ ] run_eval PASS
[ ] check_llm 配置 OK
[ ] check_llm --live OK
[ ] 6 题 agent 小集可跑
[ ] README / docs/rag-notes 已读
[ ] 已知问题已知悉
```

---

## 12. 常见故障

### `python: command not found`

使用 `.venv/bin/python`，或：

```bash
source .venv/bin/activate
```

### `source .env: No such file`

`.env` 不会自动创建：

```bash
cp .env.example .env
chmod 600 .env
```

### `Unknown scheme for proxy URL 'socks://...'`

当前环境代理变量可能是：

```text
ALL_PROXY=socks://127.0.0.1:7897/
```

OpenAI SDK/httpx 不接受该 scheme：

```bash
unset ALL_PROXY all_proxy
```

保留 `HTTPS_PROXY` 即可。

### `401 Authentication Fails`

检查：

- key 是否以 `sk-` 开头、是否完整；
- 是否复制成了 key ID / UUID；
- `base_url` 是否 `https://api.deepseek.com`；
- 账户余额/状态。

### `PolicyViolation: refusing to call a public model endpoint`

公网端点需要：

```bash
--allow-external
```

或设置 `FIN_AGENT_ALLOW_EXTERNAL=true`。

### `No module named evals.agent_eval`

确认当前 commit 包含 `evals/agent_eval.py`，并在项目根目录运行：

```bash
python -m evals.agent_eval --help
```

---

## 13. 扩展指引

### 新增指标

1. `src/contracts/financial.py` 的 `MetricCode` 加 code；
2. `METRIC_CODES` 自动更新；
3. `service.py` 加公式；
4. 加测试。

### 新增检索通道

1. 实现 `ChunkedIndex.search`；
2. 加入 `CHANNELS` 和 `build_index`；
3. 在 `run_eval` 加通道；
4. 不改 `tools.py` / `agent.py` 的调用点。

### 新增工具

1. `TOOL_SPECS` 加 JSON Schema；
2. `FinancialToolBox` 加 `_tool_<name>`；
3. 加错误路径测试；
4. 确认工具粒度和描述不会和现有工具重叠。

### 新增 provider

1. `LLMSettings.from_env` 加 provider；
2. `OpenAIPolicy` 保持 OpenAI 兼容；
3. `scripts/check_llm.py` 验证配置；
4. 不要绕过 `DataPolicy.ensure_endpoint`。

---

## 14. 访问与凭据

- GitHub：`https://github.com/Polar12e/annual-report-qa-agent`
- 分支：`main`
- 模型：DeepSeek `deepseek-flash`；凭据只放在 `.env` 或环境变量，不提交。
- 仓库中没有生产密钥；`.env` 已被忽略。
- 交接后如更换 key，请在 DeepSeek 控制台轮换，并更新本地 `.env`。

---

## 15. 一句话交接

> 这是一个“确定性财务工具 + 可溯源 RAG + 离线评测”的年报问答项目；当前正常路径已跑通，核心风险在 `ph04` 题目歧义、LLM 采样波动、嵌入/多报告尚未接入，以及页级评测与 answer-span 之间的差距。
