![CI](https://github.com/Polar12e/annual-report-qa-agent/actions/workflows/ci.yml/badge.svg)
# 上市公司年报问答 Agent（V1.2）

V1 的目标：回答需要**翻年报原文**的问题，并给出页码引用；数字仍然来自确定性工具，不由模型编造。

V0 只能算三个比率，问题一步就能回答，演示不出 Agent 的价值。V1 把 452 页年报纳入检索，能处理“营业成本为什么重述”“影响多少”这类必须读附注、跨页取数的问题。

V1.1 把检索从单路 BM25 扩成四路可切换（词法 / 概念重写 / 向量 / RRF 融合），并补了两层口语问法诊断集。结论写在 `docs/rag-notes.md`：**口语到科目的同义改写有效（`expanded` 四层不降、诊断集提升），离线向量通道无效（实测更差，默认关闭）**，向量路径等真实 embedding 模型接上再启用。

V1.2 把 LLM 从协议占位推进到真实调用：新增 `src/llm_config.py`、`scripts/check_llm.py` 和 `evals/agent_eval.py`；DeepSeek `deepseek-flash` 在 6 题小集上 6/6，32 题首轮 28/32（核心 dev+holdout 18/18）。

## 当前状态

- 全文语料：`corpus/pages.jsonl`，452 页、418,330 字符（其中 20 页无文本层）
- 检索：四个可切换通道——`bm25`（词法基线）、`expanded`（OOV 触发的概念重写，默认）、`dense`（向量）、`hybrid`（RRF 融合）
- 向量层：`Embedder` 协议 + 离线 LSA + OpenAI 兼容端点，向量落盘缓存
- 工具层：5 个带 JSON Schema 的工具，调度器永不抛异常
- 策略：脚本策略（两步：`search_report` → `read_page`）+ OpenAI 兼容的 function calling 策略
- LLM 配置：`src/llm_config.py` 统一解析 provider / model / base URL / 凭据 / 超时 / 重试 / DeepSeek thinking
- LLM 实调：DeepSeek `deepseek-flash`；6 题 agent 小集 6/6，32 题首轮 28/32（核心 dev+holdout 18/18）
- 数据边界：`src/policy.py` 预留端点拒绝、片段截断、审计钩子（不实现加密/鉴权）
- 离线评测：dev 14 + holdout 4 + 诊断集 para 8 + para_holdout 6，检索与数值两类指标
- 测试：154 个单元测试
- 学习笔记：`docs/rag-notes.md`（原理 → 代码映射 → 实测 → 走过的弯路）

## 目录结构

```
src/
├─ contracts/financial.py              数据契约（FinancialFact、MetricResult）
├─ data/csv_store.py                   CSV 读取、校验、缓存
├─ capabilities/financial_analysis/    指标计算
├─ retrieval.py                        切块 + BM25 / 概念重写 / 向量 / RRF 融合
├─ embedding.py                        Embedder 协议、LSA、OpenAI 兼容端点、向量缓存
├─ lexicon.py                          领域同义词表（口语 → 会计科目）
├─ policy.py                           数据边界钩子（端点/截断/审计）
├─ tools.py                            Agent 工具层（JSON Schema + 调度器）
├─ agent.py                            最小 Agent 循环（脚本策略）
├─ llm_config.py                       LLM provider / model / 凭据 / 超时 / 重试配置
├─ llm_policy.py                       OpenAI 兼容的 function calling 策略
└─ cli.py                              命令行入口
corpus/pages.jsonl                     452 页年报文本（PDF 抽取）
evals/questions.json                   评测题（dev / holdout / para / para_holdout）
evals/run_eval.py                      离线评测入口（四通道对比）
evals/agent_eval.py                    Agent 端到端评测（默认 dry-run，--live 调 LLM）
docs/rag-notes.md                      RAG 学习笔记
scripts/extract_pages.py               PDF → corpus 复现脚本
scripts/check_llm.py                   LLM 配置/连通性检查（默认不联网）
.env.example                           LLM 环境变量示例（不含真实 key）
.github/workflows/ci.yml               GitHub Actions（测试 + 评测）
tests/                                 154 个单元测试
data/financial_facts.csv               12 条已核验事实
data/reports/                          固定 PDF、SHA256 清单、关键页文本
```

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy openai pypdf

python -m unittest discover -s tests -v
python -m evals.run_eval                              # 四通道对比
python -m evals.run_eval --channels bm25,expanded     # 只比两个
python -m src.cli --company 000333.SZ --year 2024
python -m src.agent "2023年营业成本为什么重述？"
python -m src.agent "美的集团 2024 年毛利率是多少？" --retriever bm25
```

Linux/macOS 把激活命令换成 `source .venv/bin/activate`。当前仓库的 `.venv` 已装好 numpy/openai/pypdf，可直接激活使用。

脚本策略的检索题会走两步（先检索、再读页），输出示例：

```
PDF 第 199 页（含金额的行）：
销售费用  (2,928,031) -
营业成本  2,928,031 -
```

## 评测结果

`python -m evals.run_eval` 的当前输出（hit@1 / hit@3 / MRR）：

| 集合 | 题数 | bm25 | **expanded（默认）** | dense | hybrid |
|---|---|---|---|---|---|
| dev | 14 | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** | 0.36 / 0.57 / 0.45 | 1.00 / 1.00 / 1.00 |
| holdout | 4 | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** | 0.25 / 0.50 / 0.33 | 1.00 / 1.00 / 1.00 |
| para | 8 | 0.38 / 0.50 / 0.47 | **0.50 / 0.62 / 0.59** | 0.38 / 0.50 / 0.44 | 0.50 / 0.62 / 0.59 |
| para_holdout | 6 | 0.17 / 0.17 / 0.28 | **0.33 / 0.33 / 0.41** | 0.00 / 0.00 / 0.00 | 0.33 / 0.33 / 0.41 |

数值校验：11/11 通过（营收、成本、归母净利润、经营现金流、总资产、总负债、三个比率、2023 重述成本）。

Agent 端到端首轮（DeepSeek `deepseek-flash`，`max_steps=6`，`evals/agent_eval.py`）：

| 集合 | 通过 |
|---|---|
| dev | 14/14 |
| holdout | 4/4 |
| para | 5/8 |
| para_holdout | 5/6 |
| 合计 | 28/32 |

失败 4 题都在 `para` / `para_holdout`：`pa06` 引用了 280 页（与 436 页重复的外汇风险附注）、`pa08` 引用了 301 页摘要股东表，两题按原文补全 `expected_pages` 后单独重跑均通过；`pa07` 在重跑中通过；`ph04`（"收购多花了多少钱"）仍会在 `max_steps` 内得不出结论，题目口径需要再明确。核心 `dev` + `holdout` 为 18/18。

`para` / `para_holdout` 是**口语问法诊断集**（"公司仓库里压了多少货"→存货页），专门用来量"字面匹配不到"的损失；它们只报告，不参与 PASS/FAIL。`hybrid` 默认稠密权重为 0，所以数值等于 `expanded`；把权重开到 0.35 会掉到 dev 0.86 / holdout 0.75 —— 这就是不开的原因。

**这组数字仍然偏乐观，不能当通用 benchmark**：

- `dev`/`holdout` 的题目与检索规则出自同一份年报，而且 `holdout` 只有 4 题；
- `expanded` 的提升来自一张手写领域词表，它覆盖不到的语义缺口（"客户欠的钱"→应收账款）仍会失败；
- 检索仍是"页级命中"口径，没有衡量上下文能否支撑一句可信的回答。

它的用途是**回归测试 + 通道对比**：改检索或换 embedding 模型后，四层数字立刻告诉你有没有退化、涨在哪一层。

## 检索怎么工作

四个通道，同一个接口 `search(query, top_k) -> list[SearchHit]`，`src/tools.py` 与 Agent 循环不感知实现：

| 通道 | 做法 | 适合 |
|---|---|---|
| `bm25` | 中文 bigram BM25（k1=1.5，b=0.75）+ 报表页加分 + 比率词扩展 | 科目名、年份、金额 |
| `expanded` | 同上，但查询先按 `src/lexicon.py` 重写：口语/同义 → 会计科目，权重 0.45；**只在 OOV 压力大时触发**（coverage < 0.5） | 说人话的问法，且不伤精确查询 |
| `dense` | 任意 `Embedder` 的向量余弦；离线默认是 LSA（TF-IDF → 192 维随机化 SVD） | 语料没出现过的说法（需要预训练模型才成立） |
| `hybrid` | 加权 RRF 融合（k=60）；稠密权重默认 0 | 多路证据互补时 |

细节：

1. **分块**：每页按行切成约 360 字符的块，块间重叠 80 字符，避免表格行被切断；按页聚合时每页只留最高分块。
2. **分词**：中文按字符二元组（bigram），英文/数字按词切分；数字先去掉千分位逗号；PDF 换行导致的“归属于母公司股东的\n净利润”会先拼接。
3. **结构提示**：查数型问题给“合并及公司利润表 / 资产负债表 / 现金流量表”标题页加 18 分；带“为什么 / 重述 / 重分类 / 影响 / 附注”的问题不加分，让附注页排前面。**去掉这 18 分，dev hit@1 从 1.00 掉到 0.57。**
4. **概念重写**：`仓库` → 存货、`技术` → 研发投入、`抵押` → 质押……匹配用 bigram 而非子串（"在技术上要投多少钱"里没有"技术投入"这个词，但有"技术"）。
5. **向量缓存**：`.cache/embeddings/*.npz`，键 = 语料指纹 + provider + 维度，重建索引不重算。

已知失败模式：**纯语义缺口**（"客户欠的钱"、"账上躺着多少钱"、"一共多少人在上班"）——这需要预训练 embedding，LSA 在这份 418k 字符的语料上学不出语义（详见 `docs/rag-notes.md`）；以及跨页计算、表格列对齐。LLM agent 在 `para` 诊断集上还会因 `max_steps` 或题目口径（如 `ph04` 的"收购多花"）得不出结论；这些是诊断项，不是 `dev`/`holdout` 回归。

## 工具层

| 工具 | 作用 | 关键参数 |
|---|---|---|
| `list_available_data` | 列出已核验的公司、年份、六个指标和事实条数；不代表年报全文 | 无 |
| `query_financials` | 查询原始事实（含页码） | `company_code`、`year`、`metric_codes?` |
| `calculate_metrics` | 计算三个指标（含公式、输入记录 ID 和来源页） | `company_code`、`year` |
| `search_report` | 全文检索，可查未入库科目，返回页码、相关度和上下文 | `keyword`、`max_results?` |
| `read_page` | 读取某一页的完整文本 | `page` |

调用方式：

```python
from src.tools import FinancialToolBox

box = FinancialToolBox()
box.call("search_report", {"keyword": "质保费 重分类", "max_results": 3})
box.call("read_page", {"page": 199})
```

调度器永远返回结构化结果：

```json
{"ok": true,  "tool": "search_report", "result": {"count": 3, "matches": []}}
{"ok": false, "tool": "query_financials", "error": {"code": "INVALID_ARGUMENTS", "message": "year must be an integer"}}
```

错误码：`UNKNOWN_TOOL`、`INVALID_ARGUMENTS`、`DATA_ERROR`、`INTERNAL_ERROR`。

## Agent 循环

```
策略 decide() → 工具调用 → 观察 observation → 策略 decide() → ... → 最终回答
```

- `ScriptedPolicy`：规则版，离线可用；检索题会先 `search_report` 再 `read_page`。
- `OpenAIPolicy`：OpenAI 兼容的 function calling，需凭据；连接参数统一由 `src/llm_config.py` 解析。

推荐先接 DeepSeek `deepseek-flash`（OpenAI 兼容、支持 tool calls、中文和价格都合适）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:FIN_AGENT_PROVIDER = "deepseek"
$env:FIN_AGENT_MODEL = "deepseek-flash"

python -m src.agent "2023年营业成本为什么重述？" --policy llm --allow-external
```

也可以显式传参数：

```powershell
python -m src.agent "2023年营业成本为什么重述？" `
  --policy llm --provider deepseek --model deepseek-flash --allow-external
```

OpenAI 兼容端点：

```powershell
$env:OPENAI_API_KEY = "sk-..."
python -m src.agent "2023年营业成本为什么重述？" --policy llm --model gpt-4o-mini --allow-external
```

LLM 配置项：

| 变量 / 参数 | 作用 | 默认 |
|---|---|---|
| `FIN_AGENT_PROVIDER` / `--provider` | `auto/openai/deepseek/local/custom` | `auto` |
| `FIN_AGENT_MODEL` / `--model` | 模型名 | DeepSeek `deepseek-flash`；OpenAI `gpt-4o-mini` |
| `FIN_AGENT_BASE_URL` / `--base-url` | OpenAI 兼容端点 | 按 provider |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 凭据 | 无 |
| `FIN_AGENT_API_KEY` | 自定义网关的通用凭据 | 无 |
| `FIN_AGENT_TIMEOUT` / `--timeout` | 请求超时秒数 | 30 |
| `FIN_AGENT_MAX_RETRIES` / `--max-retries` | SDK 重试次数 | 2 |
| `FIN_AGENT_THINKING` / `--thinking` | DeepSeek 思考模式 | 关闭 |
| `FIN_AGENT_ALLOW_EXTERNAL` / `--allow-external` | 允许公网端点 | 关闭 |

`provider=local` 示例（Ollama）：

```powershell
$env:FIN_AGENT_PROVIDER = "local"
$env:FIN_AGENT_BASE_URL = "http://127.0.0.1:11434/v1"
$env:FIN_AGENT_MODEL = "qwen3:8b"

python -m src.agent "2023年营业成本为什么重述？" --policy llm
```

先检查配置、不花 token：

```powershell
python scripts/check_llm.py
python scripts/check_llm.py --live --allow-external
```

Agent 端到端小集（dry-run 默认不调用 LLM）：

```powershell
python -m evals.agent_eval --ids q01,q03,q09,q13,h04,pa01
python -m evals.agent_eval --ids q01,q03,q09,q13,h04,pa01 --live --allow-external
python -m evals.agent_eval --all --split holdout --live --allow-external
```

也可以复制 `.env.example` 为 `.env`，填入 key 后自行 `source`/`Import`；应用不会自动加载 `.env`。

依赖：`pip install numpy openai`（`openai` 会带 pydantic；本仓库已在 `.venv` 中装好）。未配置凭据时退出码 2，不抛堆栈；公网端点默认被拒绝，需要显式 `--allow-external`。DeepSeek 默认关闭思考模式，避免每步多一轮推理；需要时用 `--thinking` 打开。

当前状态：DeepSeek `deepseek-flash` 已完成真实调用验证（6 题 6/6；32 题首轮 28/32，核心 dev+holdout 18/18）；OpenAI 及其他 provider 尚未做真实调用。工具协议形状另由假 client + 本地 mock server 覆盖。若 OpenAI SDK 报 `Unknown scheme for proxy URL 'socks://...'`，先执行 `unset ALL_PROXY all_proxy`，保留 `HTTPS_PROXY` 即可。

## 数据边界（预留设计）

`src/policy.py` 只定义边界钩子，**不实现加密、鉴权或审计存储**，用于证明这些边界在开发时已被识别，后续可以不改调用点地补上：

| 能力 | 状态 |
|---|---|
| 默认拒绝公网模型端点（localhost / 内网 IP / `.internal` 等放行） | `OpenAIPolicy` 已实现；embedding 路径未接入 |
| 检索片段截断（top-k、每段字符数） | 已实现 |
| 审计钩子（只传页码、工具名，不传原文） | 已实现（默认空实现） |
| 数据分级枚举 `public/internal/confidential` | 已预留 |
| 加密、鉴权、审计落盘、PII 脱敏 | 未实现 |

```python
from src.policy import DataPolicy

policy = DataPolicy(max_snippets=3, max_snippet_chars=400, audit_hook=print)
policy.ensure_endpoint("https://api.openai.com/v1")   # 抛 PolicyViolation
policy.ensure_endpoint("http://127.0.0.1:11434/v1")   # 通过
```

## 数据口径

采用中国企业会计准则合并列，单位人民币千元。

- `revenue` 取“营业收入”（407,149,600 千元），不是“营业总收入”
- `cost_of_revenue` 以正的费用金额入库，原表以括号列示
- `net_income` 专指归属于母公司股东的净利润（38,537,237 千元）
- 2023 年营业成本经重列为 276,409,404 千元，质保费重分类增加成本 2,928,031 千元（PDF 199 / 印刷 197），该条事实标 `adjusted=true`
- 2024 年：营收同比 9.43785%，毛利率 26.41895%，资产负债率 62.32867%

## 数据校验

数据读取器在每个实例首次查询时读取并校验 CSV，后续查询复用缓存；修改文件后重新启动命令即可读取新数据。

缺失文件、空字段、未知指标、非法年份/页码、非有限数值、非法布尔值和重复记录 ID 都会报错。同一公司、同一年、同一指标允许存在多条事实（例如不同版本），但计算时无法判断该用哪一条，因此该指标返回 `invalid`，其余指标仍正常计算。

## 数据来源

`data/financial_facts.csv` 已录入美的集团 2023、2024 年六项数据共 12 条。来源为用户提供的固定 PDF（`data/reports/midea-2024-combined.pdf`，SHA256 与 `midea-2024-source.json` 一致），官方 URL 尚未独立核验。

`corpus/pages.jsonl` 由该 PDF 一次性抽取生成（pypdf），保留原始页码；20 页无文本层，检索会跳过。可用 `python scripts/extract_pages.py data/reports/midea-2024-combined.pdf` 复现。

## 向量层

```python
from src.embedding import OpenAICompatibleEmbedder
from src.retrieval import build_index

embedder = OpenAICompatibleEmbedder(model="bge-m3", base_url="http://127.0.0.1:11434/v1", dim=1024)
index = build_index("hybrid", embedder=embedder, dense_weight=0.35)
index.search("公司账上躺着多少钱", top_k=3)
```

- `Embedder` 协议只有 `name / dim / encode`，换模型不改下游一行代码。
- 离线默认 `LsaEmbedder`：TF-IDF + 随机化 SVD（numpy，固定种子，可复现），只为让流水线在没有网络时也能跑通和被测；**它不是语义模型**，实测比 BM25 差，所以默认不参与排序。
- `OpenAICompatibleEmbedder` 处理批大小、返回顺序、4xx 不重试/网络错误重试、维度校验；端点校验目前只在 `OpenAIPolicy` 实现，embedding 路径尚未注入 `DataPolicy`，接外部 embedding 前需补齐。
- 向量缓存 `.cache/embeddings/*.npz`（已加入 `.gitignore`）。

换模型后的校准清单（前缀、dim、归一化、四层重测）写在 `docs/rag-notes.md` 第 6 节。

## 后续任务

已完成：

- [x] 数据层：读取、逐行校验、缓存
- [x] 计算层：三个指标 + `ok`/`missing`/`invalid` 状态
- [x] CLI：统一错误捕获、可读的 `limitations`
- [x] 工具层：JSON Schema + 永不抛异常的调度器
- [x] 全文语料：452 页 PDF 文本抽取
- [x] 检索：分块 + BM25 + 报表页提示 + 查询扩展
- [x] 评测：dev 14 + holdout 4，检索与数值指标
- [x] Agent 循环：脚本策略 + OpenAI 兼容策略
- [x] 向量通道：`Embedder` 协议 + LSA + OpenAI 兼容端点 + 向量缓存
- [x] 混合检索：加权 RRF（k=60），稠密权重按实测结论默认关闭
- [x] 概念重写：领域词表 + bigram 匹配 + OOV 触发，`expanded` 成为默认通道
- [x] 诊断评测：para 8 + para_holdout 6，逐通道对比
- [x] LLM 配置层：provider/model/base URL/凭据/超时/重试/thinking 统一解析
- [x] 学习笔记：`docs/rag-notes.md`
- [x] 测试：154 个单元测试

待办：

- [ ] 接真实 embedding 模型（bge-m3 / text-embedding-3-large / Ollama），验证 `para_holdout` 是否变绿
- [ ] 中文分词（词典 + 最大匹配或引入 jieba），取代 char bigram
- [ ] 增加“解释型”holdout（为什么/影响/归因），现在只有查数型
- [ ] 重排：检索 20 条 → cross-encoder/LLM 打 3 条
- [ ] 上下文组装：按行块保留行号，替代表格列易错位的整页截断
- [x] DeepSeek `deepseek-flash` 真实调用（6 题小集 6/6；32 题首轮 28/32）
- [x] `evals/agent_eval.py`：agent 端到端评测脚本（dry-run 默认，`--live` 才调 LLM）
- [ ] OpenAI 及其他 provider 的真实调用验证
- [ ] agent 32 题稳定重跑（修正标签后、含拒答与数值/引用校验）
- [ ] 负向用例自动化（`evals/negative_cases.json`）
- [ ] 补官方披露 URL、下载日期，完成 PDF 页面图像复核
- [ ] 打包（build-system）
- [ ] 扩展公司与年度数据

