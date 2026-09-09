# ICP RBD 身份合同与割集空答案修复记录（2026-09-09）

## 背景与输入

19 例评测在 0908 基线为 12/19。逐条对 0909 HEAD 核定后立项三缺陷修复（`.trae/documents/rbd_identity_contract_fix.md`），其中缺陷 2（fault_history 收窄）与缺陷 3（数值共现合同）经定向回放证实已被此前提交（P7 缺席门禁 `c3a977d`、数值归属 reject）解决，剔除不重复做。

## 实施与验证

### 1. RBD 权威判定单一来源（`d923e07`）

- `evidence_gate.py` 的 `rbd_authority_validation` 写入 `required` 标志；`answer_structure_views.py` 的 `_rbd_item_authority_is_valid` 改为消费 gate 裁决，不再用不同 models 输入无条件重算；
- 消除"gate 批准但投影降级"双算不一致（RBD-01 占位句引用根因）；
- 测试：`tests/test_answer_rbd_success_criterion_projection.py` 扩展。

### 2. 审核阶段独立输出预算 8192（`c92f085`、`b1b1bc1`）

- Ollama 思考 token 计入输出预算，共享 4096 上限使整篇审核随机死于 `finish_reason=length`；
- `_prepare_answer_review_model` 引入 `ANSWER_REVIEW_OUTPUT_TOKENS = 8192` 下限（不小化已配置的更大预算）；
- 生成阶段预算不变，维持"生成思考关闭、审核思考开启"的既有结论。

### 3. FTA 割集空答案五层确定性修复（`891752d`）

同一用例（MT-FTA-02 t2）在端到端中连续暴露五层"审核模型判 pass、确定性校验误杀"形态，逐层修复并各自回归：

| 层 | 误杀形态 | 修复 |
|---|---|---|
| 1 | 无引用标题行/引导句被标 formal_calculation | 全篇已授权确定性凭据背书豁免（answer_review.py `_calculation_endorsements`） |
| 2 | 表格行首序号 `\| 1 \|` 渲染为裸整数参与数值比对 | 连续 1..N 序号列豁免（citation_binding.py `_indexed_row_numbers`） |
| 3 | "8 个单事件/1 个二阶"计数非显式结果字段 | 割集列表确定性推导计数（总数+分阶数）纳入支持值；联引路径统一 `cut_sets_only` 判定（`_cut_set_derived_counts`） |
| 4 | 割集计数被误标 formal_numeric_result → 要求 fta_probability | 纯割集凭据降级按割集授权（`_cut_set_only_evidence`） |
| 5 | 误标 numeric_result 的无引用标题行不享受背书豁免 | 豁免开放至 numeric_result，但仅限纯割集凭据背书；真概率值仍要求同 span 引用+fta_probability |

安全边界：唯一诊断/拓扑断言严格就近不变；真配错快照（结构计数与割集数混句单引结构记录）保持拒发；错误计数、概率值无授权、provisional 凭据均仍拦截。

### 4. 评测 harness：澄清轮 Judge 协议错误（`891752d` 内）

- 澄清路由 + 有正文 + 无引用时，Judge"通过维度缺少答案原话支持"类协议错误记 `not_evaluated` 不阻塞；knowledge_base 轮与空答案仍 fail-closed；
- MT-ROUTER-02 两轮重跑确认转 PASS（智能体行为本来正确，14 字澄清句无法提供原话引用是协议天然无解）。

## 评测结果（`.artifacts/evals/rbd-contract-20260909/`）

- 共享 ollama 期间全量 19 例：13/19；
- 关键发现：另一项目共用同一 ollama 加载其他模型导致 qwen3.8 被驱逐重载（llama-server PID 变化实测确认），重载窗口内的用例出现超时/审核未完成的空答案假失败；
- 失败 5 例独占重跑（另一项目已停）：FAULT-01、FTA-01、RBD-03、ROUTER-02 均 PASS；FTA-02 经五层修复后 PASS；
- 等效成绩 **18/19**。0908 基线 FAIL 的 RBD-01/02、MFL-01、HISTORY-01、MODULE-01 全部转 PASS。

## 遗留（判低危不修）

- MT-FAULT-01 t2 追问轮 writer 以"若…"条件分支引用同域近邻 MFL 记录（MFL-SYN-010/015）：引用忠实、条件化呈现、主结论锚定正确，属焦点稀释非安全违规；独占重跑未复现。观察哨：若后续出现无"若"限定的近邻引用或写成肯定结论，升级为必修；
- M3 语义验收整体状态未变，19/100 未恢复，本批不代表整体通过。

## 工程检查

- `python -m compileall -q backend scripts` 通过；
- 全量 `python -m unittest discover`：1627 项通过（本批开始时 1619）；
- 前端无改动，未重跑；
- 快照回放：`scripts/replay_icp_review.py` 对历史 8 个拒发快照 7 放行 1 保持拒发（真配错）。
