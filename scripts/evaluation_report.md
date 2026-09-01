# KnowBot RAG — Evaluation Report

Generated: 2026-09-01T12:19:52+00:00
Model: `openai/gpt-oss-120b` (router: `openai/gpt-oss-20b`)
Query set: 12 test queries + 3 out-of-scope queries

## Summary vs. QA targets

| Metric | Result | Target | Verdict |
|---|---|---|---|
| Retrieval Top-3 Recall | 88.9% | ≥70% | PASS |
| Hallucination Rate | 33.3% | <15% | FAIL |
| Citation Correctness | 100.0% | 100% | PASS |
| Avg. Answer Quality (LLM-assisted) | 4.4 / 5 | informational | — |

(9 of 12 queries were retrieval-scored 'codebase' responses; 3 were 'general' responses with no retrieval to score.)

**Out-of-scope hallucination guard:** 1/3 correctly declined or were judged ungrounded rather than fabricating an answer.

## Per-query detail

| # | Query | Mode | Retrieval Hit | Citation OK | Grounded | Quality | Answer (preview) |
|---|---|---|---|---|---|---|---|
| 1 | what target table we are writing ezcater_catering and what a | codebase | ❌ | ✅ | ❌ | 1 | **Target tables and merge keys used by the *ezcater catering* pipelines**  | Pip… |
| 2 | What does read_data do in util? | error | - | - | - | - | ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `ope… |
| 3 | What does transform do in ezcater_catering_order_line? | codebase | ✅ | ✅ | ❌ | 5 | **Answer – What the `transform` function does in `ezcater_catering_order_line.py… |
| 4 | what type of transformations we are doing in givex | codebase | ✅ | ✅ | ✅ | 5 | **Giv X pipelines apply a mix of schema‑level and row‑level transformations.**  … |
| 5 | what is the target table for food safety audit | codebase | ✅ | ✅ | ✅ | 5 | The **food‑safety‑audit** pipeline writes its results to the curated layer table… |
| 6 | What does the franchise_fees pipeline configure? | codebase | ✅ | ✅ | ✅ | 5 | **franchise_fees pipeline – what it configures**  The `franchise_fees` pipeline … |
| 7 | What does the doordash_delivery_order_operation pipeline con | error | - | - | - | - | ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `ope… |
| 8 | what is mentioned in the DP Jobs and Schedule page | codebase | ✅ | ✅ | ✅ | 4 | **DP Jobs and Schedule page (Confluence – https://gotofoods.atlassian.net/wiki/s… |
| 9 | What does the givex_ingestion_poc_airflow Confluence page sa | error | - | - | - | - | ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `ope… |
| 10 | What does the databricks_job_cluster_policy_enforcement_engi | codebase | ✅ | ✅ | ✅ | 5 | **Section 8 – Exception Handling (as written on the Confluence page “Databricks … |
| 11 | What does the brand_survey_response Confluence page say abou | codebase | ✅ | ✅ | ✅ | 5 | **Legacy procedure dbo.spMergeSurveyDetails – part 5/6 (as described on the *bra… |
| 12 | What is documented on the sprint_18_demo_9_22_2022 Confluenc | codebase | ✅ | ✅ | ❌ | 5 | The **Sprint 18 Demo – 9/22/2022** Confluence page contains the meeting agenda f… |

## Notes on flagged rows

- **#1** (`what target table we are writing ezcater_catering and what a`): not grounded — The answer's claims about ezcater pipelines, target tables, and merge keys are not present in the provided source code.
- **#2** (`What does read_data do in util?`): pipeline error — ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in organization `org_01kx8dwws3emhvd4hd8k7zvdka` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 21113, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
- **#3** (`What does transform do in ezcater_catering_order_line?`): not grounded — The answer includes many claims (e.g., building hash keys, adding timestamps, final column selection, error handling details) that are not present in the provided source code.
- **#7** (`What does the doordash_delivery_order_operation pipeline con`): pipeline error — ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in organization `org_01kx8dwws3emhvd4hd8k7zvdka` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 8234, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
- **#9** (`What does the givex_ingestion_poc_airflow Confluence page sa`): pipeline error — ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in organization `org_01kx8dwws3emhvd4hd8k7zvdka` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 26050, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
- **#12** (`What is documented on the sprint_18_demo_9_22_2022 Confluenc`): not grounded — The answer adds unsupported details (expanding AAS to Azure Analysis Services and claiming the agenda items are the only content on the page) not confirmed by the source.

## Out-of-scope queries (hallucination guard)

| Query | Mode | Guard OK | Answer (preview) |
|---|---|---|---|
| What does the unicorn_sparkle_processor.py file do? | error | ❌ | ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in… |
| Summarize the blockchain_integration Confluence page. | codebase | ✅ | I’m sorry, but none of the Confluence pages provided in the material contain a “blockchain_integrati… |
| What primary key does the mars_colony_inventory table use? | error | ❌ | ERROR: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in… |
