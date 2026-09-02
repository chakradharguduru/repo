# KnowBot RAG — Evaluation Report

Generated: 2026-09-02T14:07:04+00:00
Model: `openai/gpt-oss-120b` (router: `openai/gpt-oss-20b`)
Query set: 14 test queries + 3 out-of-scope queries

## Summary vs. QA targets

| Metric | Result | Target | Verdict |
|---|---|---|---|
| Retrieval Top-3 Recall | 100.0% | ≥70% | PASS |
| Hallucination Rate | 21.4% | <15% | FAIL |
| Citation Correctness | 85.7% | 100% | FAIL |
| Avg. Answer Quality (LLM-assisted) | 5.0 / 5 | informational | — |

(14 of 14 queries were retrieval-scored 'codebase' responses; 0 were 'general' responses with no retrieval to score.)

**Out-of-scope hallucination guard:** 3/3 correctly declined or were judged ungrounded rather than fabricating an answer.

## Results by query category

Direct evidence for RFP query-category coverage — each row is scored against the same 4 targets as the summary table above, computed only from that category's queries. Pre-existing code/config-lookup queries are tagged `reference` (not an RFP category) and reported separately below, not folded into these 4 rows.

| Category | Queries | Retrieval Accuracy | Hallucination Rate | Citation Correctness | Avg Quality Score |
|---|---|---|---|---|---|
| How-to | 3 | 100.0% | 33.3% | 66.7% | 5.0 / 5 |
| Troubleshooting | 3 | 100.0% | 33.3% | 100.0% | 5.0 / 5 |
| Architecture | 5 | 100.0% | 20.0% | 80.0% | 5.0 / 5 |
| Policy | 3 | 100.0% | 0.0% | 100.0% | 5.0 / 5 |

## Per-query detail

| # | Query | Category | Mode | Retrieval Hit | Citation OK | Grounded | Quality | Answer (preview) |
|---|---|---|---|---|---|---|---|---|
| 1 | explain Azure Functions Local Environment Configuration acco | how_to | codebase | ✅ | ❌ | ❌ | 5 | **Azure Functions – Local Environment Configuration**   *(Source: Confluence pag… |
| 2 | How to handle Temporarily Disabled Pipelines in ADF Trigger  | how_to | codebase | ✅ | ✅ | ✅ | 5 | **How to handle temporarily‑disabled pipelines in Azure Data Factory (ADF) trigg… |
| 3 | How do I get Airflow access to connect to the UI and VM in d | how_to | codebase | ✅ | ✅ | ✅ | 5 | **Getting Airflow access (UI + SSH) for DEV and PROD**  All of the steps are des… |
| 4 | what are Airflow ADF Databricks Issues mentioned in confluen | troubleshooting | codebase | ✅ | ✅ | ✅ | 5 | **Airflow – ADF – Databricks Integration: Issue & Resolution Log**   *(as docume… |
| 5 | What issues are logged in the issue tracker for our Databric | troubleshooting | codebase | ✅ | ✅ | ❌ | 5 | **Databricks‑pipeline issues recorded in the Issue Tracker**  | Ticket | Date (2… |
| 6 | What performance issues have been identified with our ADF pi | troubleshooting | codebase | ✅ | ✅ | ✅ | 5 | **Identified performance‑related issues with the current Azure Data Factory (ADF… |
| 7 | what is present in the list of data sources confluence page? | architecture | codebase | ✅ | ❌ | ❌ | 5 | **What is on the “List of Data Sources” Confluence page**  The page ( https://go… |
| 8 | What does the Customer and Loyalty domain documentation say  | architecture | codebase | ✅ | ✅ | ✅ | 5 | The documentation explains that the **Customer and Loyalty domain’s purpose is t… |
| 9 | what sources are we using in speed_of_service_delphi | architecture | codebase | ✅ | ✅ | ✅ | 5 | **Sources used by the `speed_of_service_delphi` pipeline**  | Source name (as de… |
| 10 | In economic_series_observation, what is the source type and  | architecture | codebase | ✅ | ✅ | ✅ | 5 | In the **`economic_series_observation`** configuration the two fields are define… |
| 11 | what is the target table and what are the mergekeys in the f | architecture | codebase | ✅ | ✅ | ✅ | 5 | **Target table**  - In the *form_submission* pipeline configuration the **target… |
| 12 | What does the databricks_job_cluster_policy_enforcement_engi | policy | codebase | ✅ | ✅ | ✅ | 5 | **Section 8 – Exception Handling** (from the *Databricks Job Cluster Policy Enfo… |
| 13 | what standards we need to follow for ADF Pipeline Structure  | policy | codebase | ✅ | ✅ | ✅ | 5 | **ADF Pipeline Structure & Naming Standards**   *(Source: “ADF Pipeline Structur… |
| 14 | What are the required design standards for naming and struct | policy | codebase | ✅ | ✅ | ✅ | 5 | **Required design standards for naming and structuring Delta tables**   *(Source… |

## Notes on flagged rows

- **#1** (`explain Azure Functions Local Environment Configuration acco`): not grounded — The answer includes additional explanatory claims (e.g., about virtual‑environment isolation, mirroring Azure Application Settings, purpose of the ENV key, and runtime storage requirements) that are not explicitly stated in the source material.
- **#1** (`explain Azure Functions Local Environment Configuration acco`): citation mismatch — a cited file was not among the retrieved chunks
- **#5** (`What issues are logged in the issue tracker for our Databric`): not grounded — The answer claims the table name is part of the issue, but the source lists it under the resolution, not the issue.
- **#7** (`what is present in the list of data sources confluence page?`): not grounded — The answer claims MSI – DataFeed uses sFTP, but the source material does not show sFTP as the source type for that entry
- **#7** (`what is present in the list of data sources confluence page?`): citation mismatch — a cited file was not among the retrieved chunks

## Out-of-scope queries (hallucination guard)

| Query | Mode | Guard OK | Answer (preview) |
|---|---|---|---|
| is there any information on auto loader or spark structured streaming? | codebase | ✅ | I’m sorry, but none of the material you provided contains any information about **Auto Loader** or *… |
| Summarize the blockchain_integration Confluence page. | codebase | ✅ | I’m sorry, but none of the Confluence pages provided in the material include a “blockchain_integrati… |
| is there any information on powerBi import to AI/BI Dashboards | codebase | ✅ | Based on the material you provided, the only Power BI‑related content is the **“Power BI”** Confluen… |
