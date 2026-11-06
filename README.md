# Instamart Customer Intelligence and Support Copilot

## Overview

This repository is an internal copilot for a synthetic Instamart business. It cleans messy order exports into a trusted SQLite database, segments customers and scores who is likely to lapse, triages support tickets with privacy and output guardrails, and answers policy questions and refund requests. All language-model calls go through the offline file `mock_llm.py`. No API key and no network call are required. Amounts are in Indian Rupees (INR). Python 3.11 or newer is required; this project was built with Python 3.13.

The two supplied files `generate_data.py` and `mock_llm.py` are used unmodified. Parts run in order because each one reads what the previous part wrote.

## Reproduce

Run these commands from the repository root, in order:

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python scripts/run_part_a.py`
6. `python scripts/run_part_b.py`
7. `python scripts/run_part_c.py`
8. `python scripts/run_part_d.py`

Step 4 must print `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Step 5 must print `ALL CHECKPOINTS PASSED`.

## Part A

### Audit

`tickets.csv` is read with `keep_default_na=False`, so an empty ticket stays an empty string. Blank strings are counted as missing in the tables below.

| Table | Rows | Columns |
| --- | ---: | ---: |
| customers | 1,200 | 5 |
| products | 40 | 4 |
| orders | 13,084 | 9 |
| tickets | 164 | 5 |

Customers (`customer_id` str, `city` str, `signup_date` str, `age` float64, `gender` str):

| Column | Missing | Missing % |
| --- | ---: | ---: |
| customer_id | 0 | 0.00 |
| city | 0 | 0.00 |
| signup_date | 0 | 0.00 |
| age | 61 | 5.08 |
| gender | 30 | 2.50 |

Products (`product_id` str, `product_name` str, `category` str, `price_inr` int64): no column has a missing value.

Orders (`order_id`, `customer_id`, `product_id`, and `order_ts` are str; `quantity` and `discount_pct` are int64; `status` is str; `order_amount` is str; `delivery_minutes` is float64):

| Column | Missing | Missing % |
| --- | ---: | ---: |
| order_id, customer_id, product_id, quantity, order_ts, status, discount_pct | 0 | 0.00 |
| order_amount | 53 | 0.41 |
| delivery_minutes | 2,541 | 19.42 |

The 53 missing amounts are the token `N/A`, which pandas reads as missing. A further 67 cells contain the literal `error`. Together, 120 `order_amount` values cannot be converted to a number.

Tickets: `order_id` is blank on 8 rows (4.88%) and `text` is blank on 4 rows (2.44%). The other ticket columns have no blanks.

| Audit figure | Value |
| --- | ---: |
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules

Customers:

- City: strip spaces and apply title case. This rewrote 476 city values and left six cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
- Gender: map every spelling to `male`, `female`, or `unknown`. 30 blanks became `unknown`, and the other 1,170 values were remapped. Result: female 610, male 560, unknown 30.
- Age: 6 values outside 10–100 were set to missing. Those 6 plus the 61 blank ages, 67 in total, were filled with the median age of the customer's city using `groupby` and `transform`. After filling there are 0 missing ages, and every age is between 18 and 60.
- `signup_date` was parsed to a datetime for all 1,200 rows.

Orders, in this order:

- `order_ts` was parsed to a datetime for all 13,084 rows.
- 119 exact duplicate rows were removed.
- 656 older rows for an `order_id` that also had a later status were removed. Each `order_id` keeps the latest `order_ts` only.
- 112 `order_amount` values that remained unparseable after the row drops were set to missing and filled with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`.
- 105 negative `delivery_minutes` values that remained after the row drops were set to missing. 774 completed orders still had a missing delivery time and were filled with the median delivery time of that product category among orders that already had a time (`groupby` and `transform`). Cancelled orders were left missing.

Products were stored as catalogued. No product value was changed.

After cleaning: 12,309 order rows (one per `order_id`); 11,110 completed and 1,199 cancelled; no pending rows; 0 missing `order_amount`; 0 missing `delivery_minutes` on completed orders and 1,199 missing on cancelled orders. The sum of `order_amount` over completed orders is INR 2,998,668.80.

The validation in `scripts/run_part_a.py` asserts these checkpoints and prints `ALL CHECKPOINTS PASSED`. Clean CSVs are in `data/clean/`. `data/instamart.db` holds tables `customers`, `products`, and `orders`, with `order_ts` stored as `YYYY-MM-DD HH:MM:SS`.

### SQL

Only completed orders count as revenue. Queries live in `sql/queries.sql`.

| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING | 30 customers |
| Q2 city × category revenue | 30 rows, summing to INR 2,998,668.80. Largest cell: Bengaluru / Electronics, INR 233,909.45 |
| Q3 products above average catalogue price | 11 products |
| Q4 customers with no completed order | 25 customers |
| Q5 spend above the average customer total | 550 customers. Average of customer totals: INR 2,552.06 |
| Q7 daily revenue and running total | 183 days. Final cumulative revenue: INR 2,998,668.80 |

Q6 top customer in each city, by completed spend:

| City | Customer | Total spend (INR) |
| --- | --- | ---: |
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

Exactly 8 rows have `RANK()` different from `ROW_NUMBER()`: tied customers share one rank and the next rank number is skipped, while `ROW_NUMBER()` still gives every row its own integer.

### Pandas cross-check and exploration

A `pivot_table` of completed revenue by city and category, with margins, has grand total INR 2,998,668.80, the same as the SQL total.

Completed `order_amount` by city:

| City | Sum (INR) | Mean (INR) | Count |
| --- | ---: | ---: | ---: |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00. The 1.5×IQR fences are INR -283.50 and INR 645.30. 897 orders sit above the upper fence and none sit below the lower fence.

The largest absolute correlation on completed orders is `order_amount` with `discount_pct` at -0.095. `order_amount` with `quantity` is +0.087. A correlation of this size does not establish causation.

These rows are synthetic Instamart orders for groceries, dairy, snacks, personal care, and small electronics across six Indian cities, with each completed order carrying an amount in INR, a discount, a quantity, and a delivery time. The amount distribution is right-skewed: the mean (INR 269.91) is about twice the median (INR 135.00), and the skewness is 4.61, so a typical basket sits near the lower quartile of INR 64.80 while the upper quartile is INR 297.00. The 897 values above the upper fence of INR 645.30 are large but plausible electronics and multi-unit baskets, not leftover dirty rows, and nothing falls below the lower fence. Among amount, discount, quantity, and delivery time, no pair moves together strongly. The strongest link is a slight negative association between amount and discount, and quantity tracks amount only weakly because catalogue prices differ. A correlation of this size does not establish causation.

## Part B

RFM is computed on completed orders as of 2025-10-01 00:00:00. 1,175 customers have at least one completed order. Recency median 19 days, minimum 0, maximum 180. Frequency mean 9.46, maximum 33. Monetary mean INR 2,552.06, maximum INR 14,766.55.

K-means on the standardised RFM columns (`n_init=10`, `random_state=42`):

| K | Inertia | Silhouette |
| ---: | ---: | ---: |
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Inertia is the within-cluster sum of squares, so splitting customers into more clusters can only lower it: the K=8 inertia of 366.5 is smaller than the K=2 inertia of 1,820.8 for that reason alone. That drop does not say which K is useful, because a model with one cluster per customer would have inertia near zero and no campaign behind it. K has to be read from the elbow, the silhouette score, and whether someone can act on the groups.

K=4 is the choice. Inertia falls sharply from K=3 (1,268.2) to K=4 (743.1), and the silhouette score peaks at 0.526. Four groups are also distinct enough to brief: a small high-spend set, a high-frequency set, a long-quiet set, and everyone else. Cluster numbers are labels with no ordering. Premium is the segment with the highest mean monetary value, not the segment whose id is largest.

| Segment id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | Occasional | 436 | 28.86 | 3.77 | 1,057.62 | Offer a small basket-building bundle to raise visit frequency. |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Make a personal retention call and send an electronics offer before the next expected order. |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Protect the weekly habit with a modest grocery loyalty reward. |
| 3 | At Risk | 182 | 109.04 | 2.20 | 685.99 | Send a win-back offer this week to customers who have gone quiet. |

`is_premium` is true only for the 138 Premium customers. PCA on the scaled RFM matrix explains 0.634 and 0.218 of the variance (total 0.852). Segments are written to `outputs/customer_segments.csv`.

### Lapse model

The cutoff is 2025-09-01. The cohort is every customer with a completed order before that cutoff: 1,156 customers, of whom 443 lapsed (38.32%) because they had no completed order from 2025-09-01 through 2025-09-30. Features use only orders dated before the cutoff, plus the clean customer table: recency from the cutoff, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, and tenure in days. No order on or after 2025-09-01 entered the features. Segment labels were not used as features. The stratified split (`test_size=0.2`, `random_state=42`) has 924 training rows (354 lapsed, 570 not lapsed) and 232 test rows (89 lapsed). Always predicting "not lapsed" on the test set has accuracy 61.64% and a lapsed-class F1 of 0.

No feature column contains a missing value. City, gender, and favourite category are one-hot encoded with `drop="first"` inside a pipeline, and numeric features are standardised. The encoder and the scaler are fitted on the training rows only. The scaler's learned mean of frequency is 8.1147, against a full-cohort mean of 8.0285. They differ because the scaler never saw the test rows, and the training subset does not have the same frequency mean as the whole cohort.

Logistic regression was tuned over `C` in {0.01, 0.1, 1, 10, 100} with 5-fold F1. The best setting is `C=100`, with cross-validated F1 0.6506. On the untouched test set the lapsed class has precision 0.6444, recall 0.6517, and F1 0.6480. Confusion matrix rows are actual not-lapsed then lapsed, and columns are predicted not-lapsed then lapsed: [[111, 32], [31, 58]].

The random forest (100 trees, `random_state=42`) was tuned over `max_depth` in {3, 5, 8} and `min_samples_leaf` in {5, 10, 20}, which is 45 fits. The best setting is depth 8 and leaf 10, with cross-validated F1 0.6811. Test precision 0.6591, recall 0.6517, F1 0.6554. Confusion matrix: [[113, 30], [31, 58]].

The selected model is the random forest. Missing a lapsing customer means the retention team never intervenes and the customer can leave, while wrongly targeting a loyal customer only spends a call or a coupon. On this test split both models have the same recall (0.6517), so they miss the same number of lapsing customers, but the forest is more precise (0.6591 against 0.6444) and has the higher cross-validated F1 (0.6811 against 0.6506). Both clear the 0.55 F1 bar and the baseline F1 of 0, and the forest is the model that was saved.

Oversampling the training set only, with `random_state=42`, duplicated lapsed training rows until both classes had 570 rows. The best logistic regression, refit on that balanced training set and scored on the original test set, moved from precision 0.6444, recall 0.6517, F1 0.6480 to precision 0.5891, recall 0.8539, F1 0.6972. Recall moved the most. It rose, and precision did not rise. For the retention team that means more of the customers who will actually lapse are contacted, and more loyal customers are contacted as well.

Random-forest importances, with a one-line reading:

| Feature | Importance | Reading |
| --- | ---: | --- |
| frequency | 0.2796 | How often the customer ordered before the cutoff is the strongest clue that they will order again in September. |
| monetary | 0.2712 | Total pre-cutoff spend is almost as useful as frequency. |
| recency | 0.1151 | Days since the last pre-cutoff order matter, but less than how much and how often they bought. |
| avg_delivery_minutes | 0.0612 | Slower historical deliveries play a smaller role in the score. |
| cancellation_rate | 0.0523 | A higher share of cancelled orders is a real but weaker signal. |

Logistic coefficients on the scaled features:

| Feature | Coefficient | Reading |
| --- | ---: | --- |
| recency | +0.0396 | A longer gap before the cutoff slightly raises the lapse score. |
| frequency | -1.6982 | Customers who ordered more often are much less likely to be scored as lapsed. |
| monetary | +0.2846 | Once frequency is in the model, a higher pre-cutoff spend slightly raises the lapse score rather than protecting the customer. |

The selected forest was saved to `outputs/lapse_model.joblib` and loaded back. The reloaded model reproduces the original test predictions exactly.

Among test customers in the Premium segment, the ten highest lapse probabilities are in `outputs/retention_list.csv`. 2 of those 10 had actually lapsed. The Premium flag comes from the whole-window RFM, which includes the September orders that define the label, so this list is illustrative and not a leakage-free evaluation. The model's own features still use only orders before the cutoff.

No encoder, scaler, or resampler saw test rows. No order dated 2025-09-01 or later entered the features. Segment labels from the clustering step were not used as features.

## Part C

Tickets are processed in file order. `mock_llm.reset()` is called before every full run.

4 tickets are empty or whitespace (`skipped_empty`) and 4 are longer than 400 estimated tokens (`flagged_long`, 734 tokens each). Neither kind is sent to the model.

Redaction, applied in order to all 164 tickets before any prompt is built: 55 emails, 91 phone numbers, and 132 order ids, 278 substitutions in total. Running the same three patterns on the redacted texts finds 0 matches. The user message is exactly the redacted ticket text. The run asserts that no email, phone number, or order id remains in any prompt.

Both system prompts contain `CLASSIFY`, list billing, delivery, technical, and fraud, and require a JSON object `{"category": "...", "reason": "..."}`. The zero-shot prompt does not contain `EXAMPLES:`. The few-shot prompt contains `EXAMPLES:` and two labelled examples, one of them fraud.

Every model call retries up to 3 times. The wait before retry number n is `0.05 × 2^n × uniform(0.9, 1.15)`. A full run over the 156 eligible tickets raises 19 `TransientLLMError`s. Each one is recovered on retry number 0, and each logged wait is between 0.045 and 0.0575 seconds. No ticket ends in a permanent error.

Output checks on that full run: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`. The keyword path uses `re.findall(r"[a-z]+", text.lower())`, which splits a chatty reply such as `billing-related` into the words `billing` and `related`. Invalid JSON categories are not guessed; they go to a person.

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 147 | 126 | 85.71% | 9,556 | 65.01 |
| Few-shot | 147 | 147 | 100% | 14,860 | 101.09 |

Zero-shot confusion (true category, then predicted category):

| True \ predicted | billing | delivery | technical | fraud |
| --- | ---: | ---: | ---: | ---: |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Zero-shot accuracy is 85.7% (126 of 147), and every mistake is a fraud ticket predicted billing (11) or a technical ticket predicted billing (10). Few-shot accuracy is 100% (147 of 147) and costs about 36 extra tokens per classified ticket (101.09 against 65.01; 14,860 tokens against 9,556). Those extra tokens are worth paying when a fraud ticket sent to billing is expensive, because here they removed all 21 mistakes, including the 11 fraud misses. On a pile of obvious billing and delivery tickets the shorter zero-shot prompt already matches the label, so the examples would be wasted spend.

The zero-shot budget demo uses `TOKEN_BUDGET = 6000`. 58 tickets end as `budget_exhausted` and are saved in `outputs/triage_budget_demo.csv`. The final file `outputs/triage_results.csv` is the few-shot run with `TOKEN_BUDGET = 100000`. It has 164 rows. Routing uses `outputs/customer_segments.csv`. Empty, long, invalid, and budget-exhausted tickets go to `human_review` at `NORMAL`. A fraud category goes to `security_team` at `HIGH`. Otherwise a Premium customer goes to `premium_desk` at `HIGH`, and everyone else goes to `support_queue` at `NORMAL`. A customer with no segment row is not Premium.

| Route | Tickets |
| --- | ---: |
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

The 135 JSON replies and the 12 keyword replies were accepted by those rules. The 9 invalid replies, the 4 empty tickets, and the 4 long tickets were not guessed; all 17 went to a person. In the budget demo, the 58 exhausted tickets also go to a person and are not classified.

## Part D

The five policy files are split into 40-word chunks with an overlap of 10 words. That produces 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2.

TF is how often a word occurs inside one chunk. IDF down-weights a word that occurs in many chunks, so a rarer word counts for more when chunks are ranked. One limitation of TF-IDF is that two phrasings with the same meaning but different words, such as "money back" and "refund", do not match. The query `refund policy late delivery` returns three chunks, all from `refund_policy.txt`.

`answer` retrieves three chunks and sends a `POLICY ASSISTANT` system message plus a user message of the form `CONTEXT:`, the chunks, `QUESTION:`, and the question.

| Setting | Expected document in top 3 | Answer contains the expected fact | Unanswerable questions get the fallback |
| --- | --- | --- | --- |
| Overlap 10 | 12/12 | 12/12 | 2/2 |
| Overlap 0 | 12/12 | 10/12 | 2/2 |

With no overlap, "What delivery time makes a completed order eligible for a full refund?" fails. The sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount." is cut after "customer". The retrieved answer is only the second fragment, "receives a full refund of the order amount.", so the fact "more than 45 minutes" is missing even though `refund_policy.txt` is still retrieved. "How soon is money returned when a payment fails but is deducted?" breaks the same way: "48 hours" sits on the other side of a chunk boundary. An overlap of 10 words keeps each of those sentences inside one chunk.

The refund agent calls five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. Only a completed order is refunded. Delivery over 45 minutes refunds 100% of `order_amount`. Delivery in 45 minutes or less refunds INR 0. A refund above INR 1000 is escalated and is not issued automatically. Issued and escalated actions are appended to `outputs/refund_ledger.jsonl`. The crash-and-resume demonstration below issues S1 a second time, so that ledger contains two `issue_refund` lines for ORD-100048 (INR 128.25), one `issue_refund` for ORD-100067 (INR 135.00), and one `escalate_to_human` for ORD-100145 (INR 1,999.00).

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
| --- | --- | ---: | ---: | ---: |
| S1 | REFUND | 128.25 | 5 | 972 |
| S2 | REFUND | 135.00 | 5 | 972 |
| S3 | ESCALATE | 1,999.00 | 5 | 959 |
| S4 | DENIED | 0.00 | 4 | 665 |
| S5 | DENIED | 0.00 | 3 | 382 |
| S6 | NOT_FOUND | 0.00 | 2 | 102 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,090 |

S7 finishes all six model calls under the default token limit of 6,000 (it uses 3,090 tokens) and then stops with `FALLBACK_MAX_ITERATIONS`. The S1 demonstration with `token_limit=200` ends as `FALLBACK_TOKEN_LIMIT` and does not run the tool that call had just requested. The demonstration with `time_limit_seconds=0` ends as `FALLBACK_TIMEOUT` before the first model call. Each fallback keeps the steps so far and a short human-handoff summary.

The S1 crash-and-resume run saves a checkpoint in `state/` after every tool observation. A simulated crash is raised after the second tool. `resume_agent` continues from that file. The tools executed across the crash and the resume, in order, are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. None is repeated. The final status is REFUND and the amount is INR 128.25.

`outputs/agent_runs.jsonl` has one JSON line for each of the seven scenarios, including status, final answer, model calls, tokens, and every step.

This agent is a ReAct loop: after each tool observation the model chooses the next tool or stops. Measured calls were 5 for S1 (972 tokens), 5 for S2, 5 for S3, 4 for S4, 3 for S5 (382 tokens), 2 for S6 (102 tokens), and 6 for S7 before the iteration cap. A plan-and-execute design would ask once for the whole step list and replan only on an error, so the straight refund path could be cheaper than S1's five calls. The branch is the hard part. S6 must stop when `get_order` returns not found, and S5 must deny a cancelled order instead of calculating a refund. ReAct sees that observation on the next call and changes course. A plan written up front would keep going unless a new model call replans, which spends the call the design hoped to save. S7 also shows the cost of ReAct: with `[LOOP]` in the request the model asks for `search_policy` forever, so the six-call cap has to hand the case to a person. I would keep ReAct for this refund workflow. The sequence is short, the decision is exactly the unexpected observation, and five calls on a normal refund are cheaper than issuing money the policy does not allow.

## Limitations and Next Steps

The Premium flag is a post-hoc whole-window label. It is built from RFM through 30 September 2025, which includes the September window that defines the lapse label, so `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation. No encoder, scaler, or resampler saw test rows. No order dated 2025-09-01 or later entered the lapse features. Segment labels were not used as features.

The data are synthetic. `mock_llm.py` is a deterministic stand-in, so the few-shot accuracy of 100% describes this mock and not a production classifier. TF-IDF misses a paraphrase that uses different words. The refund ledger from a full Part D run also contains the second S1 issue written by the crash-and-resume demonstration.

A next evaluation should rebuild the Premium flag from orders before the cutoff only, check that the lapse probabilities are calibrated, and sample the nine invalid ticket replies for a human labelling review.
