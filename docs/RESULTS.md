# Results and interpretation

These results are based on recorded local execution outputs. Values are rounded for readability; the public repository should retain the corresponding source artifacts or a traceable summary. They are not independently reproduced from the raw dataset here.

## 1. Forecast validation

The same 20 product series and 28-day validation window were used for the main model comparison (560 product-day observations).

| Model | Pooled product-day WAPE ↓ | Pooled MAE ↓ |
|---|---:|---:|
| Weekly seasonal naive | 47.24% | 8.50 |
| Global Ridge | 44.20% | 7.96 |
| **XGBoost** | **39.87%** | **7.18** |

Relative WAPE reduction versus seasonal naive: approximately **15.6%**, using the saved validation summary. These are **validation** results, not final holdout results.

## 2. Final XGBoost holdout

The selected model was refitted through `d_1885`, its 560 predictions were frozen, and the saved predictions were scored once on `d_1886`–`d_1913` without scoring-time refitting or tuning.

| Metric | Pooled product-day (560 rows) | Aggregate daily (28 rows) |
|---|---:|---:|
| WAPE | **34.80%** | 12.27% |
| MAE | 6.44 | 45.44 |
| RMSE | 9.82 | 57.44 |
| Signed bias | +0.86 | +17.17 |

The holdout WAPE is lower than validation WAPE (39.87%), but these periods differ; this is not proof that the model improves over time. Aggregate daily WAPE is much smaller partly because product-level errors offset when aggregated. Pooled product-day WAPE remains the primary metric.

The local audit confirmed 560 frozen rows, 20 unique products, 28 dates, zero duplicate product-day keys and matching SHA256. The latest full test suite reported **457 passed**.

## 3. Inventory simulation: base scenario

Two predefined policies faced the same recorded M5 sales as a demand proxy. Base scenario: one-day lead time, holding cost 0.01 per unit per day, shortage penalty 0.50 per unmet unit.

| Metric | Historical weekday policy | Frozen XGBoost policy |
|---|---:|---:|
| Demand proxy units | 10,365 | 10,365 |
| Fulfilled units | 10,233 | 10,254 |
| Unmet units | 132 | **111** |
| Fill rate | 98.73% | **98.93%** |
| Product-days with shortage | 17 | **14** |
| Average daily total inventory | **502.50** | 530.04 |
| Holding cost | **140.70** | 148.41 |
| Shortage cost | 66.00 | **55.50** |
| Total simulated cost | 206.70 | **203.91** |

The XGBoost-informed policy simulated **21 fewer unmet units (15.9%)** and **1.35% lower total cost** in this scenario, but held approximately **5.5% more inventory** on average. The cost difference is only 2.79 illustrative cost units.

## 4. Nine-scenario sensitivity analysis

**Positive reduction = XGBoost policy had lower simulated total cost. Negative = XGBoost policy cost more.** Shortage-cost scenarios change the accounting cost, not the physical order policy.

| Scenario | Lead time (days) | Shortage cost | Historical cost | XGBoost cost | XGBoost cost reduction |
|---|---:|---:|---:|---:|---:|
| L1_P010 | 1 | 0.10 | 153.90 | 159.51 | **−3.65%** |
| L1_P050 | 1 | 0.50 | 206.70 | 203.91 | **+1.35%** |
| L1_P100 | 1 | 1.00 | 272.70 | 259.41 | **+4.87%** |
| L2_P010 | 2 | 0.10 | 192.46 | 199.50 | **−3.66%** |
| L2_P050 | 2 | 0.50 | 256.86 | 247.50 | **+3.64%** |
| L2_P100 | 2 | 1.00 | 337.36 | 307.50 | **+8.85%** |
| L3_P010 | 3 | 0.10 | 227.25 | 237.50 | **−4.51%** |
| L3_P050 | 3 | 0.50 | 293.65 | 287.50 | **+2.09%** |
| L3_P100 | 3 | 1.00 | 376.65 | 350.00 | **+7.08%** |

**Six of nine scenarios** had lower simulated costs under XGBoost. In all three low-shortage-penalty scenarios, the additional holding cost outweighed the shortage reduction.

### Shortages by lead time

| Lead time | Historical unmet units | XGBoost unmet units | Reduction |
|---|---:|---:|---:|
| 1 day | 132 | 111 | 21 |
| 2 days | 161 | 120 | 41 |
| 3 days | 166 | 125 | 41 |

These counts repeat across cost scenarios with the same lead time because the policies and demand are unchanged; only the cost assigned to shortages changes.

## 5. Business interpretation

Better forecasting error did not automatically translate into lower inventory cost. The XGBoost-informed policy consistently reduced simulated shortages, but typically held more stock. It reduced total cost when avoiding shortages was sufficiently valuable. When shortages were relatively inexpensive, the historical policy cost less.

This is a **decision-support proof of concept**, not a claim that Walmart actually reduced stockouts, inventory or spending. Recorded sales may understate unconstrained demand; simulated unmet units are not observed historical lost sales. Cost rates, starting inventory, replenishment rules and service assumptions are illustrative. The model was not deployed or tested in live retail operations.
