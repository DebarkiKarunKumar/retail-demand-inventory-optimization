# Methodology

This document records the main design decisions and the reasons behind them. For numerical outcomes, see [RESULTS.md](RESULTS.md).

## 1. Business questions

Can a model forecast short-term retail demand more accurately than simple historical rules? Can those forecasts support inventory replenishment decisions that reduce shortages without excessive holding cost?

## 2. Data and selection

The public Walmart M5 dataset includes daily item-level sales and supporting calendar and price data. The project focuses on store `CA_1`, department `FOODS_3`, selecting 20 eligible, historically high-volume series using training-period information. This keeps the experiment manageable but limits generalization. Data preparation included checks for source integrity, chronological continuity, selected-series coverage, and consistency of the calendar mapping. Exploratory analysis used the training data rather than the final holdout.

## 3. Chronological design

| Period | M5 day columns | Role |
|---|---|---|
| Training | `d_1`–`d_1857` | Fit and historical examples |
| Validation | `d_1858`–`d_1885` | Compare approaches |
| Final fit | Through `d_1885` | Refit frozen XGBoost configuration |
| Final holdout | `d_1886`–`d_1913` | One protected evaluation |

Forecasts cover 28 days from a fixed origin. Validation sales are never used as inputs to generate later validation predictions. Final holdout sales were excluded from model development, feature selection and hyperparameter decisions.

## 4. Baselines and Ridge

- **Naive:** repeat the most recent training-day demand for all 28 days.
- **Weekly seasonal naive:** repeat the last seven observed days four times. This became the primary simple benchmark.
- **Global Ridge:** one regularized regression across the 20 products, with horizon-aware features and training examples drawn from historical origins. It uses demand and calendar signals and direct multi-horizon predictions rather than observed future demand.

These comparisons establish whether added complexity produces a measurable improvement.

## 5. XGBoost feature engineering

XGBoost is a tabular regression model, so temporal structure is represented through historical features. The reported selected configuration uses `reg:squarederror`, 300 estimators, maximum depth 5, learning rate 0.05, subsample and column subsample 0.8, minimum child weight 10, L2 regularization 5, and random state 42.

Feature groups include:

- Product identity and forecast horizon.
- Sales at the origin and historical lags (7, 14, 28 and 56 days).
- Trailing demand means, standard deviations, zero-sales proportions and same-weekday history.
- Known-in-advance target calendar features, such as weekday, month, event type and SNAP indicator.
- Historical observed price levels, changes, recent-price averages, price-history coverage and missingness indicators.

Historical demand and price features are calculated using only information available at each training or forecast origin. Missing historical price values remain missing. Known calendar fields can describe future dates, but **future sales cannot be feature inputs**. The selected configuration was chosen using historical backtests and validation; the final holdout was not used for tuning.

## 6. Evaluation

The project reuses established evaluation functions for:

- **MAE:** average absolute error, in units per product-day.
- **RMSE:** square root of average squared error, more sensitive to large misses.
- **WAPE:** `100 × sum(abs(actual − predicted)) / sum(abs(actual))` across the evaluation observations, when the denominator is nonzero.
- **Signed bias:** average `(predicted − actual)`; positive means overforecasting.

The **primary selection measure** is pooled product-day WAPE, not aggregate daily WAPE. Aggregating product forecasts before calculating error allows over- and underforecasting across products to cancel. Product-level errors remain useful diagnostics, especially for low-demand items whose individual WAPE can be unstable.

## 7. Frozen holdout and reproducibility

After model selection, the XGBoost configuration was fitted through `d_1885`. The forecast-only step produced 560 predictions for 20 products × 28 days. Their SHA256 hash was recorded before the separate scoring phase. The scoring workflow checked integrity and evaluated saved predictions without refitting. The latest reported full synthetic test run passed 457 tests.

This documents reported local execution, not an independent production audit. No further tuning should be done on the holdout.

## 8. Inventory policies

The inventory experiment compares two **predefined** demand plans:

- **Policy A:** average of the eight historical observations for each weekday over `d_1830`–`d_1885`, repeated for the 28-day evaluation window.
- **Policy B:** the frozen 28-day XGBoost forecasts, unchanged throughout the simulation.

Both policies face the same recorded sales as a **demand proxy**. Each product has identical initial inventory under both policies, based on Policy A's expected first `L+1` days plus a common safety-stock buffer. Historical standard deviation uses the 56-day window. Safety stock is `ceil(1.645 × historical_std × sqrt(protection_days))`; this is a heuristic, not a proven 95% service guarantee.

### Daily morning-review sequence

1. Receive orders due that morning.
2. Determine available inventory and outstanding orders.
3. Set an order-up-to target using the fixed demand plan over `L+1` days, truncated at the horizon, plus safety stock.
4. Order `max(0, target − inventory_position)` units, where inventory position includes available stock and outstanding orders.
5. Observe that day's recorded sales as demand proxy; fulfill available units, treat the remainder as lost demand, and calculate ending stock and illustrative costs.

An order placed on day `t` arrives before demand on day `t+L`, not on the same day. Orders that would arrive after day 28 are suppressed. The model does not issue new forecasts each day: only inventory state and outstanding orders change.

### Scenarios and costs

Nine scenarios combine lead times of 1, 2 and 3 days with shortage costs of 0.10, 0.50 and 1.00 per unmet unit. Holding cost remains 0.01 per unit per day. Total simulated cost = holding cost on end-of-day inventory + shortage cost on unmet demand. There are no ordering, procurement, salvage or disposal charges. Costs are **illustrative units**, not actual Walmart economics.

## 9. Implementation map

The project separates preparation, evaluation, model experiments and the inventory simulator into modules under `src/`, with regression tests under `tests/`. The final holdout is generated and scored in separate commands, so saved predictions can be checked before the actual sales are joined. The inventory simulator consumes those saved artifacts rather than retraining the model.

## 10. Safeguards and limitations

Synthetic tests cover order arrival timing, inventory conservation, lost-sales behavior, cost reconciliation, input immutability and scenario coverage. The simulation reports per-product and portfolio outcomes for all nine scenarios rather than selecting only favorable cases.

Limitations include a selected 20-series cohort, a single 28-day demand path, possible sales censoring, assumed starting inventory and costs, unlimited replenishment supply, no substitution or spoilage, and finite-horizon effects. The experiment demonstrates a decision-support method; it does not establish live operational benefits.
