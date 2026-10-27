# Fund universes and report archives

## What is available now

Only the 12 annual F-series funds have history in this repository. They remain
the default. The Studio's other group choices are ready for metadata and data,
but never substitute F-series observations when a selected group has no history.

The universe section offers a group, annual/quarterly outcome period, optional
series/strategy/reference-asset/outcome-month filters, and one broad buffer
category. The exact ticker list is editable after filtering. An empty list or a
portfolio larger than the list cannot run. Initial selection and every subsequent
replacement use the same selected universe. The current portfolio size limit is
12. Selecting a fund's outcome month filters products; the launch month sets the
start of a separate investment backtest.

Annual and quarterly products are kept in separate runs, as in the reference
website's tabs. The annual timing defaults are 180/75 days; choosing quarterly
sets 45/18.75 days. Both retain downside/upside reset sigma 0.07/0.05. The quarterly
settings are proposals, not calibrated claims. All actual scoring values are
shown in the component cards and exported in the configuration. Changing the
review cadence does not change these scoring values or a product's outcome term.

## Add future funds

1. Add metadata rows to `input_data/fund_catalog.csv`. Required columns:

   | Field | Meaning |
   | --- | --- |
   | ticker | Unique uppercase symbol, without whitespace |
   | series | Actual product family, e.g. F, D or G; not inferred from ticker |
   | group | Classification, e.g. US ETFs or UCITS |
   | outcome_type | Annual or Quarterly |
   | outcome_month | JAN through DEC, or Quarterly |
   | strategy | Buffer, Deep Buffer, Moderate Buffer, etc. |
   | buffer_start | Positive depth of first loss before protection, decimal fraction |
   | buffer_end | Positive depth of end of protection, decimal fraction |
   | reference_asset | SPY, QQQ, EFA, etc. |
   | currency | Currency of the NAV used in the metrics |

   For a 5–15% deferred buffer use 0.05 and 0.15, not negative percentages.
   There are no placeholder price records. Metadata alone does not make a fund
   eligible: it must also have history. The broad filter categories are immediate
   protection up to 10%, above 10–20%, above 20%, and deferred protection.

2. Put normalized histories into `input_data/fund_metrics/*.csv`. Required columns:
   `ticker`, `date`, `fund_value`, `reference_asset_return`,
   `remaining_outcome_days`, `remaining_cap_net`, `remaining_buffer_net`,
   `downside_before_buffer_net`, `fund_cap_net`, `buffer_net`.
   Percentages are decimal fractions; dates should use ISO format. Returns since
   reset and remaining days must reflect the actual fund's own outcome cycle.
   Duplicate ticker/date rows across files are rejected; files can extend histories
   but cannot overlap existing rows. Invalid held NAVs still fail rather than
   being filled. Initial allocation needs enough eligible funds and, under
   next-close execution, a preceding signal session.

3. Restart/reload the Studio. Cache signatures include catalogue and supplemental
   files. New groups become usable when history is available. The current app
   requires USD NAVs because benchmark comparisons use USD. Other currencies
   require a deliberate conversion/benchmark policy before running; this version
   does not invent FX returns. The supplied investment-launch/review calendar is
   still the holiday-adjusted monthly roll calendar in `roll_dates.csv`; review
   that convention when integrating a product with different trading calendars.

The default CLI dataset also loads supplemental CSVs. Exported configurations
contain explicit ticker lists, so replay does not silently expand when new funds
are added. Example:

```bash
python main.py --config config/ucit-conservative.json --universe FFEB,FMAR,FJUN --portfolio-size 3
```

Library callers pass the universe to `prepare_data(..., universe=tickers)` and
`BacktestConfig(universe=tickers, initial_selection='highest_score')` together.
The simulator rejects mismatched prepared/configured universes. ResearchSession
caches separate score panels by both scoring parameters and universe.

## Launch dates, purchases and reviews

A January launch uses the first observed session after the qualifying January
roll date, on or after the selected start bound. This can be later than the date
picker's lower limit. Each chosen launch month is its own backtest and starts
with portfolio NAV 100. With next-close execution, initial ranking uses the
preceding observed session; the top N eligible funds are purchased at the entry
close and begin earning subsequent returns. A January launch does not force a
purchase of FJAN, and FJAN need not even be in a custom universe.

Quarterly review dates remain the March/June/September/December calendar rolls.
For example, entering in January does not shift reviews to April/July/October.
The first review can therefore be in March. Monthly, semiannual and annual
reviews retain their existing calendars. Trigger breaches and weight-band rules
control whether reviews actually produce trades.

For explicit universes, Hold NAV is the initially selected fund or equal-weight
basket held without trading. Legacy single-fund configurations with `universe`
omitted preserve the original named-cohort benchmark and workbook parity. SPY and
BUFR remain comparative benchmarks, regardless of the chosen reference asset;
benchmark selection for other asset classes can be extended when data arrives.

## Trading assumptions

The client presets use next-close execution, zero trading costs and zero minimum
holding days. These controls already execute in the engine; the new UI provides
plain-language labels and help. Five basis points means 0.05%, or $5 per $10,000
of one-way traded capital. Fees apply to actual replacements/weight adjustments,
not initial entry. A zero holding period adds no restriction. A positive period
counts observed trading sessions since actual purchase and does not block
weight-only resizing. Named research experiments that test holding delays retain
their explicit settings rather than silently turning those tests off.

## Where reports go

Automatic archiving is on by default. Its configurable root is:

```text
G:\Shared drives\Portfolio Management\Funds\Archive\Model Portfolio Allocation\ECR Backtest Results
```

Every completed run creates a unique dated subfolder containing:

- `ecr_results.xlsx`, including the selected Fund Universe sheet;
- CSVs, `config.json`, `manifest.json` and the selected metadata in `universe.csv`;
- `charts/` with PNG reports and interactive HTML charts, plus the local Plotly library;
- top-level PNG summaries and `results.zip` containing the full downloadable report;
- `ARCHIVE_COMPLETE.txt`, written only after the folder is fully saved.

Folders include a sanitized run name, strategy ID and random suffix, so repeated
runs never overwrite each other. The ZIP matches the download button's bytes.
The app displays the actual saved path. The path belongs to the machine running
Streamlit: on your Windows localhost it can use your mapped G drive; the remote
cloud environment cannot reach that mapping. A disconnected/unwritable drive
produces a visible warning without discarding the computed results or ZIP.
Failed writes remove that attempt's partial run folder when possible and never
report success. No substitute local folder is silently presented as a shared-drive
save. You may change or disable the archive under **Archive location**.

The archive policy applies to Studio runs. CLI runs retain their explicit
`--output` or timestamped local `output/` behavior. Browser downloads still use
your browser's normal download location in addition to the automatic archive.
