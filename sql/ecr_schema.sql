-- ecr schema (MySQL 8). The backtest reads only these tables; see README "Inputs and provenance".
-- Engine input columns are plain DOUBLE: DECIMAL (or DOUBLE(M,D)) rounds values and changes data_id.
-- Load CSVs with pandas' default read_csv and bound Python floats, not LOAD DATA or the Workbench
-- import wizard: those parse about 300 values to a neighbouring double and change data_id.

CREATE SCHEMA IF NOT EXISTS ecr;

-- One row per fund; drives universe filters. Blank text is '' (never NULL).
CREATE TABLE ecr.fund_catalog (
    ticker           VARCHAR(12)  PRIMARY KEY,
    series           VARCHAR(10)  NOT NULL DEFAULT '',          -- e.g. F, D, G
    fund_group       VARCHAR(50)  NOT NULL DEFAULT '',          -- "group" in the app (reserved word in SQL)
    outcome_type     VARCHAR(10)  NOT NULL CHECK (outcome_type IN ('Annual', 'Quarterly')),
    outcome_month    VARCHAR(10)  NOT NULL CHECK (outcome_month IN
                         ('JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','Quarterly')),
    strategy         VARCHAR(60)  NOT NULL DEFAULT '',
    buffer_start     DOUBLE       NOT NULL,                     -- decimal fraction, e.g. 0.05
    buffer_end       DOUBLE       NOT NULL,
    reference_asset  VARCHAR(12)  NOT NULL DEFAULT '',
    currency         CHAR(3)      NOT NULL DEFAULT '',
    CHECK (buffer_start >= 0 AND buffer_end > buffer_start AND buffer_end <= 1),
    CHECK (ticker = UPPER(TRIM(ticker)))
);

-- Daily fund metrics as decimal fractions. The engine reads ticker, date, fund_value and the seven
-- inputs; the stored scores (dbb_score .. ecr_2_0) are for validation only and are never loaded.
CREATE TABLE ecr.website_metrics (
    `date`                                 DATE         NOT NULL,
    ticker                                 VARCHAR(12)  NOT NULL,
    fund_sleeve                            VARCHAR(32)  NOT NULL,
    series                                 VARCHAR(16)  NULL,
    fund_value                             DOUBLE       NOT NULL CHECK (fund_value > 0),
    fund_return                            DOUBLE       NULL,
    reference_asset_value                  DOUBLE       NULL,
    reference_asset_return                 DOUBLE       NULL,
    remaining_outcome_days                 SMALLINT     NOT NULL,   -- a NULL would make the column float
    remaining_cap                          DOUBLE       NULL,
    remaining_cap_net                      DOUBLE       NULL,
    reference_asset_return_to_realize_cap  DOUBLE       NULL,
    remaining_buffer                       DOUBLE       NULL,
    remaining_buffer_net                   DOUBLE       NULL,
    downside_before_buffer                 DOUBLE       NULL,
    downside_before_buffer_net             DOUBLE       NULL,
    reference_asset_to_buffer_end          DOUBLE       NULL,
    unrealized_option_payoff               DOUBLE       NULL,
    unrealized_option_payoff_net           DOUBLE       NULL,
    fund_cap_net                           DOUBLE       NULL,
    buffer_net                             DOUBLE       NULL,
    dbb_score                              DOUBLE       NULL,
    buffer_integrity                       DOUBLE       NULL,
    cap_integrity                          DOUBLE       NULL,
    reset_proximity                        DOUBLE       NULL,
    period_position                        DOUBLE       NULL,
    ecr_2_0                                DOUBLE       NULL,
    PRIMARY KEY (`date`, ticker),
    KEY ix_website_metrics_ticker_date (ticker, `date`)
);

-- SPY and BUFR closes. NULL means no price (BUFR pre-launch, 12 sessions in 2024): keep the rows.
-- A run needs both closes on its first and last session.
CREATE TABLE ecr.benchmark_prices (
    price_date  DATE PRIMARY KEY,
    spy         DOUBLE CHECK (spy > 0),
    bufr        DOUBLE CHECK (bufr > 0)
);

-- Option-cycle calendars, one row per date. The engine reads cadence = 'monthly' only.
CREATE TABLE ecr.roll_dates (
    cadence    VARCHAR(12) NOT NULL CHECK (cadence IN ('monthly', 'quarterly', 'semi_annual', 'annual')),
    roll_date  DATE        NOT NULL,
    PRIMARY KEY (cadence, roll_date)
);

-- S&P 500 close with a regime label (-1 / 0 / 1; NULL for the first weeks of 2000).
CREATE TABLE ecr.sp500_regimes (
    price_date  DATE PRIMARY KEY,
    sp500_close DOUBLE NOT NULL,
    regime      SMALLINT CHECK (regime IN (-1, 0, 1))
);

-- Accounts: the app, Studio and tests only read; loads use a separate writer account.
-- CREATE USER 'ecr_reader'@'%' IDENTIFIED BY '...';  GRANT SELECT ON ecr.* TO 'ecr_reader'@'%';
-- CREATE USER 'ecr_writer'@'%' IDENTIFIED BY '...';  GRANT SELECT, INSERT, UPDATE, DELETE ON ecr.* TO 'ecr_writer'@'%';
