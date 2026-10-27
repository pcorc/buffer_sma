"""Vectorized Website Metrics AM:AR. No legacy ECR scoring paths."""
import numpy as np
import pandas as pd

from config.settings import ScoreConfig

COMPONENTS = ('dbb', 'buffer_integrity', 'cap_integrity',
              'reset_proximity', 'period_position', 'ecr2')
INPUTS = ('reference_asset_return', 'remaining_outcome_days', 'remaining_cap_net',
          'remaining_buffer_net', 'downside_before_buffer_net', 'fund_cap_net', 'buffer_net')


def score_funds(frame: pd.DataFrame, config: ScoreConfig = ScoreConfig()) -> pd.DataFrame:
    """Return components and validity without mutating inputs or inventing missing scores.

    Zero remaining days is a valid formula input. Trading eligibility is separate.
    Preserve Excel AO's potentially negative cap ratio; AR applies its own MAX(0).
    """
    data = frame.loc[:, INPUTS].apply(pd.to_numeric, errors='coerce')
    valid = pd.Series(np.isfinite(data.to_numpy()).all(axis=1), index=frame.index)
    valid &= (data.fund_cap_net > 0) & (data.buffer_net > 0)
    valid &= data.remaining_outcome_days >= 0
    result = pd.DataFrame(np.nan, index=frame.index, columns=COMPONENTS)
    x = data.loc[valid]
    with np.errstate(over='ignore', invalid='ignore'):
        dbb = np.exp(config.dbb_decay * x.downside_before_buffer_net)
        buffer = (x.remaining_buffer_net / x.buffer_net).clip(0, 1)
        cap = (x.remaining_cap_net / x.fund_cap_net).clip(upper=1)
        sigma = np.where(x.reference_asset_return <= 0, config.sigma_down, config.sigma_up)
        reset = np.exp(-x.reference_asset_return**2 / (2 * sigma**2))
        period = np.exp(-(x.remaining_outcome_days - config.period_center)**2
                        / (2 * config.period_width**2))
        composite = ((config.dbb_weight * dbb + config.reset_weight * reset)
                     * buffer.clip(lower=0)**config.buffer_exponent
                     * cap.clip(lower=0)**config.cap_exponent
                     + config.period_bonus * period)
    for name, values in zip(COMPONENTS, (dbb, buffer, cap, reset, period, composite)):
        result.loc[valid, name] = values
    valid &= np.isfinite(result.to_numpy()).all(axis=1)
    result.loc[~valid, list(COMPONENTS)] = np.nan
    result['score_valid'] = valid
    return result
