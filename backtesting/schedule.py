"""Review the existing roll calendar at explicit calendar-month intervals."""
import pandas as pd

REVIEW_MONTHS = {
    'monthly': set(range(1, 13)),
    'quarterly': {3, 6, 9, 12},
    'semiannual': {6, 12},
    'annual': {12},
}


def review_dates(dates, cycles, start, stop, cadence):
    if cadence == 'daily':
        return set(dates[start:stop])
    months = REVIEW_MONTHS[cadence]
    expected = pd.period_range(dates[start], dates[stop - 1], freq='M')
    expected = {period for period in expected if period.month in months}
    selected = cycles[cycles.month.isin(months)]
    if not expected.issubset(set(selected.to_period('M'))):
        raise ValueError(f'{cadence.title()} cycle calendar does not cover the requested period')
    for cycle in selected:
        if dates[start] <= cycle <= dates[stop - 1] and cycle not in dates:
            raise ValueError(f'No observation on cycle date {cycle.date()}; check holiday calendar')
    return set(selected)
