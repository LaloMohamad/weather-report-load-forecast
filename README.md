# Weather Report - Load Forecast

Day-ahead electricity load (demand) forecasting pipeline for the German
(DE) market zone, built by **Team Weather Report** for the DDMO SoSe 2026
live forecasting challenge (`team_weather_report` on the
[public leaderboard](https://bartzbeielstein.github.io/challenge-leaderboard/)).

This repo exists to make the code publicly reviewable — for the OpenSSF
Scorecard, and as the home for the reproducibility ZIP releases (see
[Releases](../../releases)).

## What this is

A recursive multi-step LightGBM forecaster over aggregated German load,
built on top of [`spotforecast2`](https://github.com/sequential-parameter-optimization/spotforecast2)
and [`spotforecast2-safe`](https://github.com/sequential-parameter-optimization/spotforecast2-safe)
(both by Prof. Dr. T. Bartz-Beielstein). Weather features (temperature,
apparent temperature, degree-hours) come from 15 population-weighted German
load centers via Open-Meteo. Hyperparameters and lag selection are tuned per
submission with SpotOptim.

Full technical detail is in `MODEL_CARD.md`.

## Contents

| File | Purpose |
| --- | --- |
| `scripts/team_weather_report_submit.py` | The live daily submission pipeline |
| `scripts/team_weather_report_backtest.py` | Pseudo-live backtest harness (no submission side effects) |
| `MODEL_CARD.md` | Full model card — architecture, evaluation history, known limitations |
| `pyproject.toml` / `uv.lock` / `requirements.txt` | Pinned dependencies |

Both scripts are copies from this team's private course repository
(`bart26l-vorlesung`, hosted on RWTH's GitLab); this public repo mirrors
just the two operational scripts plus the model card, not the full lecture
book source.

## License

AGPL-3.0-or-later — matching the license of `spotforecast2-safe`, which
this code depends on and builds directly on top of.

## Data attribution

Load, generation, and price data: ENTSO-E Transparency Platform. Weather
data: Open-Meteo. No API keys or credentials are included in this repo.
