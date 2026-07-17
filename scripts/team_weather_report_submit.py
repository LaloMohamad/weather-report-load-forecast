#!/usr/bin/env python3
"""Standalone reproduction of the operational pipeline in
``lecture/14_team_4_submission.qmd`` (chapter 14, team_4 advanced submission),
runnable independently of Quarto.

It mirrors the chapter's *non-teaching* path --- download -> coverage guard ->
PACF lag selection -> ConfigEntsoe -> MultiTask(spotoptim) -> y_0 -> submission
CSV -> validate --- and optionally opens the leaderboard PR. The ~7 plotly
*teaching* visuals (endogenous / outliers / imputation / weather / rbf /
holidays / poly) and the prose availability tables are intentionally omitted;
five run-diagnostic figures (ACF, feature importance, SHAP, the forecast plot,
and team_4-vs-ENTSO-E) are saved as PNGs.

The reason this script exists: a render of chapter 14 aborted when ENTSO-E
returned transient HTTP 504/503 and ``download_new_data`` exhausted its shallow
5x/5s retry budget. This script wraps the download in an outer exponential-
backoff loop and an optional cached-data fallback so a transient outage no
longer kills the whole job.

Run it against the book's pinned venv (do NOT add a PEP-723 ``# /// script``
block --- that would build an isolated env and re-resolve spotforecast2*):

    cd /path/to/bart26l-vorlesung
    export ENTSOE_API_KEY=...
    uv run python scripts/team_weather_report_submit.py         # full live run
    uv run python scripts/team_weather_report_submit.py --help  # all flags

SYNC NOTE --- this file duplicates chapter-14 logic. When ch14 changes any of:
the env homes (`team4-imports`), the date constants (`team4-constants`), the
coverage guards (`team4-interim`/`team4-cutoff`, incl. the frontier
completeness guard and the value-sanity QC), the PACF rule (`fig-team4-acf`), the forecaster factory
(`team4-factory`), the ConfigEntsoe arguments (`team4-config`), the SpotOptim
search space (`team4-search-space`), the pipeline method order
(`team4-prepare`...`team4-train`), the shape plausibility check
(`team4-shape-check`), or the submission schema
(`team4-submission-write`), re-sync this script. The .qmd is the source of truth.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Headless backend MUST be selected before anything (incl. shap) imports pyplot.
import matplotlib

matplotlib.use("Agg")

import pandas as pd

# --- defaults pinned to the chapter-14 values (see SYNC NOTE) -----------------
COUNTRY = "DE"
# Team identity follows the feature set (qmd team4-submission-write): with
# ENTSO-E's day-ahead Forecasted Load as a model input, submit as
# team_4_entsoe; without it, as team_4. --team-id still overrides.
# Since the chapter split (2026-06-06) 14_team_4_submission.qmd is the PLAIN
# team_4 variant (both flags False); the ENTSO-E-prior variant lives in
# The ENTSO-E load forecast remains disabled for the regular team submission.
INCLUDE_ENTSOE_FORECAST_LOAD = False
INCLUDE_ENTSOE_NET_LOAD = False          # derived from Forecasted Load -> off together with it
DEFAULT_TEAM_ID = "team_weather_report"
START_DOWNLOAD = "202201010000"
DATA_SUBDIR = "ddmo_ch14_team4"             # under ~/spotforecast2_data/ (= chapter `team4-imports`)
CACHE_SUBDIR = "ddmo_ch14_team4"            # under ~/.spotforecast2_cache/
OUTLIER_IQR_K = 5
MAX_ACTUAL_LAG_HOURS = 36                # team4-interim guard (qmd line 252)
GAP_SCAN_DAYS = 28                       # team4-interim interior-gap guard
MAX_ACTUAL_GAP_HOURS = 12                #   (third check; added 2026-06-04)
MAX_INTRAHOUR_RANGE_MW = 8_000           # team4-cutoff value-sanity QC: clean-data
MAX_ADJ_STEP_MW = 6_000                  #   q0.9999 is 7.3/4.1 GW (added 2026-06-05)
QC_WINDOW_DAYS = 3                       #   scan scope for the value-sanity QC
MAX_DEVIATION_MW = 11_000                # deviation rule (sf2-safe >= 18.1.0): dropout vs
DEVIATION_REF = "Forecasted Load"        #   day-ahead forecast; calibration in diagnosis/
DEVIATION_SLOTS = 1                      #   experiments/e9: 1 clean slot < -11 GW in 2.4 y,
                                         #   06-07 frontier dropout is single-slot -> slots=1
TARGET_CORRUPTION_POLICY = "truncate"    # chapter default since 2026-06-05; "abort" = conservative
LAG_FALLBACK = [1, 2, 24, 168]
TRAIN_YEARS = 3  # matches the documented cross-team download-window fairness invariant (CLAUDE.md)
PREDICT_SIZE = 24
REFIT_SIZE = 7
NUMBER_FOLDS = 10
IMPUTATION_WINDOW_SIZE = 24
N_TRIALS_SPOTOPTIM = 200
N_INITIAL_SPOTOPTIM = 12
N_TRIALS_OPTUNA = 15

logger = logging.getLogger("team4_submit")


class Abort(Exception):
    """Raise to stop with a specific process exit code (see exit-code table)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Dates:
    now: pd.Timestamp
    today: pd.Timestamp
    yesterday: pd.Timestamp
    tomorrow: pd.Timestamp
    last_target: pd.Timestamp
    start_dl: str
    end_dl: str


@dataclass
class Coverage:
    last_full_hour: pd.Timestamp
    first_pred: pd.Timestamp
    live_n_steps: int


# --- CLI + logging ------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="team_weather_report_submit",
        description="Team Weather Report live forecast and submission pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--team-id", default=DEFAULT_TEAM_ID,
                   help="registry check + submission directory")
    p.add_argument("--skip-download", action="store_true",
                   help="reuse the cached interim CSV; no ENTSO-E API call")
    p.add_argument("--max-retries", type=int, default=5,
                   help="outer download attempts (each wraps the library's 5x loop)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-socket read timeout (s) for ENTSO-E downloads; "
                        "0 disables (sf2-safe >= 16.2.0)")
    p.add_argument("--backoff", type=float, default=5.0,
                   help="base seconds for exponential backoff between outer attempts")
    p.add_argument("--allow-stale", action="store_true",
                   help="permit cache fallback when the API is down "
                        "(still gated by the D-1 23:00 freshness check)")
    p.add_argument("--data-end", default=None,
                   help="override END_DOWNLOAD as YYYYMMDDHHMM (freezes the window)")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the leaderboard validator subprocess")
    p.add_argument("--push", action="store_true",
                   help="branch + commit + open the auto-merging PR via gh")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_SPOTOPTIM,
                   help="n_trials_spotoptim (lower for a fast smoke run)")
    p.add_argument("--n-initial", type=int, default=N_INITIAL_SPOTOPTIM,
                   help="n_initial_spotoptim")
    p.add_argument("--max-time", type=float, default=None,
                   help="wall-clock budget for the SpotOptim search in "
                        "MINUTES (max_time_spotoptim, sf2-safe >= 21.2.0; "
                        "default: no limit -- the search runs all n_trials)")
    p.add_argument("--train-years", type=int, default=TRAIN_YEARS,
                   help="training window length in years")
    p.add_argument("--tensorboard", action="store_true",
                   help="stream the SpotOptim tuning run to TensorBoard "
                        "(sf2 >= 7.0.0 / spotoptim >= 1.0; watch with "
                        "'tensorboard --logdir <data_home>/tensorboard')")
    p.add_argument("--tensorboard-path", default=None,
                   help="TensorBoard event-file directory "
                        "(default: <data_home>/tensorboard/<target-date>)")
    p.add_argument("--figures-dir", default=None,
                   help="where to save diagnostic PNGs "
                        "(default: <data_home>/figures/<target-date>/)")
    p.add_argument("--no-figures", action="store_true",
                   help="skip all diagnostic figures")
    p.add_argument("--prediction-band", action="store_true",
                   help="also fit q10/q50/q90 LightGBM quantile heads "
                        "(spotforecast2-safe >= 19.4.0) and write a "
                        "prediction-band CSV diagnostic (point submission "
                        "unchanged)")
    p.add_argument("--leaderboard-root",
                   default=str(Path.home() / "workspace" / "challenge-leaderboard"),
                   help="sibling challenge-leaderboard clone")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: show downloader INFO; -vv: DEBUG everywhere")
    return p.parse_args(argv)


def setup_logging(verbosity: int) -> None:
    # Keep the packages quiet by default (like the chapter's basicConfig WARNING)
    # but give our own logger a dedicated handler so its INFO always shows.
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [team4_submit] %(levelname)s %(message)s",
                                           "%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbosity >= 2 else logging.INFO)
    logger.propagate = False
    if verbosity >= 1:
        logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.INFO)
    if verbosity >= 2:
        for name in ("spotforecast2", "spotforecast2_safe"):
            logging.getLogger(name).setLevel(logging.DEBUG)


# --- environment, registry, dates ---------------------------------------------
def configure_environment() -> None:
    """Mirror the qmd ``team4-imports`` cell: fork start method + isolated homes.

    Must run before any spotforecast2* import / MultiTask / download call;
    get_data_home()/get_cache_home() read these env vars at call time.
    """
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    import multiprocessing as mp

    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        logger.warning("could not set multiprocessing start method to 'fork'; "
                       "joblib-backed backtesting may be affected on this platform.")
    os.environ["SPOTFORECAST2_DATA"] = str(Path.home() / "spotforecast2_data" / DATA_SUBDIR)
    os.environ["SPOTFORECAST2_CACHE"] = str(Path.home() / ".spotforecast2_cache" / CACHE_SUBDIR)


def repair_weather_cache_gaps() -> None:
    """Forward/back-fill NaN in cached per-city weather parquet files.

    ``spotforecast2_safe``'s WeatherService rejects a cached row if *any*
    fetched column has NaN, not just the ones this pipeline actually reads
    (temperature-derived features only). Open-Meteo's historical archive can
    have narrow, column-specific coverage gaps for a given city (seen:
    ``wind_speed_10m``/``wind_direction_10m`` missing for ~154 days in one
    city's 2022 data, unrelated to temperature) that sit silently in the
    cache until a run's requested date range happens to be fully covered by
    the cache alone — at which point WeatherService takes its strict
    direct-read path and raises, blocking the whole run.

    This runs before any weather fetch so a latent gap in an unused column
    never blocks a submission. Every fill is logged with the affected
    columns and row count so an unexpected gap in a column that *does*
    matter (e.g. ``temperature_2m``) stays visible rather than silently
    papered over.
    """
    import pandas as pd

    cache_home = os.environ.get("SPOTFORECAST2_WEATHER_CACHE") or os.environ.get("SPOTFORECAST2_CACHE")
    if not cache_home:
        return
    for f in sorted(Path(cache_home).glob("weather_cache_*.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("repair_weather_cache_gaps: skipping unreadable %s (%s)", f.name, exc)
            continue
        nan_counts = df.isna().sum()
        bad = nan_counts[nan_counts > 0]
        if bad.empty:
            continue
        df = df.ffill().bfill()
        df.to_parquet(f)
        logger.warning(
            "repair_weather_cache_gaps: filled NaN in %s -> %s",
            f.name, dict(bad),
        )


def check_team_registry(team_id: str, lb_root: Path) -> None:
    """qmd ``team4-registry`` (lines 54-66): fail-loud if team_id is unknown."""
    import yaml

    teams_yml = lb_root / "teams.yml"
    if not teams_yml.exists():
        raise Abort(3, f"teams.yml not found at {teams_yml}; "
                       f"is challenge-leaderboard cloned at {lb_root}?")
    teams = yaml.safe_load(teams_yml.read_text())["teams"]
    team = next((t for t in teams if t["id"] == team_id), None)
    if team is None:
        raise Abort(3, f"{team_id} is not registered in {teams_yml}")
    logger.info("team %s : %s | github: %s", team_id,
                team.get("display_name", "?"),
                ", ".join(team.get("github_handles", [])))


def compute_dates(now: pd.Timestamp, data_end: str | None) -> Dates:
    """qmd ``team4-constants`` (lines 146-167). `now` is captured once upstream."""
    today = now.normalize()
    yesterday = today - pd.Timedelta(days=1)
    tomorrow = today + pd.Timedelta(days=1)
    last_target = tomorrow + pd.Timedelta(hours=23)
    if data_end:
        try:
            pd.to_datetime(data_end, format="%Y%m%d%H%M", utc=True)
        except Exception as exc:  # noqa: BLE001
            raise Abort(1, f"--data-end {data_end!r} is not YYYYMMDDHHMM: {exc}")
        end_dl = data_end
    else:
        end_dl = (tomorrow + pd.Timedelta(days=1)).strftime("%Y%m%d%H%M")
    return Dates(now, today, yesterday, tomorrow, last_target, START_DOWNLOAD, end_dl)


# --- download (robust) + interim cache ----------------------------------------
def robust_download(api_key: str, dates: Dates, *, max_retries: int,
                    backoff: float, force: bool = True,
                    timeout: float | None = 60.0) -> bool:
    """qmd ``team4-download`` (lines 172-185), wrapped in an outer exponential-
    backoff loop. Returns True on success, False after exhausting attempts.

    The venv's download_new_data already retries 5x/5s internally and raises
    RuntimeError on persistent HTTP 504/503; this outer loop multiplies that
    resilience without editing shipped safety code.
    """
    from spotforecast2_safe.downloader.entsoe import download_new_data

    for attempt in range(1, max_retries + 1):
        try:
            download_new_data(
                api_key=api_key,
                country_code=COUNTRY,
                start=dates.start_dl,
                end=dates.end_dl,
                force=force,
                keep_forecast_future=True,
                timeout=timeout,
            )
            logger.info("ENTSO-E download succeeded on attempt %d/%d", attempt, max_retries)
            return True
        except Exception as exc:  # noqa: BLE001 -- download_new_data raises RuntimeError
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("ENTSO-E download attempt %d/%d failed: %s",
                           attempt, max_retries, exc)
            if attempt < max_retries:
                logger.warning("retrying in %.0fs (exponential backoff) ...", wait)
                time.sleep(wait)
    logger.error("ENTSO-E download failed after %d outer attempts.", max_retries)
    return False


def download_side_tables(api_key: str, dates: Dates,
                         timeout: float | None = 60.0) -> None:
    """qmd renewable/price cells (``team4-renewable-inspect`` / ``team4-price-inspect``
    download companions): refresh the two provider side-tables.

    Failures are non-fatal by design: the providers behind the ``include_*``
    flags degrade via ``on_exog_provider_failure="skip"``, so a transient
    side-table outage costs features, not the submission.
    """
    from spotforecast2_safe.downloader.entsoe import (
        download_day_ahead_price,
        download_renewable_forecast,
    )

    for name, fn, country in (
        ("renewable forecast", download_renewable_forecast, COUNTRY),
        ("day-ahead price", download_day_ahead_price, "DE_LU"),
    ):
        try:
            fn(api_key=api_key, country_code=country,
               start=dates.start_dl, end=dates.end_dl, force=False,
               timeout=timeout)
            logger.info("side-table download ok: %s", name)
        except Exception as exc:  # noqa: BLE001 -- provider skip handles the gap
            logger.warning("side-table download failed (%s): %s -- the matching "
                           "provider will be skipped.", name, exc)


def load_interim() -> pd.DataFrame:
    """qmd ``team4-interim`` read (lines 236-238)."""
    from spotforecast2_safe.data.fetch_data import get_data_home

    interim_csv = get_data_home() / "interim" / "energy_load.csv"
    if not interim_csv.exists():
        raise Abort(2, f"no interim cache at {interim_csv}; cannot proceed "
                       f"without a successful download.")
    interim = pd.read_csv(interim_csv, index_col=0, parse_dates=True)
    interim.index = pd.to_datetime(interim.index, utc=True)
    logger.info("interim CSV: %s  (%d rows, %s -> %s)",
                interim_csv, len(interim), interim.index.min(), interim.index.max())
    return interim


def assert_coverage(interim: pd.DataFrame, dates: Dates, *, fallback: bool) -> Coverage:
    """qmd coverage guards (lines 240-262) + cutoff (lines 322-325).

    `fallback=True` (cache reused because the download was skipped/failed) adds a
    stricter gate: the published actuals must reach at least *yesterday 23:00 UTC*
    --- a full most-recent complete UTC day --- before we build a submission off
    cached data. (The plan's "D-1 23:00" phrasing resolves to yesterday 23:00 for
    a target day of *tomorrow*; today 23:00 is in the future and unpublishable.)
    """
    required_last = dates.today - pd.Timedelta(hours=1)
    if interim.index.max() < required_last:
        raise Abort(1, f"ENTSO-E coverage is stale: last interim row "
                       f"{interim.index.max()} < required {required_last}.")

    actual = interim["Actual Load"].dropna()
    if actual.empty:
        raise Abort(1, "interim CSV has no published Actual Load values.")
    last_actual = actual.index.max()
    if last_actual < dates.now - pd.Timedelta(hours=MAX_ACTUAL_LAG_HOURS):
        lag_h = int((dates.now - last_actual) / pd.Timedelta(hours=1))
        raise Abort(1, f"Actual Load is stale: last published {last_actual}, "
                       f"{lag_h} h before now; tolerance {MAX_ACTUAL_LAG_HOURS} h.")

    # Interior-gap guard (qmd team4-interim, third check): a publication outage
    # can hole the *middle* of the feed while both edge checks pass (observed
    # 2026-06-02 -- a full day of actuals published more than a day late).
    recent = actual.loc[dates.now - pd.Timedelta(days=GAP_SCAN_DAYS):]
    gaps = recent.index.to_series().diff()
    oversized = gaps[gaps > pd.Timedelta(hours=MAX_ACTUAL_GAP_HOURS)]
    if not oversized.empty:
        worst_end = oversized.idxmax()
        raise Abort(1, f"Actual Load has {len(oversized)} interior gap(s) wider "
                       f"than {MAX_ACTUAL_GAP_HOURS} h in the last "
                       f"{GAP_SCAN_DAYS} days (worst: "
                       f"{worst_end - oversized.max()} -> {worst_end}). "
                       f"Re-download once the late actuals are published.")

    # Frontier completeness guard (qmd team4-cutoff): only an hour with all of
    # its quarter-hour samples published may anchor the recursion -- a partial
    # frontier hour averages to an anomalous level and drags the whole first
    # forecast day (observed on the 2026-06-05 forecast). The expected count
    # per hour derives from the feed's own cadence (15 min for DE -> 4).
    cadence = actual.index.to_series().diff().mode().iloc[0]
    samples_per_hour = int(pd.Timedelta(hours=1) / cadence)
    samples_by_hour = actual.resample("h").count()
    last_full_hour = samples_by_hour[samples_by_hour >= samples_per_hour].index.max()
    if last_full_hour < last_actual.floor("h"):
        logger.warning("frontier hour %s is incomplete (<%d samples); end_train "
                       "stepped back to %s", last_actual.floor("h"),
                       samples_per_hour, last_full_hour)

    # Value-sanity tripwire (qmd team4-cutoff; 2026-06-03/04 incident): ENTSO-E
    # published complete but physically impossible quarter-hour actuals
    # (adjacent slots oscillating by +-10 GW, intra-hour ranges up to 15 GW,
    # identical in every raw vintage). Such values pass every staleness and
    # completeness check, then poison the recursion anchor and the seasonal
    # lags. Thresholds are the clean-data q0.9999 rounded up; the late-March
    # DST week can reach ~12.6 GW ranges and would need a temporary exemption.
    # Since sf2-safe 16.4.0 the rules live in the library (same detector as
    # the chapter cell); the call below is a PREVIEW -- the authoritative
    # policy run happens inside prepare_data via the target_qc_* knobs.
    # Default policy "truncate" (since 2026-06-05): training retracts to the
    # last sound hour and the recursion bridges the gap on published
    # day-ahead exog. Gate A of the incident forensics: ENTSO-E never
    # corrects this corruption class retroactively, so aborting buys no
    # better data later; "abort" remains the conservative alternative.
    from spotforecast2_safe.exceptions import TargetCorruptionError
    from spotforecast2_safe.preprocessing import apply_target_corruption_policy

    qc_recent = interim["Actual Load"].loc[
        actual.index.max() - pd.Timedelta(days=QC_WINDOW_DAYS):
    ]
    qc_hourly = qc_recent.resample("h")
    qc_range = (qc_hourly.max() - qc_hourly.min()).dropna()
    qc_step = qc_recent.diff().abs().dropna()
    # Deviation diagnostics (third rule, sf2-safe >= 18.1.0): dropout vs the
    # published day-ahead forecast — catches what the dynamics rules cannot
    # see (e9 calibration). NaN-safe by construction: the dropna() keeps the
    # publication-lag tail out of the minimum.
    qc_dev = (
        interim["Actual Load"] - interim[DEVIATION_REF]
    ).loc[qc_recent.index.min():].dropna()
    logger.info("value-sanity QC (last %d d): max intra-hour range %.0f MW "
                "(limit %d), max adjacent step %.0f MW (limit %d), "
                "min deviation vs forecast %.0f MW (limit -%d)",
                QC_WINDOW_DAYS, qc_range.max(), MAX_INTRAHOUR_RANGE_MW,
                qc_step.max(), MAX_ADJ_STEP_MW,
                qc_dev.min(), MAX_DEVIATION_MW)
    try:
        _, qc_report = apply_target_corruption_policy(
            interim[["Actual Load", DEVIATION_REF]],
            targets=["Actual Load"],
            policy=TARGET_CORRUPTION_POLICY,
            range_mw=MAX_INTRAHOUR_RANGE_MW,
            step_mw=MAX_ADJ_STEP_MW,
            window_days=QC_WINDOW_DAYS,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=logger,
            deviation_mw=MAX_DEVIATION_MW,
            deviation_ref=DEVIATION_REF,
            deviation_slots=DEVIATION_SLOTS,
        )
    except TargetCorruptionError as exc:
        raise Abort(1, str(exc))
    if qc_report.fired:
        last_sound = qc_report.first_flagged_hour - pd.Timedelta(hours=1)
        logger.warning("value-sanity QC: %d corrupt hour(s) in %d span(s) -> "
                       "policy %r; prepare_data will retract data_end to %s "
                       "and auto-extend predict_size",
                       qc_report.n_flagged_hours, len(qc_report.spans),
                       qc_report.action, last_sound)

    if fallback:
        gate = dates.yesterday + pd.Timedelta(hours=23)  # yesterday 23:00 UTC
        if last_full_hour < gate:
            raise Abort(2, f"cached actuals end {last_full_hour} < {gate} "
                           f"(yesterday 23:00 UTC); refusing to submit a stale "
                           f"forecast. Re-run when ENTSO-E is reachable.")

    first_pred = last_full_hour + pd.Timedelta(hours=1)
    live_n_steps = int((dates.last_target - first_pred).total_seconds() // 3600) + 1
    logger.info("end_train=%s  first_pred=%s  last_target=%s  predict_size=%d",
                last_full_hour, first_pred, dates.last_target, live_n_steps)
    return Coverage(last_full_hour, first_pred, live_n_steps)


def entsoe_predictions(interim: pd.DataFrame) -> pd.Series:
    """qmd ``team4-entsoe-predictions`` (lines 283-284): ENTSO-E's own day-ahead
    forecast on the hourly grid (baseline only; never trained on)."""
    preds = interim["Forecasted Load"].resample("h").mean().dropna()
    preds.name = "Forecasted Load"
    return preds


# --- PACF lag selection (logic + optional plot) -------------------------------
def select_key_lags(interim: pd.DataFrame, last_full_hour: pd.Timestamp,
                    figdir: Path | None) -> list[int]:
    """qmd ``fig-team4-acf`` (lines 518-537), plot stripped out (optional save)."""
    import numpy as np
    from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation

    acf_series = interim.loc[:last_full_hour, "Actual Load"].resample("h").mean().dropna()
    acf = calculate_lag_autocorrelation(acf_series, n_lags=200)
    conf = 1.96 / np.sqrt(len(acf_series))
    significant = acf[acf["partial_autocorrelation_abs"] > conf]
    key_lags = sorted(significant.nlargest(8, "partial_autocorrelation_abs")["lag"].astype(int))
    if not key_lags:  # degenerate fallback (very short series)
        key_lags = list(LAG_FALLBACK)
    logger.info("PACF key_lags (top %d by |PACF|): %s  [N=%d, band=+/-%.4f]",
                len(key_lags), key_lags, len(acf_series), conf)
    if figdir is not None:
        _try(lambda: _plot_acf(acf, key_lags, conf, figdir), "acf")
    return key_lags


# --- config + pipeline --------------------------------------------------------
def team4_lgbm_factory(config, *, weight_func=None, target=None):
    """qmd ``team4-factory``: LightGBM with anchored level windows (>= 72 h).

    Default L2 objective; window features deliberately exclude the 24-h scale.
    A first version (regression_l1 + 24-h mean/min/max) flattened the live
    forecast for 2026-06-05: short windows turn into pure self-feedback over
    the live recursion, and L1 + reg_alpha soft-thresholding removed the
    incentive to track the daily amplitude. The CV cannot detect this (each
    fold restarts from observed history), hence the structural fix here.
    """
    from lightgbm import LGBMRegressor
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    from spotforecast2_safe.preprocessing import RollingFeatures

    del target
    q_alpha = getattr(config, "quantile_alpha", None)
    estimator = (
        LGBMRegressor(objective="quantile", alpha=q_alpha,
                      random_state=config.random_state, verbose=-1)
        if q_alpha is not None
        else LGBMRegressor(random_state=config.random_state, verbose=-1)
    )
    return ForecasterRecursive(
        estimator=estimator,
        lags=config.lags_consider[-1],
        window_features=[  # one instance per feature: keeps generated names unique
            RollingFeatures(stats="mean", window_sizes=config.window_size),  # 72 h, as stock
            RollingFeatures(stats="mean", window_sizes=24 * 7),
            RollingFeatures(stats="mean", window_sizes=24 * 30),
        ],
        weight_func=weight_func,
    )


def build_search_space(key_lags, wide_bounds: bool = False, richer_lags: bool = False):
    """qmd ``team4-search-space`` + ``@sec-team4-boundary-management``:
    weekly-anchored SpotOptim lag pool over the chapter's *widened* ranges.

    Boundary management (qmd @sec-team4-boundary-management): a deep ENTSO-E
    search pinned ``reg_alpha`` at its old 10.0 ceiling (98.9 % of the range),
    so the regularisation caps were raised to 100.0 and switched to log10 (float
    dims, safe from the n_estimators int+log10 collapse noted below);
    ``max_depth`` was widened 32 -> 48 (``num_leaves`` still binds capacity) and
    ``learning_rate``'s lower bound refocused 1e-4 -> 5e-3 (the bottom decades
    never pay off). Every lag candidate carries the weekly anchor 167/168 (lags
    beyond the live horizon keep reading observed history during the recursion --
    the anchor-free stock candidate ``[1, 2, 24, 48]`` won the flat 2026-06-05
    run). ``warm_start_lags=key_lags`` still prepends ``str(key_lags)``
    (list-valued seed since sf2-safe 22.0.0).

    ``wide_bounds`` (default False, live behaviour unchanged): season_01 Exp08
    candidate widening num_leaves [8,1024]->[4,1024] and max_depth [3,48]->[2,48]
    -- SpotOptim keeps landing near the current lower bounds on both (~98,
    ~5), the same "cage wall" signature that motivated the O5 bagging_fraction/
    reg_lambda fix. Opt-in only, backtest-gated (`--wide-bounds` on
    team_weather_report_backtest.py) -- do not flip this default to True until
    a backtest actually shows improvement over the 601.6 MW `rich` baseline,
    per this project's own stated rule (see roadmap.md "Current Production
    Default"). Threading it as a parameter here rather than editing the
    live values directly keeps a backtest fully isolated from whatever
    tonight's scheduled submission is doing, which defaults to False and
    never sees this at all unless explicitly asked for.

    ``richer_lags`` (default False, live behaviour unchanged): season_01 Exp09
    candidate. 48h, 167/168h (weekly), and 336h (2-week) are already covered by
    at least one existing candidate list below -- the one genuine gap is 72h
    (3-day anchor), present in none of the 6 stock candidates. Adds exactly one
    new candidate list with 72h included alongside the existing anchors, rather
    than rewriting the whole pool -- keeps this a narrow, single-variable test
    (does adding a 72h anchor help) instead of a confounded rewrite of
    everything at once. Same opt-in/backtest-gated pattern as ``wide_bounds``.
    Rolling-average *features* (24h/48h/168h means, as opposed to lag
    *candidates*) are a separate, not-yet-implemented sub-step of Exp09 --
    the existing ``RollingFeatures`` in ``team4_lgbm_factory`` only has one
    72h window today; adding more is future work, not part of this change.
    """
    num_leaves_lower = 4 if wide_bounds else 8
    max_depth_lower = 2 if wide_bounds else 3
    lag_candidates = [
        "[1, 2, 3, 11, 12, 22, 23, 24, 47, 48, 167, 168]",   # stock 12-lag
        "[1, 2, 11, 12, 23, 24, 167, 168]",                   # stock 8-lag
        "[1, 2, 24, 48, 167, 168]",                           # compact + weekly anchor
        "[1, 2, 23, 24, 47, 48, 167, 168]",                   # cycle neighbours + weekly
        str(sorted(set(key_lags) | {1, 2, 24, 48, 168})),     # PACF picks + canonical
        "[1, 2, 3, 23, 24, 25, 47, 48, 167, 168, 169, 336]",  # extended + 2-week lag
    ]
    if richer_lags:
        lag_candidates.append(
            "[1, 2, 24, 48, 72, 167, 168, 169, 336]"  # Exp09: + 3-day anchor
        )
    return {
        "estimator__num_leaves": (num_leaves_lower, 1024),
        # Widened 32 -> 48 (qmd @sec-team4-boundary-management): the deep ENTSO-E
        # optimum placed max_depth at 26 (79 % of the old 3-32 range); num_leaves
        # is the binding complexity control, so this only lifts a mild cap.
        "estimator__max_depth": (max_depth_lower, 48),
        # Lower bound refocused 1e-4 -> 5e-3: the optimum (~0.069) never visits the
        # bottom two decades, so reclaim those trials for the region that pays off.
        "estimator__learning_rate": (0.005, 0.3, "log10"),
        # Linear integer range, deliberately NOT (10, 5000, "log10"): SpotOptim
        # < 0.12.7 int-cast log-transformed bounds, so an int+log10 dimension
        # collapsed to the decade exponents {10, 100, 1000, 10000} — the last
        # of which silently EXCEEDED the declared cap (observed 2026-06-05;
        # fixed upstream in spotoptim 0.12.7). The linear range stays:
        # transparent and optimizer-version independent.
        "estimator__n_estimators": (100, 5000),
        # Widened 0.5 -> 0.3 (O5 boundary fix 2026-06-27): every t5/i3 run landed
        # at pos=0.02 against the old 0.5 lower bound — optimizer consistently
        # wanted less bagging than allowed.
        "estimator__bagging_fraction": (0.3, 1.0),
        "estimator__feature_fraction": (0.5, 1.0),
        # Widened 10 -> 100 AND switched to log10 (qmd @sec-team4-boundary-
        # management): the deep search pinned reg_alpha at the old 10.0 ceiling
        # (98.9 %) -- the optimiser wanted more L1 than allowed. log10 is safe
        # here (float dims) and spends trials evenly across the decades these
        # regularisers span; reg_lambda lower bound widened 0.001 -> 0.0001
        # (O5 boundary fix 2026-06-27): every run landed at pos=0.02 against 0.001.
        "estimator__reg_alpha": (0.001, 100.0, "log10"),
        "estimator__reg_lambda": (0.0001, 100.0, "log10"),
        "lags": lag_candidates,
    }


def build_config(key_lags, cov: Coverage, *, n_trials, n_initial, train_years, max_time=None):
    """qmd ``team4-config``."""
    from spotforecast2_safe.data import Period
    from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
    from spotforecast2_safe.data.entsoe_loader import (
        entsoe_data_loader,
        entsoe_test_data_loader,
    )

    periods = [
        Period(name="daily",     n_periods=12, column="hour",      input_range=(1, 24)),
        Period(name="weekly",    n_periods=7,  column="dayofweek", input_range=(0, 6)),
        Period(name="monthly",   n_periods=12, column="month",     input_range=(1, 12)),
        Period(name="quarterly", n_periods=4,  column="quarter",   input_range=(1, 4)),
        Period(name="yearly",    n_periods=12, column="dayofyear", input_range=(1, 365)),
    ]
    cfg = ConfigEntsoe(
        country_code=COUNTRY,
        data_filename="interim/energy_load.csv",
        targets=["Actual Load"],
        agg_weights=[1.0],
        bounds=None,
        data_loader=entsoe_data_loader,
        test_data_loader=entsoe_test_data_loader,
        forecaster_factory=team4_lgbm_factory,
        periods=periods,
        lags_consider=key_lags,
        train_size=pd.Timedelta(days=365 * train_years),
        end_train_default=cov.last_full_hour.isoformat(),
        delta_val=pd.Timedelta(hours=PREDICT_SIZE * REFIT_SIZE * NUMBER_FOLDS),
        predict_size=cov.live_n_steps,
        cv_block_size=PREDICT_SIZE,
        refit_size=REFIT_SIZE,
        number_folds=NUMBER_FOLDS,
        imputation_window_size=IMPUTATION_WINDOW_SIZE,
        n_trials_optuna=N_TRIALS_OPTUNA,
        n_trials_spotoptim=n_trials,
        n_initial_spotoptim=n_initial,
        max_time_spotoptim=max_time,
        # Seed the SpotOptim search with the dynamic PACF lags (sf2-safe >=
        # 22.0.0: the field carries the seed list itself; the package default
        # is the static DEFAULT_WARM_START_LAGS).
        warm_start_lags=list(key_lags),
        include_weather_windows=True,
        include_degree_hours=True,               # hdh/cdh: splits U-shaped temp-load curve so LightGBM need not rediscover the 15°C kink
        include_apparent_temperature=True,        # apparent_temperature + dew_point: humidity effect on perceived heat (AC demand)
        use_population_weighted_weather=True,     # 15 German load centres weighted by population (Zimmermann & Ziel 2025)
        include_holiday_features=True,
        include_holiday_adjacency_features=True,  # Brückentag + day before/after holiday (sf2-safe >= 15.9.0)
        include_ephemeris_features=True,          # continuous solar geometry (sf2-safe >= 19.3.0; ConfigEntsoe parity >= 19.4.0)
        include_day_type_features=True,           # is_workday + day_type class (sf2-safe >= 19.3.0; ConfigEntsoe parity >= 19.4.0)
        poly_features_degree=2,
        max_poly_features=40,
        state="NW",
        random_state=42,
        on_weather_failure="skip",
        # Provider-based day-ahead/static priors (chapter `team4-config`, sf2-safe >= 16.1.0).
        # The forecast-load flag also decides DEFAULT_TEAM_ID (module constants).
        include_entsoe_forecast_load=INCLUDE_ENTSOE_FORECAST_LOAD,
        include_entsoe_renewable_forecast=True,
        include_entsoe_net_load=INCLUDE_ENTSOE_NET_LOAD,
        include_entsoe_day_ahead_price=True,
        include_covid_infection_rate=True,
        on_exog_provider_failure="skip",
        exog_max_gap_hours=3,
        exog_max_tail_gap_hours=48,
        exog_provider_window="train",
        # Target-side corruption policy (sf2-safe >= 16.4.0): same rules as
        # the assert_coverage value-sanity QC, applied authoritatively at
        # 15-min cadence inside prepare_data (qmd team4-config). Default
        # "truncate" (since 2026-06-05): retract data_end to the last sound
        # hour + auto-extend predict_size; "abort" = conservative alternative.
        target_qc_range_mw=MAX_INTRAHOUR_RANGE_MW,
        target_qc_step_mw=MAX_ADJ_STEP_MW,
        target_qc_window_days=QC_WINDOW_DAYS,
        target_corruption_policy=TARGET_CORRUPTION_POLICY,
        # Deviation rule (sf2-safe >= 18.1.0): catches dropouts that stay
        # below the dynamics thresholds (2026-06-07 frontier: 5.6 GW steps
        # at Actual - Forecast = -11.6 GW). Calibration: e9.
        target_qc_deviation_mw=MAX_DEVIATION_MW,
        target_qc_deviation_ref=DEVIATION_REF,
        target_qc_deviation_slots=DEVIATION_SLOTS,
    )
    cfg.data_frame_name = "bart26k-live-team4"
    logger.info("config ready: predict_size=%s n_trials_spotoptim=%s "
                "number_folds=%s", cfg.predict_size, cfg.n_trials_spotoptim,
                cfg.number_folds)
    return cfg


def report_boundary_positions(mt, search_space, *, warn_frac=0.10):
    """qmd ``team4-boundary-diagnostics`` (@sec-team4-boundary-diagnostics):
    log where this run's tuned optimum sits inside each numeric search-space
    interval and warn if it presses a bound.

    Boundary management (qmd @sec-team4-boundary-management): an optimum pinned
    to a bound means the optimiser wanted to go further -- widen that side and
    re-run; a value deep inside a very wide interval means trials were wasted --
    tighten it. Position is read in the dimension's own scale (log for log10
    dims). Best-effort and log-only -- never blocks the submission.
    """
    import math

    try:
        est = mt.results["spotoptim"]["Actual Load"]["forecaster"].estimator
        raw = est.get_params()
    except Exception as exc:  # noqa: BLE001
        logger.warning("boundary diagnostics skipped (cannot read tuned params): %s", exc)
        return

    flagged = []
    for name, spec in search_space.items():
        if not (isinstance(spec, tuple) and len(spec) in (2, 3)):
            continue  # categorical dim (e.g. "lags")
        key = name.split("estimator__", 1)[-1]
        val = raw.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        low, high = float(spec[0]), float(spec[1])
        log = len(spec) == 3 and spec[2] == "log10"
        if log and (val <= 0 or low <= 0):
            continue
        pos = ((math.log10(val) - math.log10(low)) / (math.log10(high) - math.log10(low))
               if log else (float(val) - low) / (high - low))
        flag = "> upper" if pos > 1 - warn_frac else ("< lower" if pos < warn_frac else "")
        logger.info("  bound %-18s = %-11.5g in [%g, %g]%s  pos=%.2f%s",
                    key, val, low, high, " log10" if log else "", pos,
                    f"  <-- {flag}" if flag else "")
        if flag:
            flagged.append(f"{key} {flag}")
    if flagged:
        logger.warning("boundary check: %d tuned dim(s) near a bound (%s) -- consider "
                       "widening that side and re-running "
                       "(qmd @sec-team4-boundary-management).",
                       len(flagged), ", ".join(flagged))
    else:
        logger.info("boundary check: all tuned dimensions sit interior.")


def _save_weather_snapshot(mt, dates: Dates) -> None:
    """Persist the weather forecast used for this prediction as a dated parquet file.

    Open-Meteo's forecast endpoint overwrites its own data with reanalysis values
    after ~5 days, so the weather signal seen at prediction time is gone on any
    later re-run. This snapshot freezes exactly what the model saw.
    Stored at _cache/weather_snapshots/<target_date>.parquet (gitignored).
    """
    try:
        wa = getattr(mt, "weather_aligned", None)
        if wa is None or wa.empty:
            logger.warning("weather snapshot skipped: weather_aligned not available")
            return
        from spotforecast2_safe.data.fetch_data import get_data_home
        snap_dir = get_data_home() / "weather_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{dates.tomorrow.date()}.parquet"
        wa.to_parquet(snap_path)
        logger.info("weather snapshot saved -> %s  (%d rows, %d cols)",
                    snap_path, len(wa), wa.shape[1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("weather snapshot failed (non-fatal): %s", exc)


def _save_reproducibility_package(
    mt, dates: Dates, cfg, submission_path: Path
) -> None:
    """Bundle everything a third party needs to reproduce this one submission.

    A weather snapshot alone isn't enough -- the trained model, the exact
    hyperparameters/lags SpotOptim selected, the raw ENTSO-E training data
    (also revisable after the fact, same instability problem as weather), the
    exact package patch state, and the git commit all matter too. This copies
    all of it into one self-contained, dated folder rather than requiring
    someone to hunt across three different cache directories that live
    outside the repo and aren't retained forever.

    Stored at _cache/reproducibility/<target_date>/ (gitignored -- these are
    data artifacts, not code; zip and attach separately if handing off to a
    verifying group, same convention as team_4's `software_link` ZIP release).

    Non-fatal: never blocks a submission. A missing piece here is a real gap
    worth investigating, but it isn't a reason to fail a live push.
    """
    try:
        import json
        import shutil
        import subprocess
        from importlib.metadata import version, PackageNotFoundError
        from spotforecast2_safe.data.fetch_data import get_data_home

        repo_root = Path(__file__).resolve().parent.parent
        pkg_dir = repo_root / "_cache" / "reproducibility" / str(dates.tomorrow.date())
        pkg_dir.mkdir(parents=True, exist_ok=True)

        data_home = get_data_home()
        cache_home = Path(
            os.environ.get("SPOTFORECAST2_CACHE")
            or (Path.home() / ".spotforecast2_cache" / CACHE_SUBDIR)
        )

        copied = {}

        if submission_path.exists():
            shutil.copy2(submission_path, pkg_dir / "submission.csv")
            copied["submission.csv"] = True

        snap_path = data_home / "weather_snapshots" / f"{dates.tomorrow.date()}.parquet"
        if snap_path.exists():
            shutil.copy2(snap_path, pkg_dir / "weather_snapshot.parquet")
            copied["weather_snapshot.parquet"] = True

        # Tuning results / model files have no target-date in their filename
        # (they're timestamped by *run* time, not target date) -- take the
        # most recently modified one, which is this run's, since it was just
        # written moments ago by run_pipeline().
        tuning_dir = cache_home / "tuning_results"
        newest_tuning = max(
            tuning_dir.glob("*.json"), default=None, key=lambda p: p.stat().st_mtime
        ) if tuning_dir.exists() else None
        if newest_tuning is not None:
            shutil.copy2(newest_tuning, pkg_dir / "tuning_results.json")
            copied["tuning_results.json"] = newest_tuning.name

        models_dir = cache_home / "models" / "bart26k-live-team4"
        newest_model = max(
            models_dir.glob("*.joblib"), default=None, key=lambda p: p.stat().st_mtime
        ) if models_dir.exists() else None
        if newest_model is not None:
            shutil.copy2(newest_model, pkg_dir / "model.joblib")
            copied["model.joblib"] = newest_model.name

        interim_dir = data_home / "interim"
        if interim_dir.exists():
            interim_out = pkg_dir / "interim"
            interim_out.mkdir(exist_ok=True)
            for f in interim_dir.glob("*.csv"):
                shutil.copy2(f, interim_out / f.name)
            copied["interim/"] = sorted(p.name for p in interim_dir.glob("*.csv"))

        def _pkg_version(name: str) -> str:
            try:
                return version(name)
            except PackageNotFoundError:
                return "not installed"

        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True,
        ).stdout.strip() or "unknown (not a git checkout?)"
        commit_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True,
        ).stdout.strip())

        manifest = {
            "target_date": str(dates.tomorrow.date()),
            "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "git_commit": commit_hash,
            "git_working_tree_dirty": commit_dirty,
            "variant_flags": {
                k: getattr(cfg, k, None) for k in (
                    "dayahead_temp", "dayahead_temp_hourly",
                    "dayahead_temp_anomaly", "peak_weight", "cdh_weight",
                    "quantile_alpha", "train_years",
                )
            },
            "package_versions": {
                "spotforecast2": _pkg_version("spotforecast2"),
                "spotforecast2_safe": _pkg_version("spotforecast2_safe"),
                "lightgbm": _pkg_version("lightgbm"),
            },
            "files": copied,
            "note": (
                "git_working_tree_dirty=true means this run's code may not "
                "exactly match git_commit -- check `git diff` at the time, "
                "if still available, before treating this as reproducible."
            ),
        }
        (pkg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        logger.info("reproducibility package saved -> %s  (%s)",
                    pkg_dir, ", ".join(k for k in copied if copied[k]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("reproducibility package failed (non-fatal): %s", exc)


class _PeakWeightFunc:
    """Demand-proportional sample weights: w = (actual_load / median).clip(0.5, 3.0).

    Defined as a module-level class so joblib can pickle it when saving models.
    Local closures are not picklable.
    """

    def __init__(self, load_series):
        self.load_series = load_series
        self.median_load = float(load_series.median())

    def __call__(self, index):
        vals = self.load_series.reindex(index).fillna(self.median_load)
        return (vals / self.median_load).clip(lower=0.5, upper=3.0).values.astype(float)


class _CDHWeightFunc:
    """CDH-proportional sample weights: w = (1 + cdh_h / mean_cdh).clip(1.0, 3.0).

    Upweights hot hours by how far above 18°C they are. Cool hours (CDH=0) stay
    at weight 1.0 — not penalised, just normal. Unlike demand-proportional weighting
    which targets load level, this targets the temperature signal directly so the
    model learns 'hot hours matter' rather than 'high-load hours matter'.

    Defined as a module-level class so joblib can pickle it when saving models.
    """

    def __init__(self, cdh_series):
        self.cdh_series = cdh_series
        positive = cdh_series[cdh_series > 0]
        self.mean_cdh = float(positive.mean()) if len(positive) > 0 else 1.0

    def __call__(self, index):
        vals = self.cdh_series.reindex(index).fillna(0.0)
        return (1.0 + (vals / self.mean_cdh).clip(lower=0.0, upper=2.0)).values.astype(float)


def inject_dayahead_temp(mt) -> None:
    """Add population-weighted daily max/mean temperature as day-ahead features.

    For each hour in the time series, computes the max and mean temperature of
    that calendar day (Berlin local time) as a population-weighted average across
    the 15 German load centres. Both features are constant within each day.

    In training: uses realized temperatures as a proxy for the day-ahead forecast
    (accurate within ~1-2°C). In prediction: uses the actual Open-Meteo forecast
    for the target day, which is already in the weather cache.

    Injects into mt.data_with_exog and appends to mt.exog_feature_names so
    SpotOptim sees the features during hyperparameter search.
    """
    from os import environ
    from spotforecast2_safe.weather.locations import GERMAN_LOAD_CENTERS

    cache_home = (
        environ.get("SPOTFORECAST2_WEATHER_CACHE")
        or environ.get("SPOTFORECAST2_CACHE")
    )
    if cache_home is None:
        logger.warning("inject_dayahead_temp: no weather cache env var set, skipping")
        return

    cache_path = Path(cache_home)
    frames, weights = [], []
    for loc in GERMAN_LOAD_CENTERS:
        f = cache_path / f"weather_cache_{loc.latitude:.4f}_{loc.longitude:.4f}.parquet"
        if not f.exists():
            logger.warning("inject_dayahead_temp: missing cache file %s", f.name)
            continue
        frames.append(pd.read_parquet(f)["temperature_2m"])
        weights.append(loc.weight)

    if not frames:
        logger.warning("inject_dayahead_temp: no city files found, skipping")
        return

    total = sum(weights)
    temp_hourly = sum(w / total * s for w, s in zip(weights, frames))

    # Daily max/mean in Berlin local time (load follows local clock)
    temp_berlin = temp_hourly.copy()
    temp_berlin.index = temp_berlin.index.tz_convert("Europe/Berlin")
    daily_max = temp_berlin.resample("D").max().rename("temp_dayahead_max")
    daily_mean = temp_berlin.resample("D").mean().rename("temp_dayahead_mean")
    # Cooling degree: demand rises above the ~18°C comfort threshold (same
    # threshold used by the existing CDH feature for historical temperature).
    CDH_BASE = 18.0
    daily_cdh = (daily_max - CDH_BASE).clip(lower=0.0).rename("cdh_dayahead")

    new_cols = ["temp_dayahead_max", "temp_dayahead_mean", "cdh_dayahead"]
    new_series = [daily_max, daily_mean, daily_cdh]

    for df in (mt.data_with_exog, mt.exo_pred):
        if df is None:
            continue
        idx_berlin = df.index.tz_convert("Europe/Berlin").normalize()
        for col, series in zip(new_cols, new_series):
            df[col] = series.reindex(idx_berlin).values

    mt.exog_feature_names = list(mt.exog_feature_names) + new_cols
    logger.info(
        "inject_dayahead_temp: added temp_dayahead_max + temp_dayahead_mean + cdh_dayahead "
        "(pop-weighted, %d cities, %d rows)",
        len(frames), len(mt.data_with_exog),
    )


def inject_dayahead_temp_hourly(mt) -> None:
    """Inject hourly population-weighted temperature forecast as features.

    Unlike inject_dayahead_temp() which collapses to daily max/mean, this
    injects the temperature for each specific hour — so the model at 18:00
    sees 'this hour will be 33°C' rather than 'daily max is 33°C'. Captures
    intraday temperature trajectories (cool morning → hot evening).

    Adds two features per row: temp_h (°C) and cdh_h (max(0, temp_h - 18)).
    Variant label: rich_dat_h.
    """
    from os import environ
    from spotforecast2_safe.weather.locations import GERMAN_LOAD_CENTERS

    cache_home = (
        environ.get("SPOTFORECAST2_WEATHER_CACHE")
        or environ.get("SPOTFORECAST2_CACHE")
    )
    if cache_home is None:
        logger.warning("inject_dayahead_temp_hourly: no weather cache env var set, skipping")
        return

    cache_path = Path(cache_home)
    frames, weights = [], []
    for loc in GERMAN_LOAD_CENTERS:
        f = cache_path / f"weather_cache_{loc.latitude:.4f}_{loc.longitude:.4f}.parquet"
        if not f.exists():
            logger.warning("inject_dayahead_temp_hourly: missing cache file %s", f.name)
            continue
        frames.append(pd.read_parquet(f)["temperature_2m"])
        weights.append(loc.weight)

    if not frames:
        logger.warning("inject_dayahead_temp_hourly: no city files found, skipping")
        return

    total = sum(weights)
    temp_hourly = sum(w / total * s for w, s in zip(weights, frames))
    temp_hourly.name = "temp_h"
    cdh_hourly = (temp_hourly - 18.0).clip(lower=0.0).rename("cdh_h")

    new_cols = ["temp_h", "cdh_h"]
    new_series = [temp_hourly, cdh_hourly]

    for df in (mt.data_with_exog, mt.exo_pred):
        if df is None:
            continue
        for col, series in zip(new_cols, new_series):
            df[col] = series.reindex(df.index).values

    mt.exog_feature_names = list(mt.exog_feature_names) + new_cols
    logger.info(
        "inject_dayahead_temp_hourly: added temp_h + cdh_h "
        "(pop-weighted, %d cities, %d rows)",
        len(frames), len(mt.data_with_exog),
    )


def inject_dayahead_temp_anomaly(mt) -> None:
    """Inject hourly temperature anomaly as features.

    Instead of raw temperature, injects temp_h - clim_mean(hour, month) computed
    from the 3-year weather archive.  Anomaly values stay in-distribution even on
    extreme heat wave days (e.g. +14°C above June norm vs 38°C absolute), avoiding
    LightGBM OOD leaf-clamping.

    Adds two features per row: temp_anom_h (°C above/below norm) and
    cdh_anom_h (max(0, temp_anom_h)).  Variant label: rich_anom.
    """
    from os import environ
    from spotforecast2_safe.weather.locations import GERMAN_LOAD_CENTERS

    cache_home = (
        environ.get("SPOTFORECAST2_WEATHER_CACHE")
        or environ.get("SPOTFORECAST2_CACHE")
    )
    if cache_home is None:
        logger.warning("inject_dayahead_temp_anomaly: no weather cache env var set, skipping")
        return

    cache_path = Path(cache_home)
    frames, weights = [], []
    for loc in GERMAN_LOAD_CENTERS:
        f = cache_path / f"weather_cache_{loc.latitude:.4f}_{loc.longitude:.4f}.parquet"
        if not f.exists():
            logger.warning("inject_dayahead_temp_anomaly: missing cache file %s", f.name)
            continue
        frames.append(pd.read_parquet(f)["temperature_2m"])
        weights.append(loc.weight)

    if not frames:
        logger.warning("inject_dayahead_temp_anomaly: no city files found, skipping")
        return

    total = sum(weights)
    temp_hourly = sum(w / total * s for w, s in zip(weights, frames))

    # Climatological mean per (hour, month) from the full archive
    clim_mean = temp_hourly.groupby(
        [temp_hourly.index.hour, temp_hourly.index.month]
    ).transform("mean")
    anomaly = (temp_hourly - clim_mean).rename("temp_anom_h")
    cdh_anomaly = anomaly.clip(lower=0.0).rename("cdh_anom_h")

    new_cols = ["temp_anom_h", "cdh_anom_h"]
    new_series = [anomaly, cdh_anomaly]

    for df in (mt.data_with_exog, mt.exo_pred):
        if df is None:
            continue
        for col, series in zip(new_cols, new_series):
            df[col] = series.reindex(df.index).values

    mt.exog_feature_names = list(mt.exog_feature_names) + new_cols
    logger.info(
        "inject_dayahead_temp_anomaly: added temp_anom_h + cdh_anom_h "
        "(pop-weighted, %d cities, climatological mean from %d archive hours)",
        len(frames), len(temp_hourly),
    )


def run_pipeline_with_weather_retry(cfg, search_space):
    """run_pipeline(), auto-repairing and retrying once on a weather-cache gap.

    Any of the 15 per-city caches can independently reach the point where its
    own history extends far enough to hit WeatherService's strict direct-read
    path for the first time (see repair_weather_cache_gaps()) — this can
    happen to a *different* city on every run, discovered progressively as
    each cache catches up, not just once at startup. Catching the specific
    failure here and retrying covers any city, not just the one already seen.
    """
    try:
        return run_pipeline(cfg, search_space)
    except ValueError as exc:
        if "missing row(s) in weather frame" not in str(exc):
            raise
        logger.warning(
            "run_pipeline failed on a weather-cache gap (%s); "
            "repairing all city caches and retrying once.", exc,
        )
        repair_weather_cache_gaps()
        return run_pipeline(cfg, search_space)


def run_pipeline(cfg, search_space):
    """qmd ``team4-prepare``...``team4-train``, figures dropped."""
    from spotforecast2.multitask import MultiTask

    mt = MultiTask(cfg, task="spotoptim")
    logger.info("prepare_data() ...")
    mt.prepare_data()

    # Outlier bounds from the load history (qmd team4-outlier-bounds, lines 1216-1223).
    s = mt.df_pipeline["Actual Load"].dropna()
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    low = max(0.0, med - OUTLIER_IQR_K * iqr)
    high = med + OUTLIER_IQR_K * iqr
    mt.config.bounds = [(low, high)]
    logger.info("outlier bounds (K=%d): [%.0f, %.0f] MW  (median=%.0f, IQR=%.0f)",
                OUTLIER_IQR_K, low, high, med, iqr)

    mt.detect_outliers()
    mt.impute()
    nan_after = int(mt.df_pipeline["Actual Load"].isna().sum())
    if nan_after != 0:
        raise Abort(1, f"imputation left {nan_after} NaN in Actual Load (must be 0)")
    logger.info("imputation OK (0 NaN remaining)")

    mt.build_exogenous_features()

    if getattr(cfg, "dayahead_temp", False):
        inject_dayahead_temp(mt)
    if getattr(cfg, "dayahead_temp_hourly", False):
        inject_dayahead_temp_hourly(mt)
    if getattr(cfg, "dayahead_temp_anomaly", False):
        inject_dayahead_temp_anomaly(mt)

    if getattr(cfg, "peak_weight", False):
        load = mt.df_pipeline["Actual Load"].dropna()
        mt.weight_func = _PeakWeightFunc(load)
        logger.info("peak_weight: demand-proportional weights (median=%.0f MW, clip=[0.5, 3.0])",
                    mt.weight_func.median_load)

    if getattr(cfg, "cdh_weight", False):
        from os import environ
        from spotforecast2_safe.weather.locations import GERMAN_LOAD_CENTERS
        cache_home = (environ.get("SPOTFORECAST2_WEATHER_CACHE")
                      or environ.get("SPOTFORECAST2_CACHE"))
        if cache_home is None:
            logger.warning("cdh_weight: no weather cache env var set, skipping")
        else:
            cache_path = Path(cache_home)
            frames, weights = [], []
            for loc in GERMAN_LOAD_CENTERS:
                f = cache_path / f"weather_cache_{loc.latitude:.4f}_{loc.longitude:.4f}.parquet"
                if f.exists():
                    frames.append(pd.read_parquet(f)["temperature_2m"])
                    weights.append(loc.weight)
            if not frames:
                logger.warning("cdh_weight: no city files found, skipping")
            else:
                total = sum(weights)
                temp_hourly = sum(w / total * s for w, s in zip(weights, frames))
                cdh_hourly = (temp_hourly - 18.0).clip(lower=0.0)
                mt.weight_func = _CDHWeightFunc(cdh_hourly)
                logger.info("cdh_weight: CDH-proportional weights (mean_cdh=%.2f°C, clip=[1.0, 3.0])",
                            mt.weight_func.mean_cdh)

    logger.info("starting SpotOptim search (n_trials=%s, sequential) -- the slow "
                "step (~5-10 min on a laptop) ...", cfg.n_trials_spotoptim)
    t0 = time.monotonic()
    # qmd team4-train: run_task_spotoptim, not run() -- the generic dispatcher
    # does not forward a custom search space.
    mt.run_task_spotoptim(search_space=search_space, show=False)
    logger.info("SpotOptim done in %.1f s", time.monotonic() - t0)
    report_boundary_positions(mt, search_space)
    return mt


def extract_y0(mt, dates: Dates) -> pd.Series:
    """qmd (lines 1699-1705)."""
    future = mt.results["spotoptim"]["Actual Load"]["future_pred"]
    logger.info("raw forecast: %d steps  %s -> %s",
                len(future), future.index.min(), future.index.max())
    y0 = future.loc[dates.tomorrow:dates.last_target]
    logger.info("y_0 slice: %d steps  %s -> %s", len(y0),
                y0.index.min() if len(y0) else None,
                y0.index.max() if len(y0) else None)
    return y0


SHAPE_MIN_CORR = 0.6     # qmd team4-shape-check: profile agreement threshold
SHAPE_MIN_RANGE = 0.5    #   amplitude: forecast range >= half the reference range


def warn_if_implausible_shape(y0: pd.Series, preds_entsoe: pd.Series,
                              interim: pd.DataFrame) -> None:
    """qmd ``team4-shape-check``: warn-only daily-profile plausibility check.

    Compares y_0 against ENTSO-E's day-ahead profile (fallback: actuals one
    week earlier) on Pearson correlation and daily-range ratio. Never aborts
    -- the operator decides (Art. 14); the 2026-06-05 flat forecast (r < 0,
    range ratio 0.30) is the incident this catches.
    """
    reference, ref_name = preds_entsoe.reindex(y0.index).dropna(), "ENTSO-E day-ahead"
    if len(reference) < 12:  # fallback: same weekday one week earlier
        week_ago = interim["Actual Load"].resample("h").mean().reindex(
            y0.index - pd.Timedelta(hours=168))
        week_ago.index = week_ago.index + pd.Timedelta(hours=168)
        reference, ref_name = week_ago.dropna(), "actuals one week earlier"
    if len(reference) < 12:
        logger.warning("shape check skipped: no reference profile available.")
        return
    common = y0.index.intersection(reference.index)
    r = float(y0.loc[common].corr(reference.loc[common]))
    ref_range = float(reference.loc[common].max() - reference.loc[common].min())
    range_ratio = float((y0.max() - y0.min()) / ref_range) if ref_range > 0 else float("nan")
    logger.info("shape check vs. %s (%d h): r=%.2f (min %.1f), range ratio=%.2f (min %.1f)",
                ref_name, len(common), r, SHAPE_MIN_CORR, range_ratio, SHAPE_MIN_RANGE)
    if r < SHAPE_MIN_CORR or range_ratio < SHAPE_MIN_RANGE:
        logger.warning("=" * 72)
        logger.warning("FORECAST SHAPE IMPLAUSIBLE: daily profile does not track the")
        logger.warning("%s reference (r=%.2f, range ratio=%.2f). Inspect the", ref_name, r, range_ratio)
        logger.warning("diagnostic figures before pushing (warn-only by design, Art. 14).")
        logger.warning("=" * 72)
    else:
        logger.info("shape check passed: daily profile and amplitude look plausible.")


def assert_no_leakage(mt) -> None:
    """qmd ``team4-load-model`` leakage guard (lines 1739-1746). Always runs
    (even with --no-figures): ENTSO-E's own forecast must never reach the model."""
    try:
        fc = mt.results["spotoptim"]["Actual Load"]["forecaster"]
        feat_names = list(fc.estimator.feature_name_)
    except Exception as exc:  # noqa: BLE001
        logger.warning("leakage guard skipped (cannot read fitted features): %s", exc)
        return
    leak = {"Forecasted Load", "Actual"}
    dwe = getattr(mt, "data_with_exog", None)
    exog_names = getattr(mt, "exog_feature_names", None)
    targets = {
        "training frame": set(dwe.columns) if dwe is not None else set(),
        "selected exogenous features": set(exog_names) if exog_names is not None else set(),
        "fitted model features": set(feat_names),
    }
    for where, cols in targets.items():
        if leak & cols:
            raise Abort(1, f"ENTSO-E forecast leaked into the {where} -- "
                           f"refusing to submit (CR-3 data-governance invariant).")
    logger.info("leakage guard passed: ENTSO-E forecast absent from target, "
                "exog, and model (%d features).", len(feat_names))


# --- diagnostic figures (best-effort; never block the submission) -------------
def family_of(col: str) -> str:
    """qmd helper (lines 1384-1396): map a feature name to its family."""
    c = col.lower()
    if "holiday" in c or "brueckentag" in c:
        return "holiday"
    if "poly" in c:
        return "polynomial"
    if "window" in c:
        return "weather_window"
    if any(k in c for k in ("sin", "cos", "rbf")):
        return "cyclical/RBF"
    if c.startswith("lag"):
        return "lag"
    return "weather/other"


def _try(fn, name: str) -> None:
    """Run a plot closure; log success/failure but never raise (CSV is critical)."""
    try:
        fn()
        logger.info("saved diagnostic: %s.png", name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not save %s.png: %s", name, exc)


def _plot_acf(acf, key_lags, conf: float, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.bar(acf["lag"], acf["autocorrelation"], width=0.9, color="#1F4E79")
    ax.axhline(conf, color="#999999", lw=0.8, ls="--")
    ax.axhline(-conf, color="#999999", lw=0.8, ls="--")
    for lag in key_lags:
        row = acf[acf["lag"] == lag]
        if not row.empty:
            val = float(row["autocorrelation"].iloc[0])
            ax.annotate(f"lag {lag}", xy=(lag, val), xytext=(lag, val + 0.08),
                        ha="center", fontsize=8, color="#E07B00",
                        arrowprops=dict(arrowstyle="->", color="#E07B00", lw=0.8))
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("ACF of Actual Load (annotated: data-selected key_lags)")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "acf.png", dpi=150)
    plt.close(fig)


def _plot_importance(fc, feat_names, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    importances = fc.estimator.feature_importances_
    ranking = sorted(zip(feat_names, importances), key=lambda kv: kv[1], reverse=True)[:20]
    family_color = {
        "lag": "#B22222", "weather/other": "#E07B00", "weather_window": "#F0A33C",
        "cyclical/RBF": "#1F4E79", "holiday": "#2E8B57", "polynomial": "#7B5EA7",
    }
    labels = [n for n, _ in ranking][::-1]
    values = [v for _, v in ranking][::-1]
    colors = [family_color.get(family_of(n), "#888888") for n in labels]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("split count (feature importance)")
    ax.set_title("Top-20 feature importances (coloured by family; lags in red)")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in family_color.values()]
    ax.legend(handles, family_color.keys(), fontsize=7, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "feature_importance.png", dpi=150)
    plt.close(fig)


def _plot_shap(fc, X_tr, figdir: Path) -> None:
    import shap
    import matplotlib.pyplot as plt

    step = max(1, len(X_tr) // 2000)
    X_sample = X_tr.iloc[::step]
    explainer = shap.TreeExplainer(fc.estimator)
    sv = explainer.shap_values(X_sample)
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(figdir / "shap.png", dpi=150)
    plt.close(fig)


def _plot_forecast(mt, preds_entsoe, figdir: Path) -> None:
    import plotly.io as pio

    from spotforecast2.plots.plotter import make_plot

    pkg = mt.results["spotoptim"]["Actual Load"]
    pkg["future_forecast"] = preds_entsoe.reindex(pkg["future_pred"].index)
    fig = make_plot(pkg)
    # kaleido v1 serialises via orjson, which rejects raw pandas Timestamps
    # in the figure ("Type is not JSON serializable: Timestamp"); round-trip
    # through Plotly's own JSON encoder first (no-op once sf2 >= 5.1 builds
    # the figure JSON-safe).
    pio.from_json(fig.to_json()).write_image(str(figdir / "forecast.png"), scale=2)


def _plot_vs_entsoe(y0, preds_entsoe, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    entsoe_fc = preds_entsoe.reindex(y0.index)
    overlap = entsoe_fc.dropna().index
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.plot(y0.index, y0.values / 1000, marker="o", ms=3, color="#1F4E79",
            label="team_4 forecast (y_0)")
    if len(overlap) > 0:
        ax.plot(entsoe_fc.index, entsoe_fc.values / 1000, marker="x", ms=4,
                color="#E07B00", label="ENTSO-E day-ahead forecast")
        mad = float((y0.loc[overlap] - entsoe_fc.loc[overlap]).abs().mean())
        logger.info("mean |team_4 - ENTSO-E| over %d overlap hours: %.0f MW",
                    len(overlap), mad)
    else:
        logger.info("ENTSO-E day-ahead forecast not yet published for the target "
                    "day; plotting team_4 only.")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load [GW]")
    ax.set_title("team_4 vs. ENTSO-E day-ahead forecast -- target day")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figdir / "team4_vs_entsoe.png", dpi=150)
    plt.close(fig)


def save_diagnostics(mt, y0, preds_entsoe, figdir: Path) -> None:
    """qmd figures fig-team4-importance / -shap / -predict / -vs-entsoe."""
    figdir.mkdir(parents=True, exist_ok=True)
    fc = mt.results["spotoptim"]["Actual Load"]["forecaster"]

    X_tr = feat_names = None
    try:
        X_tr, _ = fc.create_train_X_y(
            y=mt.data_with_exog["Actual Load"],
            exog=mt.data_with_exog[mt.exog_feature_names],
        )
        feat_names = list(fc.estimator.feature_name_)
    except Exception as exc:  # noqa: BLE001
        logger.warning("design-matrix reconstruction failed; skipping "
                       "importance/SHAP: %s", exc)

    if feat_names is not None:
        _try(lambda: _plot_importance(fc, feat_names, figdir), "feature_importance")
    if X_tr is not None:
        _try(lambda: _plot_shap(fc, X_tr, figdir), "shap")
    _try(lambda: _plot_forecast(mt, preds_entsoe, figdir), "forecast")
    _try(lambda: _plot_vs_entsoe(y0, preds_entsoe, figdir), "team4_vs_entsoe")


def save_prediction_band(mt, y0: pd.Series, outdir: Path) -> None:
    """Optional #3 probabilistic head (qmd ``team4-probabilistic-head``).

    Fits one ``LightGBM(objective="quantile", alpha=q)`` ``ForecasterRecursive``
    per quantile via ``quantile_lgbm_forecaster_factory`` (spotforecast2-safe
    >= 19.4.0), assembles a non-crossing q10/q50/q90 band over the live horizon
    with ``predict_quantile_band``, and writes it next to the figures. The point
    submission ``y0`` is unchanged --- this is a forecaster-level diagnostic.

    Best-effort: any failure logs a warning and is swallowed (the band must
    never block the submission). The heads use the default single-window lag
    config, not the anchored team4 factory, so the band is illustrative.
    """
    from spotforecast2_safe.multitask.factories import (
        predict_quantile_band,
        quantile_lgbm_forecaster_factory,
    )

    y_full = mt.data_with_exog["Actual Load"]
    y_train = y_full.dropna()
    exog_all = mt.data_with_exog[mt.exog_feature_names]
    exog_train = exog_all.loc[y_train.index]
    future_exog = exog_all.loc[y0.index]

    heads = quantile_lgbm_forecaster_factory(mt.config, quantiles=[0.1, 0.5, 0.9])
    for fc in heads.values():
        fc.fit(y=y_train, exog=exog_train)
    band = predict_quantile_band(heads, steps=len(y0), exog=future_exog)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "prediction_band.csv"
    out = band.copy()
    out.insert(0, "timestamp_utc", out.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out.to_csv(path, index=False)
    logger.info("wrote q10/q50/q90 prediction band -> %s  (point submission "
                "unchanged)", path)


# --- submission write + validate + push ---------------------------------------
def assert_contract(y0: pd.Series, dates: Dates) -> None:
    """qmd ``team4-submission-write`` contract (lines 1919-1924)."""
    assert len(y0) == 24, f"expected 24 hourly steps for y_0, got {len(y0)}"
    assert y0.index.min() == dates.tomorrow, (
        f"first step {y0.index.min()} != TOMORROW {dates.tomorrow}")
    assert (y0 > 0).all(), "non-positive forecast value -- spec requires > 0"
    assert y0.notna().all(), "NaN in forecast -- spec forbids"


def write_submission(y0: pd.Series, dates: Dates, team_id: str, lb_root: Path) -> Path:
    """qmd ``team4-submission-write`` (lines 1926-1935)."""
    sub_dir = lb_root / "submissions" / team_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / f"{dates.tomorrow.date().isoformat()}.csv"
    df = pd.DataFrame({
        "timestamp_utc": y0.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_mw": y0.round(2).values,
    })
    df.to_csv(path, index=False)
    logger.info("wrote %s  (%d rows, %.1f-%.1f MW)",
                path, len(df), df.forecast_mw.min(), df.forecast_mw.max())
    return path


def validate_submission(path: Path, lb_root: Path) -> int:
    """qmd ``team4-validate`` (lines 1947-1962): run the leaderboard's validator."""
    rel = path.relative_to(lb_root)
    cmd = ["uv", "run", "python", "scripts/validate_submission.py",
           "--path", str(rel), "--skip-deadline"]
    logger.info("validating: %s  (cwd=%s)", " ".join(cmd), lb_root)
    res = subprocess.run(cmd, cwd=lb_root, capture_output=True, text=True)
    if res.stdout.strip():
        logger.info("validator stdout: %s", res.stdout.strip())
    if res.stderr.strip():
        logger.warning("validator stderr: %s", res.stderr.strip())
    logger.info("validator exit code: %d", res.returncode)
    return res.returncode


def _fork_owner(lb_root: Path) -> str:
    import re
    res = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=lb_root, capture_output=True, text=True,
    )
    m = re.search(r"[:/]([^/]+)/[^/]+(?:\.git)?$", res.stdout.strip())
    return m.group(1) if m else "LaloMohamad"


def _push_steps(path: Path, dates: Dates, team_id: str, lb_root: Path):
    target = dates.tomorrow.date().isoformat()
    branch = f"submission/{team_id}-{target}-{dates.now.strftime('%H%M%S')}"
    rel = path.relative_to(lb_root)
    title = f"submission({team_id}): forecast for {target}"
    head = f"{_fork_owner(lb_root)}:{branch}"
    return branch, target, [
        # Branch from a freshly-fetched upstream/main, not local "main" --
        # local main is never fast-forwarded between runs, so branching from
        # it silently drags along every commit upstream has gained since
        # whenever main last moved (confirmed 255 commits stale on
        # 2026-07-12, ballooning a 1-file submission PR to 219 files).
        ["git", "fetch", "upstream", "main"],
        ["git", "switch", "-c", branch, "upstream/main"],
        ["git", "add", str(rel)],
        ["git", "commit", "-m", title],
        ["git", "push", "-u", "origin", branch],
        ["gh", "pr", "create", "--base", "main",
         "--repo", "bartzbeielstein/challenge-leaderboard", "--head", head,
         "--title", title,
         "--body", "Generated by bart26l-vorlesung "
                     "scripts/team_weather_report_submit.py."],
    ]


def push_submission(path: Path, dates: Dates, team_id: str, lb_root: Path) -> int:
    """qmd push block (lines 1972-1988): branch, commit, open the auto-merging PR."""
    branch, target, steps = _push_steps(path, dates, team_id, lb_root)
    for cmd in steps:
        logger.info("$ %s", " ".join(cmd))
        res = subprocess.run(cmd, cwd=lb_root, capture_output=True, text=True)
        if res.stdout.strip():
            logger.info("%s", res.stdout.strip())
        if res.returncode != 0:
            logger.error("push step failed: %s\n%s", " ".join(cmd), res.stderr.strip())
            return res.returncode
    logger.info("submission PR opened for %s on branch %s", target, branch)
    return 0


def print_push_instructions(path: Path, dates: Dates, team_id: str, lb_root: Path) -> None:
    _, target, steps = _push_steps(path, dates, team_id, lb_root)
    logger.info("CSV written but NOT pushed. To submit, run (or pass --push):")
    logger.info("  cd %s", lb_root)
    for cmd in steps:
        logger.info("  %s", " ".join(cmd))


def _figures_dir(args: argparse.Namespace, dates: Dates) -> Path:
    if args.figures_dir:
        figdir = Path(args.figures_dir).expanduser()
    else:
        figdir = (Path.home() / "spotforecast2_data" / DATA_SUBDIR / "figures"
                  / dates.tomorrow.date().isoformat())
    figdir.mkdir(parents=True, exist_ok=True)
    logger.info("diagnostics -> %s", figdir)
    return figdir


def _log_banner(dates: Dates, args: argparse.Namespace, lb_root: Path) -> None:
    logger.info("=" * 72)
    logger.info("team4_submit -- standalone chapter-14 pipeline")
    logger.info("now=%s  today=%s  tomorrow(target)=%s",
                dates.now, dates.today.date(), dates.tomorrow.date())
    logger.info("download window: %s -> %s", dates.start_dl, dates.end_dl)
    logger.info("data_home=%s", os.environ.get("SPOTFORECAST2_DATA"))
    logger.info("cache_home=%s", os.environ.get("SPOTFORECAST2_CACHE"))
    logger.info("team=%s  leaderboard=%s", args.team_id, lb_root)
    logger.info("tuning: n_trials=%s n_initial=%s (sequential)",
                args.n_trials, args.n_initial)
    logger.info("=" * 72)


# --- orchestration ------------------------------------------------------------
def _run(args: argparse.Namespace) -> int:
    configure_environment()
    repair_weather_cache_gaps()
    lb_root = Path(args.leaderboard_root).expanduser()
    check_team_registry(args.team_id, lb_root)

    now = pd.Timestamp.now(tz="UTC")
    dates = compute_dates(now, args.data_end)
    _log_banner(dates, args, lb_root)

    # ---- download / cache decision ----
    api_key = os.environ.get("ENTSOE_API_KEY")
    used_cache = False
    if args.skip_download:
        logger.warning(">> --skip-download set: reusing cached interim, NO live "
                       "ENTSO-E call.")
        used_cache = True
    elif not api_key:
        if args.allow_stale:
            logger.error(">> ENTSOE_API_KEY missing and --allow-stale set: "
                         "falling back to cached interim.")
            used_cache = True
        else:
            raise Abort(3, "ENTSOE_API_KEY is not set. Export it, or pass "
                           "--skip-download / --allow-stale to use cached data.")
    else:
        timeout = args.timeout if args.timeout > 0 else None
        if not robust_download(api_key, dates, max_retries=args.max_retries,
                               backoff=args.backoff, timeout=timeout):
            if args.allow_stale:
                logger.error("=" * 72)
                logger.error("ENTSO-E DOWNLOAD FAILED -- --allow-stale set: reusing "
                             "cached interim data.")
                logger.error("The submission will be built from possibly-stale "
                             "cached actuals (gated by the yesterday-23:00 check).")
                logger.error("=" * 72)
                used_cache = True
            else:
                raise Abort(2, "ENTSO-E download failed and --allow-stale not set; "
                               "aborting (no stale submission).")
    if api_key and not args.skip_download:
        download_side_tables(api_key, dates,
                             timeout=args.timeout if args.timeout > 0 else None)

    interim = load_interim()
    cov = assert_coverage(interim, dates, fallback=used_cache)
    preds_entsoe = entsoe_predictions(interim)

    figdir = None if args.no_figures else _figures_dir(args, dates)
    key_lags = select_key_lags(interim, cov.last_full_hour, figdir)

    cfg = build_config(key_lags, cov, n_trials=args.n_trials,
                       n_initial=args.n_initial, train_years=args.train_years,
                       max_time=args.max_time)
    if args.tensorboard:
        # Plain config attributes — SpotOptimStrategy (sf2 >= 7.0.0) forwards
        # tensorboard_* to the SpotOptim constructor. SpotOptim (>= 1.0) tunes
        # sequentially, logging each evaluation as it completes.
        from spotforecast2_safe.data.fetch_data import get_data_home

        tb_path = args.tensorboard_path or str(
            get_data_home() / "tensorboard" / str(dates.tomorrow.date())
        )
        cfg.tensorboard_log = True
        cfg.tensorboard_path = tb_path
        logger.info("TensorBoard logging on -> %s  (watch: tensorboard --logdir %s)",
                    tb_path, tb_path)
    mt = run_pipeline_with_weather_retry(cfg, build_search_space(key_lags))
    _save_weather_snapshot(mt, dates)
    y0 = extract_y0(mt, dates)
    warn_if_implausible_shape(y0, preds_entsoe, interim)

    assert_no_leakage(mt)
    if figdir is not None:
        save_diagnostics(mt, y0, preds_entsoe, figdir)

    if args.prediction_band:
        band_dir = figdir or _figures_dir(args, dates)
        _try(lambda: save_prediction_band(mt, y0, band_dir), "prediction_band")

    y0 = y0.clip(lower=1.0)
    assert_contract(y0, dates)
    path = write_submission(y0, dates, args.team_id, lb_root)
    _save_reproducibility_package(mt, dates, cfg, path)

    code = 0
    if not args.no_validate:
        if validate_submission(path, lb_root) != 0:
            logger.error("validator rejected the submission.")
            code = 4

    if args.push:
        if code != 0:
            logger.error("skipping --push because validation failed.")
        elif push_submission(path, dates, args.team_id, lb_root) != 0:
            logger.error("push failed.")
            code = code or 1
    else:
        print_push_instructions(path, dates, args.team_id, lb_root)

    if code == 0:
        logger.info("DONE: submission for %s written%s.",
                    dates.tomorrow.date().isoformat(),
                    " and validated" if not args.no_validate else "")
    return code


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        return _run(args)
    except Abort as abort:
        logger.error("%s", abort)
        return abort.code
    except Exception:  # noqa: BLE001
        logger.exception("unexpected failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
