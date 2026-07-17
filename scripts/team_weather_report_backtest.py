#!/usr/bin/env python3
"""Pseudo-live backtests for the Team Weather Report pipeline.

This script intentionally does not submit anything. It runs
``team_weather_report_submit.py`` against an isolated copy of the cached data,
masks target actuals after a simulated run time, and compares the 24h forecast
against the now-known actual load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE_DATA_SUBDIR = "ddmo_ch14_team4"
BACKTEST_ROOT_SUBDIR = "ddmo_ch14_team4_backtest"
DEFAULT_OUTPUT_DIR = ROOT / "_cache" / "team4_backtests" / "season_01" / "02_bounds_t10i5"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pseudo-live backtest for team_weather_report_submit.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", choices=["plain", "wasitplain", "rich"], default="plain",
                   help="plain=original baseline; wasitplain=re-run to verify; rich=+degree_hours/apparent_temp/population_weather")
    p.add_argument("--quantile-alpha", type=float, default=None,
                   help="if set, trains LightGBM with objective=quantile at this alpha (e.g. 0.6) "
                        "and sets SpotOptim CV metric to pinball loss at the same alpha")
    p.add_argument("--dayahead-temp", action="store_true", default=False,
                   help="inject population-weighted daily max/mean temperature as day-ahead features")
    p.add_argument("--dayahead-temp-hourly", action="store_true", default=False,
                   help="inject population-weighted hourly temperature as day-ahead features (variant: rich_dat_h)")
    p.add_argument("--anomaly", action="store_true", default=False,
                   help="inject hourly temperature anomaly (deviation from clim. mean) as features (variant: anom)")
    p.add_argument("--peak-weight", action="store_true", default=False,
                   help="demand-proportional sample weights: upweight high-load hours during training (suffix: _wt)")
    p.add_argument("--cdh-weight", action="store_true", default=False,
                   help="CDH-proportional sample weights: upweight hot hours by cooling degree hours (suffix: _cdhwt)")
    p.add_argument("--wide-bounds", action="store_true", default=False,
                   help="season_01 Exp08: widen num_leaves [8,1024]->[4,1024] and max_depth [3,48]->[2,48] "
                        "(SpotOptim keeps landing near the current lower bounds on both)")
    p.add_argument("--richer-lags", action="store_true", default=False,
                   help="season_01 Exp09: add a 72h-anchor lag candidate to the search pool "
                        "(48h/168h/336h already covered by existing candidates; 72h is the genuine gap)")
    p.add_argument("--target-date", required=True, help="YYYY-MM-DD target day")
    p.add_argument("--n-trials", type=int, required=True)
    p.add_argument("--n-initial", type=int, required=True)
    p.add_argument("--pseudo-hour-utc", type=int, default=21,
                   help="simulated run hour on D-1 in UTC")
    p.add_argument("--train-years", type=int, default=2)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--session", type=int, default=0,
                   help="session index written as a sub-folder under the variant folder (0, 1, 2, …)")
    p.add_argument("--actual-csv", type=Path, default=None,
                   help="optional ENTSO-E GUI CSV with Actual Total Load (MW)")
    p.add_argument("--keep-data-copy", action="store_true",
                   help="keep the isolated copied data home after the run")
    p.add_argument(
        "--recover-saved-models",
        action="store_true",
        help="reuse an existing isolated data copy and saved SpotOptim models; "
             "build predictions and metrics without training",
    )
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def _data_home(subdir: str) -> Path:
    return Path.home() / "spotforecast2_data" / subdir


def _config_suffix(args: argparse.Namespace) -> str:
    """Suffix distinguishing modifier combos that share --variant/date/trials.

    Must include every flag that changes model behavior. Without this, two
    combos differing only in e.g. --cdh-weight vs --peak-weight would resolve
    to the identical isolated data-copy and SpotOptim cache directory — safe
    when run one after another, but silently corrupting (rmtree racing a
    concurrent read, or one run's cleanup deleting the other's live directory)
    if two such combos for the same target date are ever run concurrently.
    """
    parts = []
    if args.dayahead_temp:
        parts.append("dat")
    if args.dayahead_temp_hourly:
        parts.append("dath")
    if args.anomaly:
        parts.append("anom")
    if args.peak_weight:
        parts.append("wt")
    if args.cdh_weight:
        parts.append("cdhwt")
    if args.quantile_alpha is not None:
        parts.append(f"q{int(args.quantile_alpha * 100)}")
    if args.wide_bounds:
        parts.append("wideb")
    if args.richer_lags:
        parts.append("richlag")
    return ("_" + "_".join(parts)) if parts else ""


def prepare_data_copy(args: argparse.Namespace) -> tuple[Path, Path]:
    src = _data_home(LIVE_DATA_SUBDIR)
    if not src.exists():
        raise FileNotFoundError(f"Live data home not found: {src}")

    subdir = (
        f"{BACKTEST_ROOT_SUBDIR}/"
        f"{args.variant}{_config_suffix(args)}_{args.target_date}_t{args.n_trials}_i{args.n_initial}"
    )
    dst = _data_home(subdir)
    if dst.exists():
        if BACKTEST_ROOT_SUBDIR not in str(dst):
            raise RuntimeError(f"Refusing to delete suspicious path: {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("copying live data home (%s) -> isolated backtest copy (%s) ...", src, dst)
    shutil.copytree(src, dst)
    logger.info("copy complete.")

    return subdir, dst


def mask_future_actuals(args: argparse.Namespace, pseudo_now: pd.Timestamp, data_home: Path) -> None:
    csv = data_home / "interim" / "energy_load.csv"
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    if "Actual Load" not in df.columns:
        raise KeyError(f"{csv} has no 'Actual Load' column")
    df.loc[df.index > pseudo_now, "Actual Load"] = pd.NA
    df.to_csv(csv)


def actual_load_from_live_cache(args: argparse.Namespace, index: pd.DatetimeIndex) -> pd.Series:
    if args.actual_csv is not None:
        df = pd.read_csv(args.actual_csv)
        if "Actual Total Load (MW)" not in df.columns:
            raise KeyError(f"{args.actual_csv} has no 'Actual Total Load (MW)' column")
        mtu_col = next((c for c in df.columns if c.startswith("MTU")), None)
        if mtu_col is None:
            raise KeyError(f"{args.actual_csv} has no MTU column")
        start = df[mtu_col].astype(str).str.split(" - ", regex=False).str[0]
        ts = pd.to_datetime(start, format="%d/%m/%Y %H:%M").dt.tz_localize(
            "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
        actual = pd.Series(
            pd.to_numeric(df["Actual Total Load (MW)"], errors="coerce").values,
            index=ts,
            name="actual_mw",
        )
        return actual.resample("h").mean().reindex(index).rename("actual_mw")

    src = _data_home(LIVE_DATA_SUBDIR)
    csv = src / "interim" / "energy_load.csv"
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    if "Actual Load" not in df.columns:
        raise KeyError(f"{csv} has no 'Actual Load' column")
    actual = df["Actual Load"].astype(float)
    return actual.resample("h").mean().reindex(index).rename("actual_mw")


def metrics(forecast: pd.Series, actual: pd.Series) -> dict[str, float]:
    err = forecast - actual
    ok = err.notna() & actual.notna()
    if int(ok.sum()) != 24:
        raise RuntimeError(f"Need 24 actual hours, got {int(ok.sum())}")
    err = err[ok]
    actual = actual[ok]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(math.sqrt((err ** 2).mean())),
        "mape": float((err.abs() / actual.abs()).mean() * 100),
        "bias": float(err.mean()),
    }


def run_saved_model_pipeline(mod, cfg):
    """Rebuild preprocessing and predict with persisted forecasters only."""
    from spotforecast2.multitask import MultiTask

    mt = MultiTask(cfg, task="predict")
    mod.logger.info("recovery: prepare_data() without training ...")
    mt.prepare_data()

    bounds = []
    for zone in mt.run_state.targets:
        s = mt.df_pipeline[zone].dropna()
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        bounds.append((max(0.0, med - mod.OUTLIER_IQR_K * iqr),
                       med + mod.OUTLIER_IQR_K * iqr))
    mt.config.bounds = bounds

    mt.detect_outliers()
    mt.impute()
    mt.build_exogenous_features()
    mod.logger.info("recovery: loading saved SpotOptim models; no tuning")
    mt.run_task_predict(show=False, task_name="spotoptim", max_age_days=None)

    # The chapter helpers inspect the task under the original training label.
    mt.results["spotoptim"] = mt.results["predict"]
    return mt


def run_backtest(args: argparse.Namespace) -> int:
    target = pd.Timestamp(args.target_date, tz="UTC")
    pseudo_now = target - pd.Timedelta(days=1) + pd.Timedelta(hours=args.pseudo_hour_utc)
    module_path = ROOT / "scripts" / "team_weather_report_submit.py"
    mod = _load_module(module_path, f"team4_backtest_{args.variant}")
    mod.setup_logging(args.verbose)

    dates = mod.compute_dates(pseudo_now, None)
    data_subdir = (
        f"{BACKTEST_ROOT_SUBDIR}/"
        f"{args.variant}{_config_suffix(args)}_{args.target_date}_t{args.n_trials}_i{args.n_initial}"
    )
    if args.recover_saved_models:
        copied_home = _data_home(data_subdir)
        if not copied_home.exists():
            raise FileNotFoundError(
                f"Recovery data copy not found: {copied_home}. "
                "Run the backtest normally first."
            )
        mod.logger.info("recovery: reusing existing data copy %s", copied_home)
    else:
        data_subdir, copied_home = prepare_data_copy(args)
    mod.DATA_SUBDIR = data_subdir
    mod.CACHE_SUBDIR = data_subdir
    mod.configure_environment()
    os.environ["SPOTFORECAST2_WEATHER_CACHE"] = str(
        Path.home() / ".spotforecast2_cache" / "weather_shared"
    )
    mod.repair_weather_cache_gaps()

    # The isolated copy is disposable. Re-applying the cutoff is idempotent and
    # also makes recovery safe when its data files were restored from another
    # target-day cache.
    mask_future_actuals(args, pseudo_now, copied_home)
    mod.logger.info(
        "BACKTEST variant=%s target=%s pseudo_now=%s data_home=%s",
        args.variant,
        args.target_date,
        pseudo_now,
        copied_home,
    )

    interim = mod.load_interim()
    cov = mod.assert_coverage(interim, dates, fallback=True)
    preds_entsoe = mod.entsoe_predictions(interim)
    key_lags = mod.select_key_lags(interim, cov.last_full_hour, None)
    cfg = mod.build_config(
        key_lags,
        cov,
        n_trials=args.n_trials,
        n_initial=args.n_initial,
        train_years=args.train_years,
        max_time=args.max_time,
    )
    if args.variant == "wasitplain":
        cfg.include_weather_windows = False
        cfg.include_degree_hours = False
        cfg.include_apparent_temperature = False
        cfg.use_population_weighted_weather = False
    if args.quantile_alpha is not None:
        from spotforecast2_safe.forecaster.metrics import create_mean_pinball_loss
        cfg.quantile_alpha = args.quantile_alpha
        cfg.cv_metric = create_mean_pinball_loss(args.quantile_alpha)
    if args.dayahead_temp:
        cfg.dayahead_temp = True
    if args.dayahead_temp_hourly:
        cfg.dayahead_temp_hourly = True
    if args.anomaly:
        cfg.dayahead_temp_anomaly = True
    if args.peak_weight:
        cfg.peak_weight = True
    if args.cdh_weight:
        cfg.cdh_weight = True
    if args.recover_saved_models:
        mt = run_saved_model_pipeline(mod, cfg)
    else:
        mt = mod.run_pipeline(cfg, mod.build_search_space(
            key_lags, wide_bounds=args.wide_bounds, richer_lags=args.richer_lags))
    y0 = mod.extract_y0(mt, dates).rename("forecast_mw")
    mod.warn_if_implausible_shape(y0, preds_entsoe, interim)
    mod.assert_no_leakage(mt)
    mod.assert_contract(y0, dates)

    try:
        tuned_est = mt.results["spotoptim"]["Actual Load"]["forecaster"].estimator
        tuned_hyperparameters = {
            k: v for k, v in tuned_est.get_params().items()
            if v is None or isinstance(v, (int, float, str, bool))
        }
    except Exception as exc:  # noqa: BLE001
        mod.logger.warning("could not capture tuned hyperparameters: %s", exc)
        tuned_hyperparameters = None

    if args.anomaly:
        # "anom" already implies the rich feature set on top of which the
        # anomaly is injected — dropping the redundant "rich_" prefix.
        base_label = "anom" if args.variant == "rich" else f"{args.variant}_anom"
    elif args.dayahead_temp_hourly:
        base_label = f"{args.variant}_dat_h"
    elif args.dayahead_temp:
        base_label = f"{args.variant}_dat"
    else:
        base_label = args.variant

    variant_label = (
        base_label
        + ("_cdhwt" if args.cdh_weight else "_wt" if args.peak_weight else "")
        + (f"_q{int(args.quantile_alpha * 100)}" if args.quantile_alpha is not None else "")
    )
    actual = actual_load_from_live_cache(args, y0.index)
    summary = {
        "variant": variant_label,
        "target_date": args.target_date,
        "n_trials": args.n_trials,
        "n_initial": args.n_initial,
        "quantile_alpha": args.quantile_alpha,
        "dayahead_temp": args.dayahead_temp,
        "pseudo_now_utc": pseudo_now.isoformat(),
        "tuned_hyperparameters": tuned_hyperparameters,
        **metrics(y0, actual),
    }

    out_dir = args.output_dir / args.target_date
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{variant_label}_t{args.n_trials}_i{args.n_initial}"
    m = metrics(y0, actual)
    compare = pd.DataFrame(
        {
            "timestamp_utc": y0.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_mw": y0.round(2).values,
            "actual_mw": actual.round(2).values,
            "error_mw": (y0 - actual).round(2).values,
            "abs_error_mw": (y0 - actual).abs().round(2).values,
        }
    )
    summary_rows = pd.DataFrame([
        {"timestamp_utc": "DAILY_MAE",  "forecast_mw": "", "actual_mw": "", "error_mw": "", "abs_error_mw": round(m["mae"], 2)},
        {"timestamp_utc": "DAILY_RMSE", "forecast_mw": "", "actual_mw": "", "error_mw": "", "abs_error_mw": round(m["rmse"], 2)},
        {"timestamp_utc": "DAILY_MAPE_PCT", "forecast_mw": "", "actual_mw": "", "error_mw": "", "abs_error_mw": round(m["mape"], 2)},
        {"timestamp_utc": "DAILY_BIAS", "forecast_mw": "", "actual_mw": "", "error_mw": "", "abs_error_mw": round(m["bias"], 2)},
    ])
    pd.concat([compare, summary_rows], ignore_index=True).to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))

    if not args.keep_data_copy and not args.recover_saved_models:
        shutil.rmtree(copied_home)
    return 0


def main() -> int:
    try:
        return run_backtest(parse_args())
    except Exception as exc:  # noqa: BLE001
        logging.exception("backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
