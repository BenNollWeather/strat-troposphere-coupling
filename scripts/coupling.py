#!/usr/bin/env python3
"""Polar-vortex coupling maps for ECMWF EPS and AIFS-ENS.

The day-15 ensemble-mean height is drawn at 10, 100, 300 and 500 hPa.
The Northern Hemisphere map looks down on the North Pole. The Southern
Hemisphere map looks down on the South Pole, with 174°E (New Zealand) at
the top of the page.

ECMWF open-data messages and ARCO-ERA5 chunks are read into memory. This
script does not write those downloads to disk, and the caller deletes the
temporary work directory when the process exits.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

G0 = 9.80665
LAST_HOUR = 360
LEVELS = [10, 100, 300, 500]
N_LAT = 181
N_LON = 720
CAP_LAT = 60
HEMISPHERE_LAT = 20
POLAR_LAT_MIN = 20
# NorthPolarStereo central_longitude is the meridian at the bottom.
NORTH_CENTER_LON = -100.0
# SouthPolarStereo central_longitude is the meridian at the top.
# 174°E passes through New Zealand, so the islands sit at the top.
SOUTH_TOP_LON = 174.0
STILL_WIDTH_PX = 2160
SMOOTH_SIGMA = 1.0
SIGMA_RANGE = (-3, 6)
SIGMA_TICKS = [-3, -2, -1, 0, 1, 2, 4, 6]
SCALE_POWER = 0.75
DEADBAND = 0.2
STD_FLOOR = 1.0
CLIM_START, CLIM_END = 1991, 2020
ARCO_URL = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
)
BASES = {
    "gcs": "https://storage.googleapis.com/ecmwf-open-data",
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
}
CYCLES = {"ifs-ens": (0, 12), "aifs-ens": (0, 6, 12, 18)}
# A cycle younger than this cannot be the day-15 file the schedule is waiting for.
MIN_AGE = {"aifs-ens": timedelta(hours=6), "ifs-ens": timedelta(hours=7)}
HEADLINES = {
    "none": "No coupling predicted",
    "weak": "Weak coupling predicted",
    "moderate": "Moderate coupling predicted",
    "strong": "Strong coupling predicted",
}
LEVEL_WORDS = {
    10: "the stratosphere",
    100: "the tropopause",
    300: "the jet-stream level",
    500: "the mid-troposphere",
}
MONTH_ABBR = {
    "Jan": "Jan.", "Feb": "Feb.", "Mar": "March", "Apr": "April", "May": "May",
    "Jun": "June", "Jul": "July", "Aug": "Aug.", "Sep": "Sept.", "Oct": "Oct.",
    "Nov": "Nov.", "Dec": "Dec.",
}
MODEL_CFG = {
    "ifs-ens": dict(
        name="ECMWF EPS", short="EPS", root="ifs/0p25/enfo",
        kinds=("ef",), param="gh", to_gpm=1.0,
        image="ecmwf_ensemble_strat_trop_coupling.png",
        image_south="ecmwf_ensemble_strat_trop_coupling_south.png",
        schedule="12-hourly, when the EPS day-15 files arrive (08:35 and 20:35 UTC)",
        product_id="strat-coupling-eps",
    ),
    "aifs-ens": dict(
        name="AIFS-ENS", short="AIFS-ENS", root="aifs-ens/0p25/enfo",
        kinds=("cf", "pf"), param="z", to_gpm=1.0 / G0,
        image="aifs_ensemble_strat_trop_coupling.png",
        image_south="aifs_ensemble_strat_trop_coupling_south.png",
        schedule="6-hourly, when the AIFS-ENS day-15 files arrive (01:55, 07:55, 13:55 and 19:55 UTC)",
        product_id="strat-coupling-aifs",
    ),
}
PANEL_INFO = {
    10: ("10 hPa", "stratosphere, ~31 km"),
    100: ("100 hPa", "tropopause, ~16 km"),
    300: ("300 hPa", "jet stream, ~9 km"),
    500: ("500 hPa", "mid-troposphere, ~5.5 km"),
}
CONTOUR_STEP_M = {10: 200, 100: 100, 300: 120, 500: 60}
COUPLING_THRESHOLDS = {10: 1.0, 100: 0.5, 300: 0.5, 500: 0.5}
DATA_SUFFIXES = {".grib", ".grib2", ".grb", ".grb2", ".npz", ".nc", ".nc4"}
UPLOAD_SUFFIXES = {".png", ".json", ".txt", ".html"}
HEMISPHERES = ("north", "south")

# Filled by the download helpers so the log can show that nothing was saved.
BYTES_READ = {"ecmwf": 0, "arco": 0}


def ap_date(ts: datetime) -> str:
    return f"{MONTH_ABBR[ts.strftime('%b')]} {ts.day}"


def coupling_headline(rating: str, valid: datetime) -> str:
    """Scoreboard line. The date is when the signal is forecast, so it reads 'on'."""
    return f"{HEADLINES[rating]} on {ap_date(valid)}"


def rate_coupling(rel: np.ndarray) -> dict:
    n = 0
    for lev, value in zip(LEVELS, rel):
        if value >= COUPLING_THRESHOLDS.get(lev, 0.5):
            n += 1
        else:
            break
    rating = ("none", "none", "weak", "moderate", "strong")[n]
    if n == 0:
        detail = "No strong stratospheric warming in this forecast"
    elif n == 1:
        detail = "The stratospheric warming stays above the tropopause"
    else:
        reached = LEVELS[n - 1]
        detail = (
            f"The warm anomaly reaches {LEVEL_WORDS.get(reached, str(reached) + ' hPa')} "
            f"({reached} hPa)"
        )
    return dict(
        n_levels=n, rating=rating, headline=HEADLINES[rating], detail=detail,
        reach=LEVELS[n - 1] if n else None,
    )


def hemisphere_lat(hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        return np.linspace(90, 0, N_LAT).astype(np.float32)
    if hemisphere == "south":
        return np.linspace(0, -90, N_LAT).astype(np.float32)
    raise ValueError(hemisphere)


def split_half_degree(full: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 0.5° globe (361 latitudes, 90°N to 90°S) into the two maps."""
    if full.shape[-2] != 361:
        raise ValueError(f"expected 361 latitudes, got {full.shape[-2]}")
    return {"north": full[..., :N_LAT, :], "south": full[..., 180:, :]}


def subsample_quarter_degree(vals: np.ndarray) -> dict[str, np.ndarray]:
    """721×1440 field, 90°N to 90°S at 0.25°, to 0.5° northern and southern maps."""
    if vals.shape[-2:] != (721, 1440):
        raise ValueError(f"expected a 0.25° global grid, got {vals.shape[-2:]}")
    return split_half_degree(vals[..., ::2, ::2])


def polar_projection(hemisphere: str):
    import cartopy.crs as ccrs
    if hemisphere == "north":
        return ccrs.NorthPolarStereo(central_longitude=NORTH_CENTER_LON)
    if hemisphere == "south":
        return ccrs.SouthPolarStereo(central_longitude=SOUTH_TOP_LON)
    raise ValueError(hemisphere)


def polar_extent(hemisphere: str) -> list[float]:
    if hemisphere == "north":
        return [-180, 180, POLAR_LAT_MIN, 90]
    return [-180, 180, -90, -POLAR_LAT_MIN]


def expected_run(model: str, now: datetime) -> datetime:
    """Latest allowed cycle old enough for the day-15 file to exist."""
    cycle = now.replace(minute=0, second=0, microsecond=0)
    cycle = cycle.replace(hour=(cycle.hour // 6) * 6)
    min_age = MIN_AGE[model]
    for k in range(16):
        run = cycle - timedelta(hours=6 * k)
        if run.hour not in CYCLES[model]:
            continue
        if now - run >= min_age:
            return run
    raise RuntimeError(f"no {model} cycle is old enough at {now:%Y-%m-%d %H:%M}Z")


def under_home(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path.home().resolve())
    except ValueError:
        return False
    return True


def prepare_work(work: Path) -> Path:
    if under_home(work) or "Downloads" in work.parts:
        raise RuntimeError("WORK_DIR must be a temporary directory outside the home folder")
    if work.exists():
        shutil.rmtree(work)
    for name in ("tmp", "xdg", "mpl", "cartopy", "out"):
        (work / name).mkdir(parents=True)
    os.environ["TMPDIR"] = str(work / "tmp")
    os.environ["XDG_CACHE_HOME"] = str(work / "xdg")
    os.environ["MPLCONFIGDIR"] = str(work / "mpl")
    os.environ["CARTOPY_DATA_DIR"] = str(work / "cartopy")
    tempfile.tempdir = str(work / "tmp")
    return work


def delete_work_tree(work: Path) -> int:
    if not work.exists():
        return 0
    resolved = work.resolve()
    if resolved == Path("/") or under_home(resolved) or len(resolved.parts) < 3:
        raise RuntimeError(f"refusing to delete {resolved}")
    count = sum(1 for path in resolved.rglob("*") if path.is_file())
    shutil.rmtree(resolved)
    return count


def sweep_leaked_downloads(started: float, work: Path) -> int:
    """Delete fsspec cache files this run wrote outside the work directory.

    The work directory itself is removed separately. This does not walk /tmp,
    so unrelated files are left alone.
    """
    del work
    removed = 0
    cache = Path.home() / ".cache" / "fsspec"
    if not cache.exists():
        return 0
    for path in cache.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < started - 1:
                continue
        except OSError:
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _font_props(font_dir: Path):
    from matplotlib import font_manager
    for font_file in sorted(font_dir.glob("LibreFranklin-*.ttf")):
        font_manager.fontManager.addfont(str(font_file))
    bold = font_dir / "LibreFranklin-Bold.ttf"
    light = font_dir / "LibreFranklin-Light.ttf"
    if not bold.exists() or not light.exists():
        raise FileNotFoundError("Libre Franklin fonts are missing from assets/fonts")
    return font_manager.FontProperties(fname=str(bold)), font_manager.FontProperties(fname=str(light))


def _colormap():
    import matplotlib.colors as mcolors
    cold = ["#d9ecf7", "#9ecae1", "#5aa0cf", "#2f73b5", "#1f4e8c"]
    warm = ["#fff1b8", "#fdd069", "#f9a03f", "#ec6a2c", "#d23a24", "#a3192f", "#6b0f45", "#33092f"]
    neutral = "#f1f1ee"
    vmin, vmax = SIGMA_RANGE
    forward = lambda x: np.sign(x) * np.abs(x) ** SCALE_POWER
    inverse = lambda y: np.sign(y) * np.abs(y) ** (1 / SCALE_POWER)
    norm = mcolors.FuncNorm((forward, inverse), vmin=vmin, vmax=vmax)
    norm.clip = True
    p_lo, p_hi = float(norm(-DEADBAND)), float(norm(DEADBAND))
    blend = 0.05
    stops = list(zip(np.linspace(0.0, p_lo - blend, len(cold)), cold[::-1]))
    stops += [(p_lo, neutral), (p_hi, neutral)]
    stops += list(zip(np.linspace(p_hi + blend, 1.0, len(warm)), warm))
    cmap = mcolors.LinearSegmentedColormap.from_list("coupling", sorted(stops, key=lambda t: t[0]), N=1024)
    return cmap, norm


def _smooth_roll(field: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if SMOOTH_SIGMA:
        field = np.stack([gaussian_filter(x, SMOOTH_SIGMA, mode=("nearest", "wrap")) for x in field])
    return np.roll(field, N_LON // 2, axis=-1)


def _cap_stats(anom_m: np.ndarray, anom_s: np.ndarray, lat: np.ndarray, hemisphere: str):
    if hemisphere == "north":
        cap_rows = np.where(lat >= CAP_LAT)[0]
        hem_rows = np.where(lat >= HEMISPHERE_LAT)[0]
    else:
        cap_rows = np.where(lat <= -CAP_LAT)[0]
        hem_rows = np.where(lat <= -HEMISPHERE_LAT)[0]
    w_cap = np.cos(np.deg2rad(lat[cap_rows]))[:, None] * np.ones((1, N_LON))
    w_hem = np.cos(np.deg2rad(lat[hem_rows]))[:, None] * np.ones((1, N_LON))
    cap_m = np.array([np.average(anom_m[i][cap_rows], weights=w_cap) for i in range(len(LEVELS))])
    cap_s = np.array([np.average(anom_s[i][cap_rows], weights=w_cap) for i in range(len(LEVELS))])
    hemi_s = np.array([np.average(anom_s[i][hem_rows], weights=w_hem) for i in range(len(LEVELS))])
    return cap_m, cap_s, hemi_s, cap_s - hemi_s


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def already_published(previous: dict | None, run: datetime, force: bool) -> bool:
    if force or not previous:
        return False
    if previous.get("run_utc") != _iso(run):
        return False
    headline = previous.get("headline", "")
    # Republish when the southern map is missing or the scoreboard still says "by".
    if not previous.get("image_south"):
        return False
    if " by " in headline:
        return False
    return True


def _truthy(value: str | bool | None) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish polar-vortex coupling maps")
    parser.add_argument("--model", default=os.environ.get("MODEL", "both"),
                        choices=["aifs-ens", "ifs-ens", "both"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--font-dir", default=str(Path(__file__).resolve().parents[1] / "assets" / "fonts"))
    parser.add_argument("--page", default=str(Path(__file__).resolve().parents[1] / "web" / "index.html"))
    return parser.parse_args()


def _models(name: str) -> list[str]:
    if name == "both":
        return ["ifs-ens", "aifs-ens"]
    return [name]


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    workers = int(os.environ.get("WORKERS", "8"))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers * 2))
    return session, workers


def _http_get(session, url_for, headers=None, expect_len=None, tries=8) -> bytes:
    import requests
    source = os.environ.get("SOURCE", "gcs")
    fallback = "aws" if source == "gcs" else "gcs"
    slots = {"gcs": threading.Semaphore(int(os.environ.get("WORKERS", "8"))), "aws": threading.Semaphore(4)}
    last = "no response"
    for attempt in range(tries):
        throttled = False
        for src in (source, fallback):
            try:
                with slots[src]:
                    response = session.get(url_for(src), headers=headers, timeout=120)
                if response.status_code in (200, 206):
                    body = response.content
                    if expect_len is None or len(body) == expect_len:
                        BYTES_READ["ecmwf"] += len(body)
                        return body
                    last = f"{src}: got {len(body)} bytes, expected {expect_len}"
                else:
                    last = f"{src}: HTTP {response.status_code}"
                    throttled = throttled or response.status_code in (429, 503)
            except requests.RequestException as exc:
                last = f"{src}: {type(exc).__name__}"
        time.sleep(min(2 ** attempt, 30) + (5 if throttled else 0))
    raise RuntimeError(f"download failed ({last})")


def _ens_url(model: str, run: datetime, kind: str, ext: str, source: str) -> str:
    day, hour = run.strftime("%Y%m%d"), run.strftime("%H")
    root = MODEL_CFG[model]["root"]
    return f"{BASES[source]}/{day}/{hour}z/{root}/{day}{hour}0000-{LAST_HOUR}h-enfo-{kind}.{ext}"


def _run_is_complete(session, model: str, run: datetime) -> bool:
    import requests
    source = os.environ.get("SOURCE", "gcs")
    fallback = "aws" if source == "gcs" else "gcs"
    for src in (source, fallback):
        try:
            if all(
                session.head(_ens_url(model, run, kind, "index", src), timeout=30).status_code == 200
                for kind in MODEL_CFG[model]["kinds"]
            ):
                return True
        except requests.RequestException:
            continue
    return False


def resolve_run(session, model: str, now: datetime) -> datetime:
    expected = expected_run(model, now)
    wait_minutes = float(os.environ.get("WAIT_MINUTES", "50"))
    poll_seconds = float(os.environ.get("POLL_SECONDS", "60"))
    deadline = time.time() + wait_minutes * 60
    while True:
        if _run_is_complete(session, model, expected):
            print(f"{model} {expected:%Y-%m-%d %H}Z is complete to +{LAST_HOUR} h", flush=True)
            return expected
        if time.time() >= deadline:
            break
        print(f"{model} {expected:%Y-%m-%d %H}Z is not complete yet; checking again", flush=True)
        time.sleep(poll_seconds)
    cycle = now.replace(minute=0, second=0, microsecond=0)
    cycle = cycle.replace(hour=(cycle.hour // 6) * 6)
    for k in range(1, 16):
        run = cycle - timedelta(hours=6 * k)
        if run.hour not in CYCLES[model] or run >= expected:
            continue
        if _run_is_complete(session, model, run):
            print(
                f"{model} {expected:%Y-%m-%d %H}Z did not arrive; using {run:%Y-%m-%d %H}Z",
                flush=True,
            )
            return run
    raise RuntimeError(f"no complete {model} run found")


def _decode_message(buf: bytes) -> dict[str, np.ndarray]:
    import eccodes
    handle = eccodes.codes_new_from_message(buf)
    try:
        ni = eccodes.codes_get(handle, "Ni")
        nj = eccodes.codes_get(handle, "Nj")
        lon0 = eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
        lat0 = eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
        dlon = eccodes.codes_get(handle, "iDirectionIncrementInDegrees")
        vals = eccodes.codes_get_values(handle).reshape(nj, ni).astype(np.float32)
    finally:
        eccodes.codes_release(handle)
    if lat0 < 0:
        vals = vals[::-1]
    shift = int(round((lon0 % 360) / dlon)) % ni
    vals = np.roll(vals, shift, axis=1)
    return subsample_quarter_degree(vals)


def _ensemble_mean(session, model: str, run: datetime) -> tuple[dict[str, np.ndarray], int]:
    import concurrent.futures as futures
    from tqdm.auto import tqdm

    cfg = MODEL_CFG[model]
    jobs = []
    for kind in cfg["kinds"]:
        raw = _http_get(session, lambda src, kind=kind: _ens_url(model, run, kind, "index", src))
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        jobs += [
            (kind, row) for row in rows
            if row.get("param") == cfg["param"] and row.get("levtype") == "pl"
            and int(row["levelist"]) in LEVELS
        ]
    counts = {lev: sum(1 for _, row in jobs if int(row["levelist"]) == lev) for lev in LEVELS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"{model}: uneven members per level {counts}")
    n_members = counts[LEVELS[0]]
    sums = {hem: np.zeros((len(LEVELS), N_LAT, N_LON), np.float64) for hem in HEMISPHERES}

    def fetch(job):
        kind, row = job
        offset, length = row["_offset"], row["_length"]
        buf = _http_get(
            session,
            lambda src, kind=kind: _ens_url(model, run, kind, "grib2", src),
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
            expect_len=length,
        )
        fields = _decode_message(buf)
        del buf
        scale = cfg["to_gpm"]
        return LEVELS.index(int(row["levelist"])), fields["north"] * scale, fields["south"] * scale

    workers = int(os.environ.get("WORKERS", "8"))
    with futures.ThreadPoolExecutor(workers) as pool:
        for level_i, north, south in tqdm(
            pool.map(fetch, jobs), total=len(jobs), desc=f"{cfg['short']} members"
        ):
            sums["north"][level_i] += north
            sums["south"][level_i] += south
    means = {hem: (sums[hem] / n_members).astype(np.float32) for hem in HEMISPHERES}
    del sums
    return means, n_members


def era5_time_index(stamp: datetime, size: int) -> int:
    """Hour offset from 1959-01-01 00:00 on the ARCO hourly axis."""
    hours = int((stamp - datetime(1959, 1, 1)).total_seconds() // 3600)
    if hours < 0 or hours >= size:
        raise RuntimeError(
            f"ERA5 time {stamp:%Y-%m-%d %H}Z is outside the ARCO record ({size} hours)"
        )
    return hours


def _arco():
    import fsspec
    import zarr
    # readahead keeps blocks in memory. filecache would write the chunks to disk.
    # xarray.open_zarr is not used: current xarray passes zarr_format, which zarr 2 rejects.
    store = fsspec.get_mapper(ARCO_URL, cache_type="readahead")
    group = zarr.open_consolidated(store, mode="r")
    units = str(group["time"].attrs.get("units", ""))
    if units != "hours since 1959-01-01 00:00:00":
        raise RuntimeError(f"unexpected ARCO time units: {units}")
    size = int(group["time"].shape[0])
    first, last = int(group["time"][0]), int(group["time"][-1])
    if first != 0 or last != size - 1:
        raise RuntimeError(
            f"ARCO time axis is not hourly from 1959-01-01 ({first} … {last}, {size} hours)"
        )
    levels = [int(v) for v in np.asarray(group["level"][:]).tolist()]
    missing = [level for level in LEVELS if level not in levels]
    if missing:
        raise RuntimeError(f"ARCO level axis is missing {missing}")
    return group, size, [levels.index(level) for level in LEVELS]


def _era5_time(year: int, month: int, day: int, hour: int) -> datetime:
    if month == 2 and day == 29 and not calendar.isleap(year):
        day = 28
    return datetime(year, month, day, hour)


def _read_era5(group, level_idx, ti: int, tries: int = 4) -> tuple[np.ndarray, np.ndarray]:
    arr = group["geopotential"]
    for attempt in range(tries):
        try:
            north = np.asarray(arr[ti, :, :361, :], dtype=np.float32)[level_idx][:, ::2, ::2] / G0
            south = np.asarray(arr[ti, :, 360:, :], dtype=np.float32)[level_idx][:, ::2, ::2] / G0
            BYTES_READ["arco"] += north.nbytes + south.nbytes
            if north.shape != (len(LEVELS), N_LAT, N_LON) or south.shape != north.shape:
                raise ValueError(f"unexpected ERA5 shape {north.shape} {south.shape}")
            if not np.isfinite(north).all() or not np.isfinite(south).all():
                raise ValueError(f"ERA5 time index {ti} returned missing values")
            return north, south
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("ERA5 read failed")


def _climatology(valid: datetime) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import concurrent.futures as futures
    from tqdm.auto import tqdm
    if valid.hour % 6 != 0:
        raise RuntimeError("ERA5 climatology is read at 6-hourly valid times")
    group, n_times, level_idx = _arco()
    years = list(range(CLIM_START, CLIM_END + 1))
    indexes = [
        era5_time_index(_era5_time(year, valid.month, valid.day, valid.hour), n_times)
        for year in years
    ]
    sums = {hem: np.zeros((len(LEVELS), N_LAT, N_LON), np.float64) for hem in HEMISPHERES}
    squares = {hem: np.zeros_like(sums[hem]) for hem in HEMISPHERES}
    workers = int(os.environ.get("CLIM_WORKERS", "8"))
    started = time.time()
    with futures.ThreadPoolExecutor(workers) as pool:
        for north, south in tqdm(
            pool.map(lambda ti: _read_era5(group, level_idx, ti), indexes),
            total=len(indexes), desc=f"ERA5 {valid:%b %d %HZ}",
        ):
            for hem, field in (("north", north), ("south", south)):
                sums[hem] += field
                squares[hem] += field.astype(np.float64) ** 2
            del north, south
    n = len(indexes)
    out = {}
    for hem in HEMISPHERES:
        mean = sums[hem] / n
        std = np.sqrt(np.maximum((squares[hem] - sums[hem] ** 2 / n) / (n - 1), 0))
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise RuntimeError(f"the {valid:%b %d %HZ} {hem} climatology contains missing values")
        out[hem] = (mean.astype(np.float32), std.astype(np.float32))
    del sums, squares
    print(
        f"  ARCO-ERA5 {valid:%Y-%m-%d %H}Z read in {(time.time() - started) / 60:.1f} min "
        f"({BYTES_READ['arco'] / 1e6:.0f} MB in memory)",
        flush=True,
    )
    return out


def _anomalies(mean_h: np.ndarray, clim_mean: np.ndarray, clim_std: np.ndarray, hemisphere: str) -> dict:
    lat = hemisphere_lat(hemisphere)
    anom_m = mean_h - clim_mean
    anom_s = anom_m / np.maximum(clim_std, STD_FLOOR)
    cap_m, cap_s, hemi_s, rel_s = _cap_stats(anom_m, anom_s, lat, hemisphere)
    if not all(np.isfinite(arr).all() for arr in (cap_m, cap_s, hemi_s, rel_s, anom_s)):
        raise RuntimeError(f"non-finite {hemisphere} anomaly; nothing will be drawn or published")
    return dict(
        height=_smooth_roll(mean_h),
        sigma=_smooth_roll(anom_s),
        cap_m=cap_m,
        cap_sigma=cap_s,
        hemi_sigma=hemi_s,
        rel_sigma=rel_s,
    )


def _draw_still(result: dict, hemisphere: str, valid: datetime, run: datetime, cfg: dict,
                path: Path, font_dir: Path, coasts: bool = True) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Polygon

    font_bold, font_light = _font_props(Path(font_dir))
    cmap, norm = _colormap()
    figsize = (10, 12.5)
    dpi = STILL_WIDTH_PX / figsize[0]
    title_size, legend_tick, legend_caption, panel_label, credit_size = 29, 23, 21, 21, 11
    panel_note = 14
    panel_w, panel_h = 0.42, 0.336
    rects = [
        [0.05, 0.451, panel_w, panel_h], [0.53, 0.451, panel_w, panel_h],
        [0.05, 0.075, panel_w, panel_h], [0.53, 0.075, panel_w, panel_h],
    ]
    lat = hemisphere_lat(hemisphere)
    lat_asc = lat[::-1]
    lon180 = np.append(np.arange(N_LON) * 0.5 - 180.0, 180.0)
    bg, ink, muted = "white", "black", "#555b66"
    cold_caption, warm_caption = "#2f73b5", "#d23a24"
    res = result[hemisphere]
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg)
    fig.text(
        0.5, 0.988, f"Stratosphere-troposphere heights on {ap_date(valid)}",
        ha="center", va="top", fontproperties=font_bold, fontsize=title_size, color=ink,
    )
    cax = fig.add_axes([0.25, 0.880, 0.5, 0.0192])
    cax.imshow(
        cmap(np.linspace(0, 1, 1024))[None, :, :3], extent=[0, 1, 0, 1],
        aspect="auto", interpolation="bilinear", zorder=1,
    )
    cap = 0.035
    for x0, x1, col in ((0, -cap, cmap(0.0)), (1, 1 + cap, cmap(1.0))):
        cax.add_patch(Polygon(
            [(x0, 0), (x0, 1), (x1, 0.5)], closed=True, facecolor=col, edgecolor=col,
            linewidth=0.6, joinstyle="round", clip_on=False, zorder=2,
        ))
    cax.add_patch(Polygon(
        [(-cap, 0.5), (0, 1), (1, 1), (1 + cap, 0.5), (1, 0), (0, 0)], closed=True,
        facecolor="none", edgecolor="#b9bec8", linewidth=0.8, joinstyle="round",
        clip_on=False, zorder=3,
    ))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.set_yticks([])
    cax.set_xticks([float(norm(tick)) for tick in SIGMA_TICKS])
    cax.set_xticklabels([("+" if tick > 0 else "−" if tick < 0 else "") + f"{abs(tick):g}" for tick in SIGMA_TICKS])
    cax.tick_params(axis="x", length=3, width=0.8, color=muted, labelsize=legend_tick, labelcolor=ink, pad=4)
    for label in cax.get_xticklabels():
        label.set_fontproperties(font_light)
        label.set_fontsize(legend_tick)
        label.set_color(ink)
    fig.text(0.805, 0.8595, "std. dev.", ha="left", va="center", fontproperties=font_light,
             fontsize=legend_caption, color=muted)
    fig.text(0.25, 0.904, "Lower than average", ha="left", va="bottom", fontproperties=font_bold,
             fontsize=legend_caption, color=cold_caption)
    fig.text(0.75, 0.904, "Higher", ha="right", va="bottom", fontproperties=font_bold,
             fontsize=legend_caption, color=warm_caption)

    y0, y1 = (0, 90) if hemisphere == "north" else (-90, 0)
    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "50m", facecolor="none") if coasts else None
    for i, level in enumerate(LEVELS):
        ax = fig.add_axes(rects[i], projection=polar_projection(hemisphere), zorder=1)
        ax.set_extent(polar_extent(hemisphere), ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.set_boundary(mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5), transform=ax.transAxes)
        ax.set_facecolor(bg)
        ax.spines["geo"].set_edgecolor("#c8ccd4")
        ax.spines["geo"].set_linewidth(1.4)
        ax.imshow(
            res["sigma"][i], extent=[-180, 180, y0, y1], origin="upper", cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(), interpolation="bilinear", regrid_shape=900, zorder=100,
        )
        height = res["height"][i]
        wrapped = np.concatenate([height, height[:, :1]], axis=1)[::-1]
        step = CONTOUR_STEP_M[level]
        contours = np.arange(math.floor(height.min() / step) * step, height.max() + step, step)
        ax.contour(
            lon180, lat_asc, wrapped, levels=contours, colors="#2b2f36", linewidths=0.9,
            alpha=0.7, transform=ccrs.PlateCarree(), zorder=110,
        )
        if coastline is not None:
            ax.add_feature(coastline, edgecolor="#2b2f36", linewidth=0.7, alpha=0.55, zorder=111)
        ax.gridlines(
            xlocs=range(-180, 181, 60), ylocs=range(-60, 90, 30), color="black",
            alpha=0.10, linewidth=0.5, zorder=105,
        )
        label, note = PANEL_INFO[level]
        x, y, w, h = rects[i]
        top = y + h
        fig.text(x, top + 0.024, label, ha="left", va="bottom", fontproperties=font_bold,
                 fontsize=panel_label, color=ink)
        fig.text(x, top + 0.006, note, ha="left", va="bottom", fontproperties=font_light,
                 fontsize=panel_note, color=muted)
        value = res["cap_sigma"][i]
        fig.text(
            x + w, top + 0.024, f"{value:+.1f}".replace("-", "\u2212"), ha="right", va="bottom",
            fontproperties=font_bold, fontsize=panel_label,
            color=warm_caption if value >= 1 else cold_caption if value <= -1 else ink,
        )
        fig.text(x + w, top + 0.006, "polar cap", ha="right", va="bottom", fontproperties=font_light,
                 fontsize=panel_note, color=muted)
    fig.text(
        0.04, 0.012,
        f"Data: ECMWF {cfg['short']} ({ap_date(run)} {run:%H}Z run)/ERA5. Lines: actual height",
        ha="left", va="bottom", fontproperties=font_light, fontsize=credit_size, color=muted,
    )
    fig.text(
        0.96, 0.012, "BEN NOLL/THE WASHINGTON POST", ha="right", va="bottom",
        fontproperties=font_light, fontsize=credit_size, color=muted,
    )
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)


def _diagnostics(models: dict, runs: dict, valids: dict) -> str:
    thresholds = ", ".join(f"{level} hPa +{COUPLING_THRESHOLDS.get(level, 0.5):g}" for level in LEVELS)
    lines = [
        f"Stratosphere-troposphere coupling check | step +{LAST_HOUR} h | ensemble-mean geopotential height",
        "Anomaly = ensemble mean - ERA5 %d-%d mean for the same calendar day/hour; sigma = that anomaly / the ERA5 interannual std."
        % (CLIM_START, CLIM_END),
        f"Northern cap = {CAP_LAT}-90N mean, hemisphere = {HEMISPHERE_LAT}-90N mean (area-weighted); relative = cap sigma - hemisphere sigma.",
        "The RELATIVE value is what is rated: it removes hemisphere-wide warmth so only a polar-specific signal counts.",
        f"Rating: from the top level down, count the levels in a row where relative >= the threshold ({thresholds}):",
        "  stratosphere only = no coupling, reaches 100 hPa = weak, 300 hPa = moderate, 500 hPa = strong.",
        "The southern map is the same four levels, centered on the South Pole with New Zealand at the top. It is not part of the rating.",
        "A stratospheric warming usually reaches the troposphere one to three weeks later; a day-15 ensemble mean also damps the signal.",
        "",
    ]
    for model, result in models.items():
        cfg = MODEL_CFG[model]
        north = result["north"]
        rating = result["rating"]
        lines.append(
            f"{cfg['name']} ({result['n']} members): run {runs[model]:%Y-%m-%d %H}Z, "
            f"valid {valids[model]:%Y-%m-%d %H}Z"
        )
        lines.append(f"  {'level':>8}  {'cap m':>7}  {'cap sigma':>9}  {'hemi sigma':>10}  {'relative':>8}  {'needs':>6}  counts")
        for i, level in enumerate(LEVELS):
            need = COUPLING_THRESHOLDS.get(level, 0.5)
            lines.append(
                f"  {level:>5} hPa  {north['cap_m'][i]:+7.0f}  {north['cap_sigma'][i]:+9.2f}  "
                f"{north['hemi_sigma'][i]:+10.2f}  {north['rel_sigma'][i]:+8.2f}  {need:+6.1f}  "
                f"{'yes' if north['rel_sigma'][i] >= need else 'no'}"
            )
        south = result["south"]["cap_sigma"]
        lines.append("  southern polar cap (sigma): " + " | ".join(
            f"{level} hPa {south[i]:+.2f}" for i, level in enumerate(LEVELS)
        ))
        lines.append(
            f"  Coupling: {coupling_headline(rating['rating'], valids[model])} - {rating['detail']}\n"
        )
    return "\n".join(lines) + "\n"


def _model_record(model: str, result: dict, run: datetime, valid: datetime, published: str) -> dict:
    cfg = MODEL_CFG[model]
    north = result["north"]
    rating = result["rating"]
    return dict(
        id=model, name=cfg["name"], members=result["n"], run_utc=_iso(run), valid_utc=_iso(valid),
        step_hours=LAST_HOUR, image=cfg["image"], image_south=cfg["image_south"], levels=LEVELS,
        cap_sigma=[round(float(v), 3) for v in north["cap_sigma"]],
        hemi_sigma=[round(float(v), 3) for v in north["hemi_sigma"]],
        relative_sigma=[round(float(v), 3) for v in north["rel_sigma"]],
        south_cap_sigma=[round(float(v), 3) for v in result["south"]["cap_sigma"]],
        thresholds={str(level): COUPLING_THRESHOLDS.get(level, 0.5) for level in LEVELS},
        rating=rating["rating"],
        headline=coupling_headline(rating["rating"], valid),
        detail=rating["detail"],
        reach_hpa=rating["reach"],
        cadence=_cadence(model),
        published_utc=published,
    )


def _cadence(model: str) -> str:
    hours = [f"{hour:02d}" for hour in CYCLES[model]]
    if len(hours) == 1:
        body = hours[0]
    else:
        body = ", ".join(hours[:-1]) + " and " + hours[-1]
    noun = "update" if len(hours) == 1 else "updates"
    return f"{len(hours)} {noun} a day ({body} UTC runs)"


def _previous(public_url: str) -> dict:
    import requests
    try:
        response = requests.get(public_url + "latest.json", timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def _merge_latest(previous: dict, records: list[dict], generated: str) -> dict:
    by_id = {item["id"]: item for item in previous.get("models", []) if item.get("id")}
    for record in records:
        by_id[record["id"]] = record
    order = ["ifs-ens", "aifs-ens"]
    return dict(
        generated_utc=generated,
        levels=LEVELS,
        cap_lat=CAP_LAT,
        hemisphere_lat=HEMISPHERE_LAT,
        south_top_lon=SOUTH_TOP_LON,
        thresholds={str(level): COUPLING_THRESHOLDS.get(level, 0.5) for level in LEVELS},
        diagnostics_url="coupling_diagnostics.txt",
        models=[by_id[model] for model in order if model in by_id],
    )


def _s3_cp(src: Path, dest: str, content_type: str) -> None:
    if src.suffix.lower() not in UPLOAD_SUFFIXES:
        raise RuntimeError(f"refusing to upload {src.name}")
    lowered = dest.lower()
    if any(token in lowered for token in (".grib", ".npz", ".nc", ".zarr")):
        raise RuntimeError(f"refusing to upload downloaded data to {dest}")
    env = os.environ.copy()
    env.pop("AWS_PROFILE", None)
    if not env.get("AWS_CA_BUNDLE"):
        env.pop("AWS_CA_BUNDLE", None)
    if not env.get("REQUESTS_CA_BUNDLE"):
        env.pop("REQUESTS_CA_BUNDLE", None)
    subprocess.run(
        ["aws", "s3", "cp", str(src), dest, "--content-type", content_type,
         "--cache-control", "no-cache, max-age=0", "--only-show-errors"],
        check=True, env=env,
    )
    print(f"Uploaded {src.name} -> {dest}", flush=True)


def _write_status(path: Path, product: str, message: str, outputs: list[str], started_utc: str, schedule: str) -> None:
    payload = {
        "schema_version": 1,
        "product": product,
        "status": "success",
        "scheduled_event": os.getenv("GITHUB_EVENT_NAME") == "schedule",
        "event_name": os.getenv("GITHUB_EVENT_NAME", ""),
        "schedule_utc": schedule,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": 0,
        "workflow_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "workflow_url": (
            f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.getenv('GITHUB_REPOSITORY', '')}/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
        ),
        "repository": os.getenv("GITHUB_REPOSITORY", ""),
        "commit_sha": os.getenv("GITHUB_SHA", ""),
        "branch": os.getenv("GITHUB_REF_NAME", ""),
        "outputs": outputs,
        "message": message,
    }
    # The shell records when the process started. A manual run leaves this at 0.
    started = os.environ.get("STARTED_EPOCH")
    if started:
        payload["runtime_seconds"] = max(0, int(time.time() - float(started)))
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run(args: argparse.Namespace, work: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    session, _workers = _session()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    public_url = os.environ.get(
        "PUBLIC_URL", "https://bennollweather.s3.amazonaws.com/climate_imagery/strat_coupling/"
    )
    if not public_url.endswith("/"):
        public_url += "/"
    s3_prefix = os.environ.get("S3_PREFIX", "s3://bennollweather/climate_imagery/strat_coupling/")
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    previous = _previous(public_url)
    previous_models = {item["id"]: item for item in previous.get("models", [])}
    force = args.force or _truthy(os.environ.get("FORCE"))
    upload = not args.no_upload and _truthy(os.environ.get("UPLOAD_TO_S3", "true"))
    selected = []
    for model in _models(args.model):
        run_time = resolve_run(session, model, now)
        if already_published(previous_models.get(model), run_time, force):
            print(f"{model} {run_time:%Y-%m-%d %H}Z is already published", flush=True)
            continue
        selected.append((model, run_time))
    if not selected:
        print("Nothing new to publish. No ECMWF or ARCO data was downloaded.", flush=True)
        return

    out = work / "out"
    results = {}
    runs = {}
    valids = {}
    clim_cache: dict[datetime, dict] = {}
    for model, run_time in selected:
        print(f"Reading {MODEL_CFG[model]['name']} {run_time:%Y-%m-%d %H}Z into memory", flush=True)
        means, n_members = _ensemble_mean(session, model, run_time)
        valid = run_time + timedelta(hours=LAST_HOUR)
        if valid not in clim_cache:
            clim_cache[valid] = _climatology(valid)
        packed = {"n": n_members}
        for hemisphere in HEMISPHERES:
            clim_mean, clim_std = clim_cache[valid][hemisphere]
            packed[hemisphere] = _anomalies(means[hemisphere], clim_mean, clim_std, hemisphere)
        packed["rating"] = rate_coupling(packed["north"]["rel_sigma"])
        results[model] = packed
        runs[model] = run_time
        valids[model] = valid
        print(
            f"  {coupling_headline(packed['rating']['rating'], valid)} — {packed['rating']['detail']}",
            flush=True,
        )
        del means
    # Drop the climatology arrays before drawing so the peak is not held twice.
    clim_cache.clear()

    for model in results:
        cfg = MODEL_CFG[model]
        for hemisphere, key in (("north", "image"), ("south", "image_south")):
            dest = out / cfg[key]
            _draw_still(results[model], hemisphere, valids[model], runs[model], cfg, dest, args.font_dir)
            print(f"Drew {dest.name}", flush=True)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [
        _model_record(model, results[model], runs[model], valids[model], generated)
        for model in results
    ]
    latest = _merge_latest(previous, records, generated)
    latest_path = out / "latest.json"
    latest_path.write_text(json.dumps(latest, indent=1) + "\n")
    diagnostics = out / "coupling_diagnostics.txt"
    # Diagnostics describe the models drawn in this run. The page reads latest.json.
    diagnostics.write_text(_diagnostics(results, runs, valids))
    print(
        f"ECMWF bytes read into memory: {BYTES_READ['ecmwf'] / 1e6:.0f} MB. "
        f"ARCO-ERA5 arrays: {BYTES_READ['arco'] / 1e6:.0f} MB. None of that was written to disk.",
        flush=True,
    )
    if not upload:
        print("UPLOAD_TO_S3 is off. Local renders stay in the work directory until it is deleted.", flush=True)
        return

    for model in results:
        cfg = MODEL_CFG[model]
        _s3_cp(out / cfg["image"], s3_prefix + cfg["image"], "image/png")
        _s3_cp(out / cfg["image_south"], s3_prefix + cfg["image_south"], "image/png")
    _s3_cp(diagnostics, s3_prefix + "coupling_diagnostics.txt", "text/plain; charset=utf-8")
    _s3_cp(latest_path, s3_prefix + "latest.json", "application/json")
    _s3_cp(Path(args.page), s3_prefix + "index.html", "text/html; charset=utf-8")

    status_prefix = os.environ.get(
        "STATUS_S3_PREFIX", "s3://bennollweather/climate_imagery/status"
    ).rstrip("/")
    started_utc = os.environ.get("STARTED_UTC", generated)
    for model in results:
        cfg = MODEL_CFG[model]
        status_path = out / f"{cfg['product_id']}.json"
        outputs = [
            s3_prefix + cfg["image"],
            s3_prefix + cfg["image_south"],
            s3_prefix + "latest.json",
        ]
        _write_status(
            status_path, cfg["product_id"],
            f"Published {cfg['name']} northern and southern maps for {runs[model]:%Y-%m-%d %H}Z. "
            "ECMWF and ARCO downloads were not stored.",
            outputs, started_utc, cfg["schedule"],
        )
        _s3_cp(status_path, f"{status_prefix}/{cfg['product_id']}.json", "application/json")
    print(f"Dashboard: {public_url}index.html", flush=True)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    started = time.time()
    work = Path(os.environ.get("WORK_DIR", "/tmp/strat-coupling"))
    prepared = False
    try:
        work = prepare_work(work)
        prepared = True
        run(args, work)
    finally:
        if prepared:
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass
            leaked = sweep_leaked_downloads(started, work)
            removed = delete_work_tree(work)
            if work.exists():
                raise RuntimeError(f"working directory was not deleted: {work}")
            print(
                f"Deleted {removed} working files and {leaked} leaked download files. "
                "ECMWF and ARCO data from this run is gone.",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
