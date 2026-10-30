"""Checks that do not download ECMWF or ARCO data."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import coupling  # noqa: E402


def test_scoreboard_says_on_not_by():
    text = coupling.coupling_headline("none", datetime(2026, 11, 13, 12))
    assert text == "No coupling predicted on Nov. 13"
    assert " by " not in text
    for rating in ("weak", "moderate", "strong"):
        line = coupling.coupling_headline(rating, datetime(2026, 3, 4))
        assert line.endswith(" on March 4")
        assert " by " not in line


def test_expected_cycles_match_the_schedule():
    # AIFS-ENS fires at 01:55, 07:55, 13:55, 19:55 UTC, about 7 h 55 m after the cycle.
    assert coupling.expected_run("aifs-ens", datetime(2026, 10, 30, 7, 55)) == datetime(2026, 10, 30, 0)
    assert coupling.expected_run("aifs-ens", datetime(2026, 10, 30, 13, 55)) == datetime(2026, 10, 30, 6)
    assert coupling.expected_run("aifs-ens", datetime(2026, 10, 30, 19, 55)) == datetime(2026, 10, 30, 12)
    assert coupling.expected_run("aifs-ens", datetime(2026, 10, 30, 1, 55)) == datetime(2026, 10, 29, 18)
    # EPS fires at 08:35 and 20:35 UTC, about 8 h 35 m after 00Z and 12Z.
    assert coupling.expected_run("ifs-ens", datetime(2026, 10, 30, 8, 35)) == datetime(2026, 10, 30, 0)
    assert coupling.expected_run("ifs-ens", datetime(2026, 10, 30, 20, 35)) == datetime(2026, 10, 30, 12)
    # EPS has no 06Z or 18Z day-15 run.
    assert coupling.expected_run("ifs-ens", datetime(2026, 10, 30, 16, 0)) == datetime(2026, 10, 30, 0)


def test_era5_index_is_hours_since_1959():
    assert coupling.era5_time_index(datetime(1959, 1, 1), 552264) == 0
    assert coupling.era5_time_index(datetime(1959, 1, 1, 18), 552264) == 18
    # 1959-01-01 through 2021-12-31 is 552264 hourly steps.
    assert coupling.era5_time_index(datetime(2021, 12, 31, 23), 552264) == 552263
    # Feb. 29 in a non-leap year uses Feb. 28, which is inside the record.
    stamp = coupling._era5_time(1991, 2, 29, 12)
    assert stamp == datetime(1991, 2, 28, 12)
    assert 0 < coupling.era5_time_index(stamp, 552264) < 552264
    try:
        coupling.era5_time_index(datetime(2022, 1, 1), 552264)
    except RuntimeError:
        return
    raise AssertionError("a time after the ARCO record should be rejected")


def test_half_degree_split_keeps_both_poles():
    lat = np.linspace(90, -90, 721)[:, None] * np.ones((1, 1440))
    parts = coupling.subsample_quarter_degree(lat.astype(np.float32))
    assert parts["north"].shape == (181, 720)
    assert parts["south"].shape == (181, 720)
    assert parts["north"][0, 0] == 90
    assert parts["north"][-1, 0] == 0
    assert parts["south"][0, 0] == 0
    assert parts["south"][-1, 0] == -90


def test_same_run_without_a_southern_map_is_not_skipped():
    previous = {
        "run_utc": "2026-10-29T18:00:00Z",
        "headline": "No coupling predicted by Nov. 13",
        "image_south": "",
    }
    assert coupling.already_published(previous, datetime(2026, 10, 29, 18), force=False) is False
    done = dict(previous, headline="No coupling predicted on Nov. 13", image_south="south.png")
    assert coupling.already_published(done, datetime(2026, 10, 29, 18), force=False) is True


def test_work_directory_is_removed_with_its_downloads():
    work = Path("/tmp/strat-coupling-test-delete")
    if work.exists():
        coupling.delete_work_tree(work)
    coupling.prepare_work(work)
    (work / "out" / "member.grib2").write_bytes(b"not-a-real-grib")
    (work / "out" / "clim.npz").write_bytes(b"not-a-real-archive")
    removed = coupling.delete_work_tree(work)
    assert removed == 2
    assert not work.exists()


def test_home_directory_is_refused():
    try:
        coupling.prepare_work(Path.home() / "strat-coupling-work")
    except RuntimeError:
        return
    raise AssertionError("a home path was accepted")


def test_new_zealand_is_at_the_top_of_the_south_pole_map():
    south = coupling.polar_projection("south")
    north = coupling.polar_projection("north")
    import cartopy.crs as ccrs
    # The South Pole stays at the center.
    pole_x, pole_y = south.transform_point(174, -89.9, ccrs.Geodetic())
    assert abs(pole_x) < 50_000 and abs(pole_y) < 50_000
    nz_x, nz_y = south.transform_point(174, -41, ccrs.Geodetic())
    assert nz_y > 1_000_000
    assert abs(nz_x) < abs(nz_y) * 0.05
    # North America stays toward the bottom of the northern map.
    na_x, na_y = north.transform_point(-100, 40, ccrs.Geodetic())
    assert na_y < -1_000_000
    assert abs(na_x) < abs(na_y) * 0.05


def test_southern_map_draws_without_touching_the_network(tmp_path=None):
    work = Path("/tmp/strat-coupling-test-draw")
    if work.exists():
        coupling.delete_work_tree(work)
    coupling.prepare_work(work)
    try:
        rng = np.random.default_rng(0)
        height = 5000 + rng.normal(0, 30, size=(4, coupling.N_LAT, coupling.N_LON)).astype(np.float32)
        sigma = rng.normal(0, 0.4, size=height.shape).astype(np.float32)
        cap = np.array([0.2, -0.4, 1.2, 0.0], np.float32)
        result = {
            "south": dict(height=height, sigma=sigma, cap_sigma=cap),
            "north": dict(height=height, sigma=sigma, cap_sigma=cap),
        }
        cfg = coupling.MODEL_CFG["aifs-ens"]
        dest = work / "out" / "south.png"
        coupling._draw_still(
            result, "south", datetime(2026, 11, 13, 18), datetime(2026, 10, 29, 18),
            cfg, dest, ROOT / "assets" / "fonts", coasts=False,
        )
        assert dest.exists() and dest.stat().st_size > 1000
    finally:
        coupling.delete_work_tree(work)
        assert not work.exists()
