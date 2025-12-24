#!/usr/bin/env python3
"""
Sort a YouTube Music playlist by popularity (descending), then artist, then title.

Usage:
  python sort_ytmusic_playlist.py --playlist_id PLxxxx --credentials browser.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from ytmusicapi import YTMusic

LOG = logging.getLogger("playlist_sorter")


# ----------------------------- Data Model --------------------------------- #

@dataclass
class TrackStat:
    set_video_id: str
    video_id: str
    title: str
    artists: str
    base_views: int
    counterpart_video_id: Optional[str] = None
    counterpart_views: int = 0
    original_index: int = 0

    @property
    def combined_views(self) -> int:
        return self.base_views + self.counterpart_views

    @property
    def sort_key(self) -> Tuple[int, str, str, int, str]:
        # Sort by: Views (Desc), Artist (Asc), Title (Asc), Original Index (Asc), Unique ID (Asc)
        return (
            -self.combined_views,
            self.artists.casefold(),
            self.title.casefold(),
            self.original_index,
            self.set_video_id,
        )


# ----------------------------- Utilities ---------------------------------- #

_DIGITS_RE = re.compile(r"(\d+)")
# Supports: "123", "1,234", "123 views", "1.2K", "3.4M plays", "7B"
_HINT_RE = re.compile(
    r"^\s*([0-9][0-9,\.\s]*)\s*([KMBT])?\s*(?:views?|plays?)?\s*$",
    re.IGNORECASE,
)

_SUFFIX_MULTIPLIER = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}


def safe_int(x: Any) -> int:
    """Best-effort int conversion (does NOT interpret suffixes like 1.2M)."""
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return int(x)

    s = str(x).strip()
    if not s:
        return 0

    s_clean = s.replace(",", "")
    if s_clean.isdigit():
        return int(s_clean)

    m = _DIGITS_RE.search(s_clean)
    return int(m.group(1)) if m else 0


def parse_numeric_hint(raw: Any) -> int:
    """
    Parser for track hints, supporting K/M/B/T suffixes.
    Returns 0 if it can't be parsed cleanly.
    """
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)

    if not isinstance(raw, str):
        return 0

    s = raw.strip()
    if not s:
        return 0

    m = _HINT_RE.match(s)
    if not m:
        return 0

    num_part = (m.group(1) or "").strip().replace(" ", "").replace(",", "")
    suffix = (m.group(2) or "").upper().strip()

    try:
        if not num_part or num_part == ".":
            return 0
        value = float(num_part)
    except ValueError:
        return 0

    mult = _SUFFIX_MULTIPLIER.get(suffix, 1)
    # Truncate rather than round to avoid surprising inflation
    return int(value * mult) if value > 0 else 0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def timestamped_backup_name(playlist_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"backup_{playlist_id}_{ts}.json"


class RateLimiter:
    """Thread-safe global rate limiter."""
    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return

        sleep_for = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                sleep_for = self._next_time - now
            self._next_time = max(self._next_time, now) + self.min_interval

        if sleep_for > 0:
            time.sleep(sleep_for)


# ----------------------------- Logic Class -------------------------------- #

class PlaylistSorter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # Main-thread API instance (playlist edits + fetches)
        self.api = self._init_api()

        # Thread-local APIs for worker threads (stats fetches)
        self._tls = threading.local()
        self._creds_path = self.args.credentials.expanduser()

        self._view_cache: Dict[str, int] = {}
        self._cp_cache: Dict[str, Optional[str]] = {}
        self._view_lock = threading.Lock()
        self._cp_lock = threading.Lock()

        self.stats_delay: Optional[float] = args.stats_delay
        self.move_delay: Optional[float] = args.move_delay

        self._stats_limiter = RateLimiter(0.0)
        self._move_limiter = RateLimiter(0.0)

    def _init_api(self) -> YTMusic:
        creds = self.args.credentials.expanduser()
        if not creds.exists():
            LOG.critical("Credentials file not found: %s", creds)
            raise SystemExit(1)
        try:
            return YTMusic(str(creds))
        except Exception as e:
            LOG.critical("Failed to initialise YTMusic: %s", e)
            raise SystemExit(1)

    def _get_thread_api(self) -> YTMusic:
        """
        Lazily create a per-thread YTMusic instance.
        Tries to ensure a per-thread HTTP session where supported.
        """
        api = getattr(self._tls, "api", None)
        if api is not None:
            return api

        creds = self._creds_path
        if not creds.exists():
            LOG.critical("Credentials file not found: %s", creds)
            raise SystemExit(1)

        # Best effort: inject a dedicated requests.Session if constructor supports it.
        try:
            import requests  # type: ignore
            session = requests.Session()
            try:
                api = YTMusic(str(creds), requests_session=session)  # type: ignore
            except TypeError:
                api = YTMusic(str(creds))
        except Exception:
            api = YTMusic(str(creds))

        setattr(self._tls, "api", api)
        return api

    def _calc_auto_delays(self, playlist_len: int) -> None:
        if self.args.stats_delay is not None and self.args.move_delay is not None:
            return

        n = max(0, int(playlist_len))
        if n <= 80:
            s, m = 0.10, 0.40
        elif n <= 200:
            s, m = 0.20, 0.60
        elif n <= 500:
            s, m = 0.30, 0.80
        elif n <= 1000:
            s, m = 0.50, 1.00
        else:
            s, m = 0.80, 1.50

        if self.args.stats_delay is None:
            self.stats_delay = clamp(s, 0.05, 2.0)
        if self.args.move_delay is None:
            self.move_delay = clamp(m, 0.20, 3.0)

    def fetch_playlist_items(self) -> List[Dict[str, Any]]:
        try:
            pl = self.api.get_playlist(self.args.playlist_id, limit=None)
            tracks = pl.get("tracks", []) or []

            self._calc_auto_delays(len(tracks))
            stats_iv = self.stats_delay if self.stats_delay is not None else 0.1
            move_iv = self.move_delay if self.move_delay is not None else 0.5

            self._stats_limiter = RateLimiter(stats_iv)
            self._move_limiter = RateLimiter(move_iv)

            return tracks
        except Exception as e:
            LOG.critical("Failed to fetch playlist: %s", e)
            raise SystemExit(1)

    def _retry(self, context: str, fn):
        retries = max(0, int(self.args.retries))
        base = max(0.05, float(self.stats_delay if self.stats_delay is not None else 0.1))

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if attempt >= retries:
                    break
                sleep_time = (base * (2 ** attempt)) + random.uniform(0.0, base)
                time.sleep(sleep_time)

        LOG.warning("%s failed after %d retries: %s", context, retries, last_exc)
        return None

    def _get_single_view_count(self, video_id: str, track_hint: Optional[Dict[str, Any]] = None) -> int:
        if not video_id:
            return 0

        with self._view_lock:
            if video_id in self._view_cache:
                return self._view_cache[video_id]

        if track_hint:
            raw = track_hint.get("views") or track_hint.get("viewCount") or track_hint.get("plays")
            hinted = parse_numeric_hint(raw)
            if hinted > 0:
                with self._view_lock:
                    self._view_cache[video_id] = hinted
                return hinted

        def _call() -> int:
            self._stats_limiter.wait()
            api = self._get_thread_api()
            info = api.get_song(video_id)
            return safe_int(info.get("videoDetails", {}).get("viewCount", 0))

        val = self._retry("Views %s" % video_id, _call)
        views = safe_int(val)

        with self._view_lock:
            self._view_cache[video_id] = views
        return views

    def _get_counterpart_id(self, video_id: str) -> Optional[str]:
        if not video_id:
            return None

        with self._cp_lock:
            if video_id in self._cp_cache:
                return self._cp_cache[video_id]

        def _call() -> Optional[str]:
            self._stats_limiter.wait()
            api = self._get_thread_api()
            wp = api.get_watch_playlist(videoId=video_id, limit=1)
            tracks = wp.get("tracks", []) or []
            if not tracks:
                return None
            cp = tracks[0].get("counterpart", {}).get("videoId")
            return str(cp) if cp else None

        cp_id = self._retry("Counterpart %s" % video_id, _call)
        cp_id = str(cp_id) if cp_id else None

        with self._cp_lock:
            self._cp_cache[video_id] = cp_id
        return cp_id

    def process_track(self, track: Dict[str, Any], index: int) -> Optional[TrackStat]:
        svid = track.get("setVideoId")
        if not svid:
            return None

        vid = track.get("videoId") or ""

        artists = "Unknown Artist"
        artists_raw = track.get("artists") or []
        if isinstance(artists_raw, list) and artists_raw:
            names = [str(a.get("name")) for a in artists_raw if isinstance(a, dict) and a.get("name")]
            if names:
                artists = ", ".join(names)

        title = str(track.get("title") or "Untitled")

        # 1. Base Views
        base = self._get_single_view_count(str(vid), track_hint=track) if vid else 0

        # 2. Counterpart Views
        cp_views = 0
        cp_id = None
        if vid and (not self.args.base_only):
            cp_id = self._get_counterpart_id(str(vid))
            if cp_id and cp_id != vid:
                cp_views = self._get_single_view_count(cp_id)

        return TrackStat(
            set_video_id=str(svid),
            video_id=str(vid),
            title=title,
            artists=artists,
            base_views=base,
            counterpart_video_id=cp_id,
            counterpart_views=cp_views,
            original_index=index,
        )

    def gather_stats(self, tracks: List[Dict[str, Any]]) -> List[TrackStat]:
        n = len(tracks)
        if n == 0:
            return []

        workers = self.args.workers
        if workers <= 0:
            workers = 1
        workers = min(workers, n, 12)

        LOG.info(
            "Fetching stats for %d tracks (threads: %d, stats_delay: %.2fs)...",
            n, workers, float(self.stats_delay or 0.0),
        )

        results: List[TrackStat] = []

        if workers == 1:
            for i, t in tqdm(enumerate(tracks), total=n, unit="track", disable=self.args.no_progress):
                try:
                    stat = self.process_track(t, i)
                    if stat:
                        results.append(stat)
                except Exception as e:
                    LOG.warning("Worker error (index %d): %s", i, e)
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                futures = {executor.submit(self.process_track, t, i): i for i, t in enumerate(tracks)}
                for f in tqdm(as_completed(futures), total=n, unit="track", disable=self.args.no_progress):
                    try:
                        res = f.result()
                        if res:
                            results.append(res)
                    except Exception as e:
                        i = futures[f]
                        LOG.warning("Worker error (index %d): %s", i, e)
            except KeyboardInterrupt:
                print("\nStopping threads...")
                executor.shutdown(cancel_futures=True)
                raise

        return results

    def backup_playlist(self, current_tracks: List[Dict[str, Any]]) -> Optional[Path]:
        if self.args.no_backup:
            return None

        backup_dir = self.args.backup_dir.expanduser()
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            LOG.error("Could not create backup directory %s: %s", backup_dir, e)
            return None

        filename = backup_dir / timestamped_backup_name(self.args.playlist_id)

        data = {
            "playlist_id": self.args.playlist_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tracks": [
                {"setVideoId": t.get("setVideoId"), "videoId": t.get("videoId"), "title": t.get("title")}
                for t in current_tracks
            ],
        }

        try:
            filename.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            LOG.info("Backup saved to: %s", filename)
            return filename
        except Exception as e:
            LOG.error("Failed to write backup %s: %s", filename, e)
            return None

    def verify_order(self, desired_order: List[str]) -> bool:
        """Fetch playlist post-sort to verify order."""
        try:
            tracks = self.fetch_playlist_items()
            current_order = [t.get("setVideoId") for t in tracks if t.get("setVideoId")]
            return current_order == desired_order
        except Exception as e:
            LOG.warning("Could not verify playlist order: %s", e)
            return False

    @staticmethod
    def _extract_status_values(obj: Any) -> List[str]:
        statuses: List[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k.casefold() == "status" and isinstance(v, str):
                    statuses.append(v)
                statuses.extend(PlaylistSorter._extract_status_values(v))
        elif isinstance(obj, list):
            for item in obj:
                statuses.extend(PlaylistSorter._extract_status_values(item))
        return statuses

    @staticmethod
    def _edit_succeeded(resp: Any) -> bool:
        """
        Tolerant success check:
        - Accepts the standard status string "STATUS_SUCCEEDED".
        - If a dict is returned, searches for any "status" fields with a succeeded value.
        """
        if resp is None:
            return False
        if isinstance(resp, str):
            return resp == "STATUS_SUCCEEDED" or resp.upper().endswith("SUCCEEDED")
        if isinstance(resp, dict):
            statuses = PlaylistSorter._extract_status_values(resp)
            for s in statuses:
                if isinstance(s, str) and (s == "STATUS_SUCCEEDED" or s.upper().endswith("SUCCEEDED")):
                    return True
            return False
        return False

    def reorder(self, stats_sorted: List[TrackStat]) -> None:
        desired_order = [s.set_video_id for s in stats_sorted]

        print("Verifying playlist integrity before sorting...")
        current_tracks = self.fetch_playlist_items()
        current_order = [t.get("setVideoId") for t in current_tracks if t.get("setVideoId")]

        if len(current_order) != len(desired_order) or Counter(current_order) != Counter(desired_order):
            LOG.error("Playlist content changed during stats fetching. Aborting.")
            return

        if not self.args.dry_run:
            self.backup_playlist(current_tracks)

        pos: Dict[str, int] = {sv: i for i, sv in enumerate(current_order)}
        moves = 0

        pbar_desc = "Simulating" if self.args.dry_run else "Sorting"
        pbar = tqdm(
            total=max(0, len(current_order) - 1),
            desc=pbar_desc,
            unit="step",
            disable=self.args.no_progress,
        )

        for i in range(len(current_order) - 1):
            target_id = desired_order[i]

            if current_order[i] == target_id:
                pbar.update(1)
                continue

            j = pos.get(target_id)
            if j is None:
                LOG.error("Item %s missing from list! Aborting.", target_id)
                return

            if j < i:
                LOG.error("Logic error: index j=%d < i=%d. Playlist state inconsistent. Aborting.", j, i)
                return

            before_id = current_order[i]

            if self.args.dry_run:
                current_order.pop(j)
                current_order.insert(i, target_id)
                moves += 1
            else:
                try:
                    self._move_limiter.wait()
                    resp = self.api.edit_playlist(self.args.playlist_id, moveItem=(target_id, before_id))

                    if not self._edit_succeeded(resp):
                        raise RuntimeError("API returned failure response: %r" % (resp,))

                    current_order.pop(j)
                    current_order.insert(i, target_id)
                    moves += 1
                except Exception as e:
                    LOG.critical("Move failed for %s. Aborting: %s", target_id, e)
                    print("\nCRITICAL: API move failed. Playlist may be partially sorted. Aborting.")
                    return

            lo, hi = (i, j) if i <= j else (j, i)
            for k in range(lo, hi + 1):
                pos[current_order[k]] = k

            pbar.update(1)

        pbar.close()

        if self.args.dry_run:
            print("\nDry Run Complete. %d moves would be performed." % moves)
        else:
            print("\nDone. Performed %d moves." % moves)
            print("Verifying final order...")
            if self.verify_order(desired_order):
                print("SUCCESS: Playlist is correctly sorted.")
            else:
                print("WARNING: Playlist order verification failed. Items may have shifted.")


# ----------------------------- CLI --------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sort a YouTube Music playlist by popularity, then artist, then title.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--playlist_id", required=True, help="Playlist ID (PL...)")
    p.add_argument("--credentials", type=Path, default=Path("browser.json"), help="Auth file (browser.json)")

    p.add_argument("--base-only", action="store_true", help="Ignore counterpart views (faster)")
    p.add_argument("--workers", type=int, default=4, help="Threads for stats fetching (0 or 1 disables concurrency)")

    p.add_argument("--stats-delay", type=float, help="Minimum spacing between stats calls (seconds)")
    p.add_argument("--move-delay", type=float, help="Minimum spacing between move calls (seconds)")
    p.add_argument("--retries", type=int, default=3, help="Retry count for API calls")

    p.add_argument("--dry-run", action="store_true", help="Calculate and preview, do not edit playlist")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--quiet", action="store_true", help="Suppress info logs (warnings still shown)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--top", type=int, default=20, help="Preview N items (0 disables)")

    p.add_argument("--backup-dir", type=Path, default=Path("."), help="Directory to write backups")
    p.add_argument("--no-backup", action="store_true", help="Do not write a backup before editing")

    args = p.parse_args()

    if args.workers < 0:
        p.error("--workers must be >= 0")
    if args.retries < 0:
        p.error("--retries must be >= 0")
    if args.stats_delay is not None and args.stats_delay < 0:
        p.error("--stats-delay must be >= 0")
    if args.move_delay is not None and args.move_delay < 0:
        p.error("--move-delay must be >= 0")
    if args.top < 0:
        p.error("--top must be >= 0")

    return args


# ----------------------------- Main --------------------------------------- #

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        sorter = PlaylistSorter(args)
        tracks = sorter.fetch_playlist_items()

        if not tracks:
            print("Playlist empty.")
            return

        stats = sorter.gather_stats(tracks)
        if not stats:
            print("No stats gathered.")
            return

        sorted_stats = sorted(stats, key=lambda s: s.sort_key)

        if args.top > 0:
            print("\n--- Top %d Preview ---" % args.top)
            for i, s in enumerate(sorted_stats[:args.top], 1):
                print("%2d. %11s views | %s - %s" % (i, format(s.combined_views, ","), s.artists, s.title))
            print("-" * 60)

        if args.dry_run:
            print("Dry run. Simulating sort...")
            sorter.reorder(sorted_stats)
            return

        if not args.yes:
            if input("Reorder %d items? [y/N] " % len(sorted_stats)).strip().lower() != "y":
                print("Aborted.")
                return

        sorter.reorder(sorted_stats)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
