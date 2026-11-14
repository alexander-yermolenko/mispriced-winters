from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mispriced_winters import power_client
from mispriced_winters.config import CITIES


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NASA POWER data for the app's cities.")
    parser.add_argument("--refresh", action="store_true", help="re-download cities that are already cached")
    parser.add_argument("--only", nargs="*", metavar="KEY", help="city keys to download (default: all)")
    args = parser.parse_args()

    cities = [city for city in CITIES if not args.only or city.key in args.only]
    failures = 0
    for i, city in enumerate(cities, start=1):
        prefix = f"[{i}/{len(cities)}] {city.label}"
        if power_client.is_cached(city.key) and not args.refresh:
            print(f"{prefix}: already cached")
            continue
        started = time.time()
        try:
            frame = power_client.get_daily(city.lat, city.lon, city.key, refresh=True)
        except Exception as exc:
            failures += 1
            print(f"{prefix}: FAILED — {exc}")
            continue
        print(
            f"{prefix}: {len(frame):,} days "
            f"({frame.index.min():%Y-%m-%d} → {frame.index.max():%Y-%m-%d}) in {time.time() - started:.1f} s"
        )
    print(f"Cache folder: {power_client.cache_path('x').parent}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
