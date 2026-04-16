import csv
from pathlib import Path

LOG_DIR = Path("logs")


def find_latest_log():
    logs = sorted(LOG_DIR.glob("yaw_log_*.csv"))
    if not logs:
        raise RuntimeError("No log files found in logs/ directory.")
    return logs[-1]


def analyze_log(filepath):
    rows = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_runtime = float(rows[-1]["elapsed_s"])

    search_to_center = None
    center_to_handoff = None
    first_detection = None

    for r in rows:
        event = r["event"]
        t = float(r["elapsed_s"])

        if event == "STATE_CHANGE_SEARCH_TO_CENTER" and search_to_center is None:
            search_to_center = t

        if event == "CENTERING_WINDOW_STARTED" and first_detection is None:
            first_detection = t

        if event == "STATE_CHANGE_CENTER_TO_HANDOFF" and center_to_handoff is None:
            center_to_handoff = t

    print("\n==============================")
    print("MISSION AFTER ACTION REVIEW")
    print("==============================\n")

    print(f"Log file: {filepath.name}")
    print(f"Total runtime: {total_runtime:.2f} seconds")

    if search_to_center:
        print(f"\nTime to confirmed detection: {search_to_center:.2f} s")

    if center_to_handoff:
        print(f"Time to target centering: {center_to_handoff:.2f} s")

    if search_to_center and center_to_handoff:
        tracking_time = center_to_handoff - search_to_center
        print(f"Tracking duration before handoff: {tracking_time:.2f} s")

    print("\nEvent Summary:")
    print("------------------------------")

    for r in rows:
        if r["event"] != "":
            print(f"{float(r['elapsed_s']):6.2f}s  |  {r['event']}")

    print("\n==============================\n")


def main():
    latest = find_latest_log()
    analyze_log(latest)


if __name__ == "__main__":
    main()