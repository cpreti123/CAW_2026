import cv2
import numpy as np
import time
import csv
from pathlib import Path
from pymavlink import mavutil

# =========================
# MAVLink / control settings
# =========================
UDP_IN = "udp:0.0.0.0:14560"   # must match MAVProxy output add <WINDOWS_IP>:14560

SEARCH_YAW_STEP_DEG = 6.0
SEARCH_YAW_RATE_DPS = 30.0
SEARCH_INTERVAL_S = 0.35

TRACK_CONTROL_HZ = 8.0
TRACK_YAW_RATE_DPS = 60.0
TRACK_KP_STEP = 10.0
TRACK_MAX_STEP_DEG = 8.0

# =========================
# Vision settings
# =========================
LOWER_GREEN = np.array([35, 50, 30])
UPPER_GREEN = np.array([90, 255, 200])
AREA_MIN = 800

# =========================
# Detection / state machine settings
# =========================
DETECTION_CONFIRM_FRAMES = 5
CENTER_THRESHOLD = 0.10
CENTER_HOLD_TIME_S = 1.0

# =========================
# Logging settings
# =========================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"yaw_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"


# =========================
# Helper functions
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def condition_yaw_relative(m, delta_deg: float, yaw_rate_deg_s: float):
    if abs(delta_deg) < 1e-3:
        return

    direction = 1 if delta_deg > 0 else -1   # 1=cw, -1=ccw
    angle = abs(delta_deg)

    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        angle,
        yaw_rate_deg_s,
        direction,
        1,   # relative
        0, 0, 0
    )


def find_green_target(frame):
    """
    Returns:
        target_found (bool)
        target_x_norm (float or None)
        debug_frame (annotated frame)
    """
    debug_frame = frame.copy()
    h, w = frame.shape[:2]
    cx_img = w // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_found = False
    target_x_norm = None

    cv2.line(debug_frame, (cx_img, 0), (cx_img, h), (255, 255, 255), 1)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > AREA_MIN:
            x, y, ww, hh = cv2.boundingRect(c)
            cx = x + ww // 2
            cy = y + hh // 2

            cv2.rectangle(debug_frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.circle(debug_frame, (cx, cy), 5, (0, 255, 0), -1)

            target_x_norm = (cx - cx_img) / float(cx_img)
            target_found = True

            cv2.putText(
                debug_frame,
                f"x_norm={target_x_norm:+.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

    return target_found, target_x_norm, debug_frame


class CsvLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = open(filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp",
            "elapsed_s",
            "state",
            "target_found",
            "target_x_norm",
            "detection_count",
            "yaw_command_deg",
            "event"
        ])
        self.file.flush()

    def log(self, elapsed_s, state, target_found, target_x_norm, detection_count, yaw_command_deg, event=""):
        self.writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{elapsed_s:.3f}",
            state,
            int(target_found),
            "" if target_x_norm is None else f"{target_x_norm:.4f}",
            detection_count,
            "" if yaw_command_deg is None else f"{yaw_command_deg:.4f}",
            event
        ])
        self.file.flush()

    def close(self):
        self.file.close()


# =========================
# Main
# =========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0) failed).")

    print("Waiting for MAVLink heartbeat from SITL...")
    m = mavutil.mavlink_connection(UDP_IN)
    m.wait_heartbeat()
    print(f"Heartbeat OK (sysid={m.target_system}, compid={m.target_component}).")

    logger = CsvLogger(LOG_FILE)
    start_time = time.time()

    print(f"Logging to: {LOG_FILE}")
    print()
    print("MANUAL SETUP:")
    print("1. In QGroundControl, take off manually to ~10m and hover.")
    print("2. Then come back here and press ENTER to start autonomous search.")
    input()

    state = "SEARCH"
    detection_count = 0
    centered_since = None

    last_search_cmd = 0.0
    last_track_cmd = 0.0
    track_dt_min = 1.0 / TRACK_CONTROL_HZ

    print("Autonomy started.")
    print("SEARCH -> CENTER -> HANDOFF")
    print("Press 'q' in the video window to quit.")

    logger.log(0.0, state, False, None, detection_count, None, "START_AUTONOMY")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            elapsed_s = time.time() - start_time
            target_found, target_x_norm, debug_frame = find_green_target(frame)
            yaw_command_deg = None
            event = ""

            if state == "SEARCH":
                cv2.putText(debug_frame, "STATE: SEARCH", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                if target_found:
                    detection_count += 1
                else:
                    detection_count = 0

                now = time.time()
                if (now - last_search_cmd) >= SEARCH_INTERVAL_S and detection_count < DETECTION_CONFIRM_FRAMES:
                    yaw_command_deg = SEARCH_YAW_STEP_DEG
                    condition_yaw_relative(m, yaw_command_deg, SEARCH_YAW_RATE_DPS)
                    last_search_cmd = now

                if detection_count >= DETECTION_CONFIRM_FRAMES:
                    state = "CENTER"
                    centered_since = None
                    event = "STATE_CHANGE_SEARCH_TO_CENTER"
                    print("Target confirmed. Switching to CENTER.")

            elif state == "CENTER":
                cv2.putText(debug_frame, "STATE: CENTER", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                if not target_found:
                    detection_count = 0
                    centered_since = None
                    state = "SEARCH"
                    event = "STATE_CHANGE_CENTER_TO_SEARCH"
                    print("Lost target. Returning to SEARCH.")
                else:
                    now = time.time()
                    if (now - last_track_cmd) >= track_dt_min:
                        yaw_command_deg = clamp(
                            TRACK_KP_STEP * target_x_norm,
                            -TRACK_MAX_STEP_DEG,
                            TRACK_MAX_STEP_DEG
                        )
                        condition_yaw_relative(m, yaw_command_deg, TRACK_YAW_RATE_DPS)
                        last_track_cmd = now

                    if abs(target_x_norm) < CENTER_THRESHOLD:
                        if centered_since is None:
                            centered_since = time.time()
                            event = "CENTERING_WINDOW_STARTED"
                        elif time.time() - centered_since >= CENTER_HOLD_TIME_S:
                            state = "HANDOFF"
                            event = "STATE_CHANGE_CENTER_TO_HANDOFF"
                            print("HANDOFF READY: target centered and stable.")
                    else:
                        centered_since = None

            elif state == "HANDOFF":
                cv2.putText(debug_frame, "STATE: HANDOFF", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(debug_frame, "HOTL READY", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if state == "SEARCH":
                cv2.putText(debug_frame, f"detect_count={detection_count}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            logger.log(
                elapsed_s=elapsed_s,
                state=state,
                target_found=target_found,
                target_x_norm=target_x_norm,
                detection_count=detection_count,
                yaw_command_deg=yaw_command_deg,
                event=event
            )

            cv2.imshow("Search -> Detect -> Center -> Handoff", debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.log(
                    elapsed_s=time.time() - start_time,
                    state=state,
                    target_found=target_found,
                    target_x_norm=target_x_norm,
                    detection_count=detection_count,
                    yaw_command_deg=None,
                    event="USER_QUIT"
                )
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        print(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()