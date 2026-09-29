"""
SAR Drone - command-line ground station (run it in Windows PowerShell).

Talks to the PIX6 over the 3DR 433 MHz telemetry radio (or to Mission
Planner's simulator) using MAVLink. Type commands at the  GCS>  prompt.

  python scripts\\gcs_console.py --connect tcp:127.0.0.1:5762      (SITL simulator)
  python scripts\\gcs_console.py --connect COM5 --baud 57600        (real drone via radio)

SAFETY: test everything in SITL first. On the real drone keep the RC
transmitter in your hand - its mode switch overrides this console.
Only ONE program can open the radio's COM port at a time, so disconnect
Mission Planner from that COM port before starting this.
"""
import argparse
import math
import os
import sys
import threading
import time

from pymavlink import mavutil

EARTH_R = 6378137.0
FIX = {0: "NO GPS", 1: "NO FIX", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK-FLOAT", 6: "RTK-FIXED"}
HELP = """
Commands:
  status                    one-line vehicle status
  watch                     live status every second (Ctrl+C to stop watching)
  mode <NAME>               e.g. mode GUIDED | LOITER | AUTO | RTL | LAND | STABILIZE | ALT_HOLD
  arm                       arm motors (asks for YES)
  disarm                    disarm motors (only works on the ground)
  takeoff <alt_m>           take off to alt (needs GUIDED + armed)
  goto <lat> <lon> [alt_m]  fly to a point (needs GUIDED, max 300 m away)
  upload <file.waypoints>   upload a mission, e.g. upload outputs\\search_mission.waypoints
  start                     start the uploaded mission (switches to AUTO)
  rtl | land | loiter       quick safety modes (no confirmation)
  param get <NAME>          read a flight-controller parameter, e.g. param get SERIAL2_BAUD
  param set <NAME> <VALUE>  change a parameter (saved permanently on the PIX6)
  reboot                    reboot the flight controller (only when disarmed)
  link                      diagnostics: which messages are arriving from the drone
  help                      this list
  quit                      exit (the drone keeps doing whatever it was doing)
"""


class Vehicle:
    """Keeps a live copy of the drone's state, updated by a background thread."""

    def __init__(self, conn, baud):
        print(f"Connecting to {conn} ...")
        self.m = mavutil.mavlink_connection(conn, baud=baud, source_system=255)
        if self.m.wait_heartbeat(timeout=15) is None:
            sys.exit("NO HEARTBEAT - check COM port / baud / that the drone or SITL is running.")
        if self.m.target_component == 0:
            self.m.target_component = 1          # the autopilot itself is always component 1
        print(f"HEARTBEAT OK - system {self.m.target_system}, component {self.m.target_component}")
        self.s = {"lat": 0.0, "lon": 0.0, "alt": 0.0, "hdg": 0.0, "fix": "?", "sats": 0,
                  "volt": 0.0, "batt": -1, "armed": False, "mode": "?"}
        self.acks = {}
        self.params = {}
        self.counts = {}
        self.lock = threading.Lock()
        self.running = True
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 2, 1)
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._gcs_heartbeat, daemon=True).start()
        time.sleep(1.5)

    # --- background threads -------------------------------------------------
    def _rx(self):
        while self.running:
            msg = self.m.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            t = msg.get_type()
            self.counts[t] = self.counts.get(t, 0) + 1
            with self.lock:
                if t == "HEARTBEAT" and msg.get_srcSystem() == self.m.target_system \
                        and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                    self.s["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    self.s["mode"] = self.m.flightmode
                elif t == "GLOBAL_POSITION_INT":
                    self.s.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                                  alt=msg.relative_alt / 1000, hdg=msg.hdg / 100)
                elif t == "GPS_RAW_INT":
                    self.s.update(fix=FIX.get(msg.fix_type, msg.fix_type), sats=msg.satellites_visible)
                elif t == "SYS_STATUS":
                    self.s.update(volt=msg.voltage_battery / 1000, batt=msg.battery_remaining)
                elif t == "COMMAND_ACK":
                    self.acks[msg.command] = msg.result
                elif t == "PARAM_VALUE":
                    pid = msg.param_id.decode() if isinstance(msg.param_id, bytes) else msg.param_id
                    self.params[pid.strip("\x00")] = msg.param_value
            if t == "STATUSTEXT":                       # pre-arm failures, warnings, etc.
                print(f"\n  [drone] {msg.text}")

    def _gcs_heartbeat(self):
        while self.running:                            # tells the PIX6 a ground station is alive
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                      mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            time.sleep(1)

    # --- helpers --------------------------------------------------------------
    def state(self):
        with self.lock:
            return dict(self.s)

    def status_line(self):
        s = self.state()
        batt = f"{s['batt']}%" if s["batt"] >= 0 else "?"
        return (f"{s['mode']:9} {'ARMED' if s['armed'] else 'disarmed':8} | GPS {s['fix']:4} "
                f"sats {s['sats']:2} | {s['lat']:.7f}, {s['lon']:.7f} | alt {s['alt']:6.1f} m | "
                f"hdg {s['hdg']:5.1f} | batt {s['volt']:.1f} V {batt}")

    def command(self, cmd, p=(0, 0, 0, 0, 0, 0, 0), wait=3.0):
        self.acks.pop(cmd, None)
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, cmd, 0, *p)
        end = time.time() + wait
        while time.time() < end:
            if cmd in self.acks:
                r = self.acks[cmd]
                return r == mavutil.mavlink.MAV_RESULT_ACCEPTED, r
            time.sleep(0.05)
        return False, None

    def param(self, name, value=None, tries=3, wait=4.0):
        """Read (value=None) or write a parameter; retries because radio links drop packets."""
        name = name.upper()
        ts, tc = self.m.target_system, self.m.target_component
        for attempt in range(tries):
            self.params.pop(name, None)
            if value is None:
                self.m.mav.param_request_read_send(ts, tc, name.encode(), -1)
            else:
                self.m.mav.param_set_send(ts, tc, name.encode(), float(value),
                                          mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            end = time.time() + wait
            while time.time() < end:
                if name in self.params:
                    return self.params[name]
                time.sleep(0.05)
            if attempt < tries - 1:
                print(f"  (no reply yet, retrying {attempt + 2}/{tries} ...)")
        return None

    def set_mode(self, name):
        mapping = self.m.mode_mapping() or {}
        name = name.upper()
        if name not in mapping:
            print(f"  Unknown mode '{name}'. Available: {', '.join(sorted(mapping))}")
            return False
        self.m.set_mode(mapping[name])
        end = time.time() + 4
        while time.time() < end:
            if self.state()["mode"] == name:
                print(f"  Mode is now {name}")
                return True
            time.sleep(0.1)
        print(f"  Mode change to {name} NOT confirmed (see any [drone] message above).")
        return False


def dist_m(lat1, lon1, lat2, lon2):
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return EARTH_R * math.hypot(x, math.radians(lat2 - lat1))


def load_waypoints(path):
    """Read a Mission Planner 'QGC WPL 110' file into mission items."""
    items = []
    with open(path) as f:
        if not f.readline().startswith("QGC WPL"):
            raise ValueError("not a QGC WPL waypoints file")
        for line in f:
            c = line.strip().split("\t")
            if len(c) < 12:
                continue
            frame = int(c[2])
            items.append(dict(frame=6 if frame == 3 else frame, cmd=int(c[3]),
                              p=tuple(float(x) for x in c[4:8]),
                              lat=float(c[8]), lon=float(c[9]), alt=float(c[10])))
    return items


def upload_mission(v, items):
    m, ts, tc = v.m, v.m.target_system, v.m.target_component
    v.running = False                  # pause the background reader while we talk mission protocol
    time.sleep(1.2)
    try:
        m.mav.mission_clear_all_send(ts, tc)
        time.sleep(0.5)
        m.mav.mission_count_send(ts, tc, len(items))
        sent, deadline = set(), time.time() + 20
        while time.time() < deadline:
            msg = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
                               blocking=True, timeout=5)
            if msg is None:
                continue
            if msg.get_type() == "MISSION_ACK":
                if msg.type != 0:
                    print(f"  Mission REJECTED (code {msg.type})")
                    return False
                if len(sent) == len(items):
                    return True
                continue
            it = items[msg.seq]
            m.mav.mission_item_int_send(ts, tc, msg.seq, it["frame"], it["cmd"], 0, 1, *it["p"],
                                        int(it["lat"] * 1e7), int(it["lon"] * 1e7), it["alt"])
            sent.add(msg.seq)
            deadline = time.time() + 20
        print("  Upload timed out.")
        return False
    finally:
        v.running = True
        threading.Thread(target=v._rx, daemon=True).start()
        threading.Thread(target=v._gcs_heartbeat, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description="SAR drone command-line ground station")
    ap.add_argument("--connect", required=True, help="tcp:127.0.0.1:5762 (SITL) or COM5 (radio)")
    ap.add_argument("--baud", type=int, default=57600, help="3DR radio default is 57600")
    ap.add_argument("--max-alt", type=float, default=60, help="safety cap for takeoff/goto (m)")
    a = ap.parse_args()

    v = Vehicle(a.connect, a.baud)
    print(v.status_line())
    print(HELP)
    print("REMINDER: SITL first. On a real drone, keep the RC transmitter ready to take over.\n")

    while True:
        try:
            line = input("GCS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        s = v.state()
        try:
            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "status":
                print(" ", v.status_line())
            elif cmd == "watch":
                try:
                    while True:
                        print(" ", v.status_line())
                        time.sleep(1)
                except KeyboardInterrupt:
                    print()
            elif cmd == "mode" and len(args) == 1:
                v.set_mode(args[0])
            elif cmd in ("rtl", "land", "loiter"):
                v.set_mode(cmd)
            elif cmd == "arm":
                if input("  Type YES to ARM the motors: ").strip() != "YES":
                    print("  Cancelled.")
                    continue
                ok, r = v.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, (1, 0, 0, 0, 0, 0, 0), wait=5)
                end = time.time() + 3
                while ok and not v.state()["armed"] and time.time() < end:
                    time.sleep(0.1)                 # wait for the heartbeat to confirm
                print("  ARMED" if ok else f"  Arm REFUSED (result {r}) - read the [drone] messages above")
            elif cmd == "disarm":
                ok, r = v.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
                print("  Disarmed" if ok else f"  Disarm refused (result {r}) - is it still flying? Use 'land'.")
            elif cmd == "takeoff" and len(args) == 1:
                alt = float(args[0])
                if not 2 <= alt <= a.max_alt:
                    print(f"  Altitude must be between 2 and {a.max_alt} m"); continue
                if s["mode"] != "GUIDED" or not s["armed"]:
                    print("  Needs mode GUIDED and ARMED first ('mode GUIDED', then 'arm')."); continue
                ok, r = v.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, (0, 0, 0, 0, 0, 0, alt))
                print(f"  Taking off to {alt:.0f} m" if ok else f"  Takeoff refused (result {r})")
            elif cmd == "goto" and len(args) in (2, 3):
                lat, lon = float(args[0]), float(args[1])
                alt = float(args[2]) if len(args) == 3 else max(s["alt"], 10)
                d = dist_m(s["lat"], s["lon"], lat, lon)
                if s["mode"] != "GUIDED":
                    print("  Needs mode GUIDED first."); continue
                if d > 300 or not 2 <= alt <= a.max_alt:
                    print(f"  Refused: {d:.0f} m away (max 300 m) or altitude outside 2-{a.max_alt} m."); continue
                v.m.mav.set_position_target_global_int_send(
                    0, v.m.target_system, v.m.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 0b110111111000,
                    int(lat * 1e7), int(lon * 1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0)
                print(f"  Flying to {lat:.7f}, {lon:.7f} at {alt:.0f} m ({d:.0f} m away)")
            elif cmd == "upload" and len(args) == 1:
                if not os.path.exists(args[0]):
                    print(f"  File not found: {args[0]}"); continue
                items = load_waypoints(args[0])
                print(f"  Uploading {len(items)} mission items ...")
                print("  MISSION ACCEPTED" if upload_mission(v, items) else "  Upload FAILED")
            elif cmd == "param" and len(args) >= 2 and args[0] in ("get", "set"):
                if args[0] == "get" and len(args) == 2:
                    val = v.param(args[1])
                    print(f"  {args[1].upper()} = {val:g}" if val is not None
                          else f"  No answer - is '{args[1].upper()}' a real parameter name?")
                elif args[0] == "set" and len(args) == 3:
                    if s["armed"]:
                        print("  Refused: disarm before changing parameters."); continue
                    want = float(args[2])
                    val = v.param(args[1], want)
                    if val is None:
                        print("  No confirmation from the flight controller - check the name.")
                    elif abs(val - want) < 1e-4:
                        print(f"  {args[1].upper()} set to {val:g} (saved)")
                    else:
                        print(f"  PIX6 reports {args[1].upper()} = {val:g} (value not accepted)")
                else:
                    print("  Usage: param get NAME   |   param set NAME VALUE")
            elif cmd == "link":
                before = dict(v.counts)
                print(f"  Talking to system {v.m.target_system}, component {v.m.target_component}. "
                      f"Counting messages for 5 s ...")
                time.sleep(5)
                diff = {k: v.counts.get(k, 0) - before.get(k, 0) for k in v.counts}
                diff = {k: n for k, n in diff.items() if n > 0}
                if not diff:
                    print("  NOTHING received in 5 s - the link is down.")
                for k, n in sorted(diff.items(), key=lambda kv: -kv[1]):
                    print(f"    {k:24} {n / 5:5.1f} per second")
            elif cmd == "reboot":
                if s["armed"]:
                    print("  Refused: disarm first."); continue
                if input("  Type YES to REBOOT the flight controller: ").strip() != "YES":
                    print("  Cancelled."); continue
                v.command(mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, (1, 0, 0, 0, 0, 0, 0))
                print("  Reboot sent. Over USB the link drops: type 'quit' and reconnect in ~15 s.")
            elif cmd == "start":
                if v.set_mode("AUTO"):
                    v.command(mavutil.mavlink.MAV_CMD_MISSION_START)
            else:
                print("  Unknown command or wrong arguments. Type 'help'.")
        except ValueError as e:
            print(f"  Bad number/argument: {e}")

    v.running = False
    print("Console closed. The drone continues its current mode.")


if __name__ == "__main__":
    main()
