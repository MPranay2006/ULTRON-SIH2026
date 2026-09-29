"""
Step 6.7 - test the UART/MAVLink link between the Raspberry Pi 5 and the PIX6.
Run on the Pi (venv active):  python ~/sar_drone/mavlink_test.py
Stop with Ctrl+C.
"""
import sys
import time
from pymavlink import mavutil

PORT = "/dev/ttyAMA0"   # Pi 5 GPIO14/15 UART (enabled with dtparam=uart0=on)
BAUD = 115200           # must match SERIAL2_BAUD = 115 on the PIX6

print(f"Opening {PORT} at {BAUD} baud ...")
m = mavutil.mavlink_connection(PORT, baud=BAUD)

print("Waiting for heartbeat from the flight controller (10 s max) ...")
if m.wait_heartbeat(timeout=10) is None:
    sys.exit("NO HEARTBEAT - check wiring (TX->RX, RX->TX, GND), baud rate, and SERIAL2 settings.")
print(f"HEARTBEAT OK - system {m.target_system}, component {m.target_component}")

# Ask the PIX6 to stream all telemetry at 5 messages per second
m.mav.request_data_stream_send(m.target_system, m.target_component,
                               mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)

FIX = {0: "NO GPS", 1: "NO FIX", 2: "2D FIX", 3: "3D FIX", 4: "DGPS", 5: "RTK FLOAT", 6: "RTK FIXED"}
gps, att, last_print = {}, {}, 0
while True:
    msg = m.recv_match(type=["GPS_RAW_INT", "GLOBAL_POSITION_INT", "ATTITUDE"],
                       blocking=True, timeout=5)
    if msg is None:
        print("(no telemetry for 5 s - is the stream request being ignored?)")
        continue
    t = msg.get_type()
    if t == "GPS_RAW_INT":
        gps["fix"], gps["sats"] = FIX.get(msg.fix_type, msg.fix_type), msg.satellites_visible
    elif t == "GLOBAL_POSITION_INT":
        gps["lat"], gps["lon"] = msg.lat / 1e7, msg.lon / 1e7
        gps["alt"], gps["hdg"] = msg.relative_alt / 1000, msg.hdg / 100
    elif t == "ATTITUDE":
        att["roll"], att["pitch"] = msg.roll * 57.2958, msg.pitch * 57.2958
    if time.time() - last_print > 1 and gps and att:
        last_print = time.time()
        print(f"GPS {gps.get('fix','?'):8} sats {gps.get('sats','?'):>2} | "
              f"lat {gps.get('lat',0):.7f} lon {gps.get('lon',0):.7f} | "
              f"rel_alt {gps.get('alt',0):6.2f} m | hdg {gps.get('hdg',0):6.1f} | "
              f"roll {att['roll']:6.1f} pitch {att['pitch']:6.1f}")
