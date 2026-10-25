# Thermal soak test

Does the enclosure keep the Pi 5 out of throttling while it detects? One
run per node per enclosure setup, two hours, recorded.

**PI** = a terminal on the node. **LAPTOP** = PowerShell on Windows.

The node measures itself: every `THERMAL_SAMPLE_S` (30 s) it records the
SoC temperature, the firmware's throttle flags, fan RPM, CPU clock and the
detection rate, to `data/thermal/thermal-YYYY-MM-DD.csv` and to
`/api/system/thermal`. Nothing extra to install.

## Limits

| | |
|---|---|
| 80 °C | The Pi 5 starts lowering its clock |
| 85 °C | Hard throttle |
| **Pass** | Peak below 80 °C **and** no throttle flag at any point in the run |

`under_voltage` counts as a fail too, but it is the power supply, not the
enclosure: fix the supply (official 27 W) and rerun.

## Run it

1. **Label the setup · PI.** In `~/argus/deploy/node.env`, set what is being
   tested, then restart:

   ```bash
   cd ~/argus && sed -i 's/^THERMAL_ENCLOSURE=.*/THERMAL_ENCLOSURE=HighPi Pro 5S, closed, fan on/' deploy/node.env && grep -q '^THERMAL_ENCLOSURE=' deploy/node.env || echo 'THERMAL_ENCLOSURE=HighPi Pro 5S, closed, fan on' >> deploy/node.env; sudo systemctl restart argus-node
   ```

   Change the text for each variant (lid open, fan off, in the sun, ...).

2. **Close the case and leave the node detecting for 2 hours** in its real
   spot. Normal load is enough; for a worst case run with `--every-frame`
   (the detector on every frame).

3. **Read the result · LAPTOP.** Dashboard → node picker → **Thermal**,
   window **2 h**: peak, average, throttled samples and PASS/FAIL. Click
   **CSV** to keep the record, or straight from the node:

   ```powershell
   curl.exe -o thermal-node-a-2h.csv "http://10.0.0.140:8000/api/system/thermal.csv?minutes=120"
   ```

4. **Log it** in the hardware doc: node, enclosure text, ambient
   temperature, peak, average, PASS/FAIL, and the CSV file name.

## Quick check without the dashboard · PI

```bash
vcgencmd measure_temp; vcgencmd get_throttled; cat /sys/devices/platform/cooling_fan/hwmon/hwmon*/fan1_input
```

`throttled=0x0` is healthy. Any other value: the low bits are happening
now, bits 16 and up have happened since boot (`/api/system/thermal`
decodes them).
