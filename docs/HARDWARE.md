# Hardware Documentation

## Bill of Materials

| Component | SKU | Quantity | Purpose |
|-----------|-----|----------|---------|
| Raspberry Pi 5 (4GB/8GB) | - | 1 | Main compute unit |
| Raspberry Pi Camera Module 3 | 46-1 | 1 | Primary vision sensor |
| HighPi Pro 5S Case | 9015 | 1 | Protective enclosure |
| USB-C PD Power Supply 27W | 492-2 | 1 | Power delivery |
| Raspberry Pi Active Cooler | 374-1 | 1 | Thermal management |
| MicroSD Card 32GB | 1350-1 | 1 | OS and storage |
| AI Hat+ 2 | - | 1 | Neural accelerator (optional) |

## Component Details

### Raspberry Pi Camera Module 3 (SKU: 46-1)

**Specifications:**
- 12MP Sony IMX708 sensor
- HDR support
- Autofocus
- Wide field of view (75°) or Ultra-wide (120°)
- Low-light performance
- 1080p @ 50fps, 720p @ 100fps

**Connection:**
- Uses CSI-2 ribbon cable
- Connect to CAM/DISP 0 or CAM/DISP 1 port on Pi 5

**Configuration:**
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Test camera
libcamera-hello
libcamera-still -o test.jpg
```

### HighPi Pro 5S Case (SKU: 9015)

**Features:**
- Aluminum construction with heatsink integration
- Compatible with Active Cooler
- GPIO access cutout
- Camera ribbon cable slot
- Multiple mounting options

**Assembly Notes:**
1. Install Active Cooler on Pi 5 first
2. Route camera ribbon cable before closing case
3. Ensure GPIO pins align with case cutout
4. Use included thermal pads if provided

### USB-C PD Power Supply 27W (SKU: 492-2)

**Specifications:**
- Output: 5.1V 5A (27W)
- USB-C Power Delivery compliant
- Sufficient for Pi 5 + peripherals

**Power Budget:**
| Component | Typical Draw |
|-----------|-------------|
| Raspberry Pi 5 | 3-5W idle, 8-12W load |
| Camera Module 3 | ~0.5W |
| Active Cooler | ~0.5W |
| AI Hat+ 2 | 2-4W (if installed) |
| **Total** | ~15W max |

27W supply provides ample headroom.

### Raspberry Pi Active Cooler (SKU: 374-1)

**Features:**
- Aluminum heatsink with integrated fan
- PWM-controlled fan speed
- Connects to Pi 5's dedicated fan header
- Keeps CPU under 60°C under load

**Installation:**
1. Remove any existing thermal interface
2. Apply thermal paste (pre-applied on cooler)
3. Align with CPU and secure clips
4. Connect 4-pin cable to fan header

**Fan Control:**
Fan speed is automatic via Pi 5 firmware. Override with:
```bash
# Check fan status
cat /sys/class/thermal/cooling_device0/cur_state

# Manual control (not recommended)
echo 3 | sudo tee /sys/class/thermal/cooling_device0/cur_state
```

### MicroSD Card 32GB (SKU: 1350-1)

**Pre-installed:**
- Raspberry Pi OS 32-bit

**First Boot:**
1. Insert card into Pi 5 slot
2. Connect display and keyboard
3. Power on
4. Complete initial setup wizard

**Recommended Post-Setup:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Expand filesystem (if needed)
sudo raspi-config --expand-rootfs

# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh
```

### AI Hat+ 2 (Optional)

**Purpose:**
- Hardware acceleration for neural networks
- Offloads TensorFlow/OpenCV inference from CPU
- Enables real-time object detection

**Specifications:**
- Hailo-8L accelerator (13 TOPS)
- PCIe interface to Pi 5
- Supported by TensorFlow Lite, OpenCV DNN

**Installation:**
1. Power off Pi 5
2. Attach Hat+ 2 to GPIO header
3. Secure with standoffs
4. Boot and install drivers

```bash
# Install Hailo runtime (example)
sudo apt install hailo-driver hailo-runtime

# Verify detection
hailortcli scan
```

## Assembly Order

1. **Prepare Pi 5**
   - Unbox and inspect for damage

2. **Install Active Cooler**
   - Attach heatsink/fan to CPU
   - Connect fan cable to header

3. **Connect Camera Module**
   - Gently lift CSI connector latch
   - Insert ribbon cable (blue side toward USB ports)
   - Press latch down to secure

4. **Install AI Hat+ 2** (if using)
   - Align with GPIO header
   - Press down firmly
   - Add standoffs for stability

5. **Mount in Case**
   - Route cables through appropriate slots
   - Align Pi with case mounting points
   - Secure with included screws

6. **Insert MicroSD Card**
   - Push until click

7. **Connect Power**
   - Plug USB-C power supply
   - System should boot

## GPIO Pin Reference

### Pi 5 GPIO Header (40-pin)

```
                    3V3  (1) (2)  5V
          GPIO 2 (SDA)  (3) (4)  5V
          GPIO 3 (SCL)  (5) (6)  GND
              GPIO 4    (7) (8)  GPIO 14 (TXD)
                  GND   (9) (10) GPIO 15 (RXD)
             GPIO 17   (11) (12) GPIO 18
             GPIO 27   (13) (14) GND
             GPIO 22   (15) (16) GPIO 23
                 3V3   (17) (18) GPIO 24
 GPIO 10 (SPI MOSI)   (19) (20) GND
  GPIO 9 (SPI MISO)   (21) (22) GPIO 25
 GPIO 11 (SPI SCLK)   (23) (24) GPIO 8 (CE0)
                 GND   (25) (26) GPIO 7 (CE1)
          GPIO 0 (ID)  (27) (28) GPIO 1 (ID)
              GPIO 5   (29) (30) GND
              GPIO 6   (31) (32) GPIO 12
             GPIO 13   (33) (34) GND
             GPIO 19   (35) (36) GPIO 16
             GPIO 26   (37) (38) GPIO 20
                 GND   (39) (40) GPIO 21
```

### Reserved Pins for THE EYE

| Pin | GPIO | Function |
|-----|------|----------|
| 3, 5 | 2, 3 | I2C (sensors) |
| 7 | 4 | PIR Motion Sensor (future) |
| 11 | 17 | Status LED (future) |
| 12 | 18 | Relay Control (future) |

## Network Configuration

### Static IP (Recommended)
Edit `/etc/dhcpcd.conf`:
```
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

### Firewall Setup
```bash
# Install UFW
sudo apt install ufw

# Allow SSH and FastAPI
sudo ufw allow 22
sudo ufw allow 8000

# Enable firewall
sudo ufw enable
```

## Troubleshooting

### Camera Not Detected
```bash
# Check connection
libcamera-hello --list-cameras

# Verify driver
vcgencmd get_camera
```

### Overheating
- Verify Active Cooler fan is spinning
- Check thermal paste contact
- Ensure case ventilation is clear

### Power Issues
- Use official 27W supply
- Avoid USB hubs on Pi power port
- Check for undervoltage warnings: `vcgencmd get_throttled`
