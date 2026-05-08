# Phase 3 — FPGA Synthesis & Artix-7 Deployment

This document is the **end-to-end deployment guide** for putting the trained earthquake-detection CNN onto a Xilinx Artix-7 FPGA using **Vivado 2025**. It assumes **no prior Vivado experience**.

**Read first:** `phase1.md`, `phase2.md`, `phase2_5.md`, `phase2_6.md` for the model and benchmark context.

> **Two-machine workflow.** The Mac side (this repo) generates a Vivado HLS project tree. **Vivado does not run on macOS** — your ECE teammate's Linux or Windows laptop runs the actual synthesis and bitstream generation. This document tells both halves what to do.

---

## TL;DR

```
   ┌─────────────────────┐                    ┌─────────────────────┐
   │  THIS LAPTOP (Mac)  │                    │ TEAMMATE'S LAPTOP   │
   │                     │                    │  (Linux / Windows)  │
   │  convert_to_hls.py  │  hls_project.zip   │                     │
   │  ───────────────►   │ ─────────────────► │  Vivado 2025        │
   │                     │   (~5 MB)          │   ↓                 │
   │  Phase 2 trained    │                    │  vitis_hls (synth)  │
   │  Keras model        │                    │   ↓                 │
   │                     │                    │  Vivado IDE         │
   │                     │                    │  (block design,     │
   │                     │                    │   bitstream)        │
   │                     │                    │   ↓                 │
   │                     │                    │  USB-JTAG ─► Cmod   │
   │                     │                    │              A7-35T │
   └─────────────────────┘                    └─────────────────────┘
```

---

## Section 0 — What Vivado Even Is

If you've never used FPGA tooling, here's the mental model.

### What is an FPGA?

An FPGA is a chip full of generic digital building blocks — lookup tables (LUTs), flip-flops (FFs), DSP slices for arithmetic, block RAM (BRAM) for storage. You write a configuration ("bitstream") that wires these blocks into a custom digital circuit — so the same chip can be a video decoder one day and an earthquake detector the next.

The **Artix-7 35T** part on the Cmod A7-35T board has:
- **33,280 LUTs** (combinational logic)
- **41,600 FFs** (registers)
- **90 DSP slices** (hardware multipliers)
- **1,800 Kb of BRAM** (on-chip RAM in 50 × 36 Kb blocks)

Our 8K-parameter int8 CNN uses roughly: ~13 K LUTs, ~70 DSPs, ~150 Kb BRAM — all comfortably inside the chip.

### What is Vivado?

**Vivado** is AMD/Xilinx's software for designing FPGA bitstreams. It's a *suite* of tools, all bundled together in **Vivado 2025** under the **"Vitis Unified Software Platform"** launcher:

| Tool | What it does | When you use it |
|---|---|---|
| **Vitis HLS** | Translates **C++** describing an algorithm into **synthesizable RTL** (Verilog). Produces an "IP block" you can drop into a Vivado design. | This is where `hls4ml`'s output goes first. |
| **Vivado** (the IDE) | The main FPGA development environment. Lets you draw a **block design** (your IP + memory + AXI buses + clocks), simulate, and **synthesize the whole system to a bitstream**. | After Vitis HLS produces the IP, you wire it up here. |
| **Vitis** (the C/C++ IDE) | Optional. C/C++ environment for soft cores (MicroBlaze) or ARM cores (Zynq). Artix-7 has neither by default, so we won't use this in the simplest deployment path. | Only if you instantiate a MicroBlaze soft core later. |

You'll touch the first two, in that order: HLS → Vivado IDE.

### What is hls4ml?

[hls4ml](https://fastmachinelearning.org/hls4ml/) is a Python library that automatically generates Vitis HLS C++ from a Keras model. It writes the C++ for every Conv2D, BatchNorm, ReLU, MaxPool, GlobalAveragePooling2D, and Dense layer in our network, including the trained weights as `static const fixed<8,1>` arrays. We use it so you don't have to hand-write HLS for an 8K-parameter network.

### Why does this involve so many software stages?

Going from a `.h5` file to a flashing FPGA is a multi-stage compilation pipeline:

```
Keras  →  HLS C++  →  RTL (Verilog)  →  netlist  →  bitstream  →  FPGA
       ↑           ↑                  ↑           ↑             ↑
     hls4ml    Vitis HLS         Vivado synth   Vivado     USB-JTAG
                                 + place&route   bitgen
```

Most of this is automated by hls4ml + Vivado's tcl scripts. You mostly click through GUIs and wait.

---

## Section 1 — On the Mac: Generate the HLS Project

### 1.1 Prerequisites (already done in earlier phases)

```
~/Desktop/idp/
├── .venv-tf/                                  Phase 2 venv (Python 3.11 + TF 2.16)
├── out/eew_cnn_qat.keras                      Phase 2 trained QAT model (256 KB)
├── train_cnn.py                                builds the architecture
├── extract_weights.py                          Keras-2 helper
└── convert_to_hls.py                           Keras-3 main
```

Plus, install hls4ml (one-time):
```bash
source .venv-tf/bin/activate
pip install hls4ml
```

### 1.2 Run the conversion (Artix-7 / Vivado 2025)

```bash
cd ~/Desktop/idp
source .venv-tf/bin/activate
python convert_to_hls.py --board cmod-a7-35 --backend Vitis
```

Flag breakdown:
- `--board` — preset that picks the right part number. Options: `cmod-a7-35` (default), `arty-a7-35`, `arty-a7-100`, `nexys-a7-100`, `zedboard`.
- `--backend Vitis` — generates a project for **Vitis HLS** (Vivado 2021+ / 2025). Use `--backend Vivado` only for legacy Vivado HLS (≤ 2020.2).
- `--reuse-factor` — defaults to 1 (full parallel). If synthesis on a 35T part complains about DSP overflow, re-run with `--reuse-factor 2` to halve DSP usage.

**What happens:** a subprocess runs `extract_weights.py` in Keras-2 mode to dump the trained weights to `.npz`, then the main script runs in Keras-3 mode (which `hls4ml` requires) and writes the project tree.

Expected runtime: **~10 seconds**. Output:

```
out/hls_project/
├── firmware/                       generated HLS C++ — the actual model
│   ├── eew_cnn.cpp                 the top-level inference function
│   ├── eew_cnn.h                   header / function signature
│   ├── parameters.h                quantization config (bit widths, etc.)
│   ├── defines.h                   layer dimensions
│   ├── ap_types/                   Xilinx fixed-point header library
│   ├── nnet_utils/                 hls4ml's layer kernels
│   └── weights/                    .h files with trained weights as
│                                   static const fixed<8,1> arrays
├── tb_data/                        test bench input/output samples
├── eew_cnn_test.cpp                C-simulation test bench
├── eew_cnn_bridge.cpp              Python ↔ HLS bridge
├── build_prj.tcl                   "do everything" tcl script for Vitis HLS
├── project.tcl                     Project setup
├── vivado_synth.tcl                Post-HLS synthesis tcl
└── hls4ml_config.yml               human-readable config snapshot
```

### 1.3 Smoke-test on the Mac

The conversion script auto-runs a sanity check on real samples — you should see "Loaded weights for 7/16 layers" and a sigmoid-range output. If you want to manually re-check accuracy:

```bash
python -c "
import os; os.environ['TF_USE_LEGACY_KERAS']='0'
import sys; sys.path.insert(0, '.')
import numpy as np
from convert_to_hls import build_keras3_model, load_weights_into
m = build_keras3_model(); load_weights_into(m)
X = np.load('out/X_train.npy', mmap_mode='r')[:200].astype('float32')
y = np.load('out/y_train.npy')[:200]
print('accuracy:', ((m.predict(X, verbose=0).ravel() >= 0.5) == y).mean())
"
```

Should print `accuracy: ~0.98`.

### 1.4 Package for the teammate

```bash
cd ~/Desktop/idp/out
zip -r hls_project.zip hls_project
# 5 MB — fits in any email or USB stick
```

Send `hls_project.zip` to the teammate however you like (email, USB, Drive, git, scp). Nothing else from this Mac is needed.

---

## Section 2 — Teammate's Laptop: One-Time Setup

Goal: take a fresh Linux or Windows laptop and get it ready to build FPGA bitstreams. ETA for the whole section: **~3 hours**, mostly waiting for the installer.

### 2.1 Hardware / OS Checklist

| Item | Minimum | Recommended |
|---|---|---|
| **OS** | Ubuntu 22.04 LTS, RHEL/CentOS 8.x, **Windows 10/11** | Ubuntu 22.04 (most stable for Vivado on Linux) |
| **CPU** | x86-64, 4 cores | 8+ cores (synthesis is parallel) |
| **RAM** | **16 GB** (Vivado will swap painfully under this) | 32 GB |
| **Disk free** | **80 GB** for Vivado 2025 + Vitis full install | 120 GB SSD |
| **USB-A port** | needed to plug in the Cmod A7 | — |
| **Internet** | for installer download (~30 GB) | unmetered |

The teammate's laptop does NOT need a discrete GPU.

> If the teammate only has a Mac and can't dual-boot or use a separate machine, a **Linux VM under UTM/Parallels** technically works on Apple Silicon but is painfully slow. Better option: spin up an **AWS EC2 `c6i.4xlarge`** for an hour with a Vivado AMI from the AMD/Xilinx marketplace, build there, download the bitstream — costs ~$1-2 per build.

### 2.2 Create an AMD/Xilinx account

Go to <https://www.xilinx.com/registration/create-account.html> and sign up. The account is free; it's needed to download Vivado.

### 2.3 Download the Vivado 2025 Unified Installer

1. Go to <https://www.xilinx.com/support/download.html>
2. Pick **Vivado ML Edition — 2025.x** (whatever the latest 2025 release is at the time).
3. Download the **Web Installer** for the right OS:
   - Linux: `Xilinx_Unified_2025.x_xxxx_Lin64.bin` (~150 MB)
   - Windows: `Xilinx_Unified_2025.x_xxxx_Win64.exe` (~150 MB)

The web installer is small; it pulls the actual ~30 GB toolchain during install.

### 2.4 Run the installer

**Linux:**
```bash
chmod +x Xilinx_Unified_2025.x_xxxx_Lin64.bin
./Xilinx_Unified_2025.x_xxxx_Lin64.bin
```

**Windows:** double-click the `.exe` and click through.

In the installer GUI:

1. **Sign in** with the AMD/Xilinx account from step 2.2.
2. **Edition selection**: pick **"Vivado ML Standard"** — this is the free tier and **fully supports Artix-7**. (You don't need ML Enterprise or System Edition.)
3. **Components to install**: check these boxes —
   - ✅ **Vivado**
   - ✅ **Vitis** (this includes Vitis HLS — needed!)
   - ✅ **DocNav** (optional, helpful)
   - **Devices** — under "Production Devices", check **7-Series** (this includes Artix-7). Uncheck UltraScale/Versal/Zynq UltraScale unless you specifically need them; they're huge.
4. **Install location**: default `/tools/Xilinx/` on Linux or `C:\Xilinx\` on Windows is fine.
5. Click **Install**. Goes for **60–90 minutes** depending on disk speed and network.

> **Windows note:** add the install folder (`C:\Xilinx`) to Windows Defender's exclusion list before starting — antivirus scanning every file as it lands can triple the install time.

### 2.5 Install USB drivers for the Cmod A7

The Cmod A7 uses Digilent's USB-JTAG bridge, not Xilinx's. Install the Digilent runtime:

**Linux (Ubuntu):**
```bash
# Pull the udev rules so the user can access the JTAG without sudo
wget https://digilent.com/shop/digilent-adept-2-runtime-2.27.x/
# Or grab from https://digilent.com/shop/software/digilent-adept-2/download/
sudo dpkg -i digilent.adept.runtime_2.27.x-amd64.deb
sudo cp /usr/share/digilent/data/52-digilent-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo usermod -a -G plugdev $USER   # then log out + in
```

**Windows:** install the **Digilent Adept 2 Runtime** from <https://digilent.com/shop/software/digilent-adept-2/>. The installer drops the right USB drivers automatically. May need to plug in the Cmod once and let Windows finish driver setup.

### 2.6 Install Cmod A7 board files

Vivado doesn't ship with Cmod A7 board files by default. Grab them from Digilent's GitHub:

```bash
# Linux
git clone https://github.com/Digilent/vivado-boards.git
cp -r vivado-boards/new/board_files/* /tools/Xilinx/Vivado/2025.x/data/boards/board_files/
```

```powershell
# Windows (PowerShell)
git clone https://github.com/Digilent/vivado-boards.git
Copy-Item vivado-boards\new\board_files\* -Destination "C:\Xilinx\Vivado\2025.x\data\boards\board_files\" -Recurse
```

### 2.7 Set up shell environment

Every shell that runs Vivado/Vitis needs the settings sourced.

**Linux** — add to `~/.bashrc` (or `~/.zshrc`):
```bash
source /tools/Xilinx/Vivado/2025.x/settings64.sh
source /tools/Xilinx/Vitis_HLS/2025.x/settings64.sh
```

**Windows:** the installer creates Start Menu shortcuts that pre-set the environment ("Vivado 2025.x", "Vitis HLS 2025.x"). Use those, or run the equivalent `settings64.bat` from `cmd.exe`.

### 2.8 Verify install

In a fresh terminal:
```bash
vivado -version
# expected: Vivado v2025.x.x ...

vitis_hls -version
# expected: Vitis HLS v2025.x.x ...
```

Both commands should print version banners. If "command not found", the settings script wasn't sourced — go back to 2.7.

### 2.9 Plug in the Cmod A7 and verify JTAG

1. Plug the Cmod A7 into a USB-A port. The green `LD7` power LED should light. Two onboard LEDs (`LD0`/`LD1`) may be off, on, or blinking — depends on whatever was last flashed.
2. Open Vivado → **Tasks → Open Hardware Manager → Open Target → Auto Connect**. The board should appear as `xc7a35t_0`.

If it doesn't appear:
- Linux: `lsusb` should show `Digilent Adept USB Device`. If not, USB cable problem (some cheap USB-C-to-A cables are charge-only — try a different cable).
- Linux: re-check udev rules from 2.5; log out and back in.
- Windows: re-run the Digilent Adept installer; check Device Manager for unrecognized devices.

**You're done with one-time setup.** Time to build the bitstream.

---

## Section 3 — On the Teammate's Laptop: Run Vitis HLS

```bash
unzip hls_project.zip
cd hls_project
vitis_hls -f build_prj.tcl "csim=1 synth=1 cosim=1 export=1"
```

Four stages, each takes a few minutes:

| Stage | What it does | If it fails |
|---|---|---|
| `csim=1` | Compiles HLS C++ + test bench, runs as a normal C++ program against `tb_data/`. Compares HLS-fixed-point output against Keras-float reference. | Model conversion is broken — fix that before going further. |
| `synth=1` | High-Level Synthesis: turns C++ into RTL (Verilog). Reports estimated LUTs / FFs / BRAM / DSPs / clock period. | Read `eew_cnn_prj/solution1/syn/report/eew_cnn_csynth.rpt` for the resource breakdown. |
| `cosim=1` | RTL co-simulation: runs the Verilog against the same test bench. Verifies synthesis didn't break anything. | Almost always a precision issue — see Section 6. |
| `export=1` | Packages the RTL as a Vivado IP block. Output: `eew_cnn_prj/solution1/impl/ip/xilinx_com_hls_eew_cnn_1_0.zip`. | This is the file you import into the Vivado block design. |

For our 8K-param model on Cmod A7-35T, expected (post-synth) resource usage:
- LUTs: ~13–18 K of 33 K (40-55%)
- FFs: ~10–15 K of 41 K (25-35%)
- BRAM: ~5 of 50 (10%)
- DSPs: ~70 of 90 (75-80%) ⚠ tight
- Latency: ~50–150 clock cycles per inference at 100 MHz = 0.5–1.5 µs/inference (well below the 200 ms / 5 inferences-per-sec target from Phase 2.6)

If DSP usage tips over 90, the teammate should ping the Mac side to re-run with `--reuse-factor 2`.

### GUI alternative

If clicking is preferred:
```bash
vitis_hls
```
Then **File → Open Project** → pick `hls_project/eew_cnn_prj`. Use toolbar buttons:
- "Run C Simulation" (green play)
- "Run C Synthesis" (yellow gear)
- "Run C/RTL Cosimulation"
- "Export RTL"

The IP `.zip` lands at the same path either way.

---

## Section 4 — On the Teammate's Laptop: Build the Vivado Project

Now wire the IP into a complete FPGA design.

### 4.1 Pick a deployment style

Artix-7 has no ARM core, so unlike a Zynq board you can't just run a C program that talks to the FPGA. Three options, easiest first:

| Style | Effort | What it proves |
|---|---|---|
| **A. BRAM testbench** | Easiest. Pre-load a known waveform into Block RAM. Press a button → CNN runs once → LED on if prob > 0.5. | Bitstream classifies correctly. |
| **B. UART-driven** | Moderate. RTL state machine reads 200×3 bytes over UART, runs CNN, writes prob back. | End-to-end pipeline including I/O. Run from a serial terminal on the laptop. |
| **C. MicroBlaze soft core** | Hardest. Instantiate a MicroBlaze (a tiny CPU built out of FPGA fabric), run a C program that drives an AXI DMA into the CNN. Mirrors the Zynq workflow. | Production-grade, real-time. |

**Start with A.** It's a one-evening project, proves the model works on hardware, and gives a nice physical demo (LED). Then move to B/C if you need streaming I/O.

### 4.2 Create the Vivado project

```bash
vivado &
```

- **Create New Project** → Name: `eew_artix7` → choose a location.
- Project type: **RTL Project**.
- Add Sources / Add Constraints: skip both for now.
- **Default Part**: in the **Boards** tab, search "Cmod A7-35T". Pick the matching entry. (If it doesn't show up, board files weren't installed — go back to Section 2.6.)
- Finish.

### 4.3 Add the HLS-generated IP to the IP catalog

- **Project Manager → IP Catalog → right-click → Add Repository**
- Browse to `hls_project/eew_cnn_prj/solution1/impl/ip/`. Vivado scans it and adds the `eew_cnn` IP.

### 4.4 Build the block design (style A: BRAM testbench)

- **Flow Navigator → IP Integrator → Create Block Design** → Name: `eew_bd`.
- Add these IPs (click `+` and search):
  1. **eew_cnn** — your CNN.
  2. **Block Memory Generator** — pre-loaded with a waveform via `.coe` file (see 4.5).
  3. **Clocking Wizard** — 12 MHz Cmod oscillator → 100 MHz fabric clock.
  4. **VIO (Virtual I/O) Debug Core** — lets you push the "go" button from the laptop without needing a physical button.

- Run **Connection Automation** to wire clocks/resets.
- Manually wire:
  - Clocking Wizard `clk_out1 (100 MHz)` → `eew_cnn ap_clk` → `BRAM clk` → `VIO clk`.
  - VIO output `probe_out0[0]` → `eew_cnn ap_start`.
  - BRAM data → `eew_cnn input_1`.
  - `eew_cnn ap_done` → onboard LED `LD0` (via constraints file in 4.6).
  - `eew_cnn layer16_out > 0.5 (sign-bit comparison)` → onboard LED `LD1`.

- **Validate Design** (F6). Should pass with green ticks.
- Right-click the block design in Sources → **Create HDL Wrapper** → Let Vivado manage wrapper.

### 4.5 Generate the BRAM init file

On either the Mac or the teammate's laptop:
```bash
python -c "
import numpy as np
X = np.load('out/X_train.npy', mmap_mode='r')
y = np.load('out/y_train.npy')
# pick one earthquake to bake in
idx = int(np.flatnonzero(y == 1)[0])
window = X[idx].astype(np.float32).flatten()   # 600 floats
# convert to fixed<8,1> (signed int8)
quantized = np.clip(np.round(window * 64), -128, 127).astype(np.int8)
with open('test_quake.coe', 'w') as f:
    f.write('memory_initialization_radix=10;\n')
    f.write('memory_initialization_vector=\n')
    f.write(', '.join(map(str, quantized.tolist())) + ';\n')
print('wrote test_quake.coe — index', idx)
"
```

In Vivado, double-click the Block Memory Generator IP → set **Load Init File** → browse to `test_quake.coe`.

### 4.6 Add the constraints file

Cmod A7's pin assignments are known. Grab the master XDC from <https://github.com/Digilent/digilent-xdc> → `Cmod-A7-Master.xdc`. Add to your project (**Add Sources → Add or create constraints**). Uncomment lines for:
- the 12 MHz clock (`sysclk` on pin L17)
- LD0 / LD1 outputs (pins A17, C16)
- the user button BTN0 (pin A18) — optional alternative to VIO for triggering.

### 4.7 Generate the bitstream

- **Flow Navigator → Generate Bitstream**.
- Vivado runs synthesis → implementation → bitstream generation. **15–45 minutes** depending on host speed.
- Output: `eew_artix7.runs/impl_1/eew_bd_wrapper.bit`.

---

## Section 5 — Program the Cmod A7

With the board plugged in:

- **Flow Navigator → Open Hardware Manager → Open Target → Auto Connect**. `xc7a35t_0` should appear.
- **Program Device** → select `eew_bd_wrapper.bit`. Upload takes ~2 seconds.
- The yellow `DONE` LED on the board should light = bitstream is running.

### Trigger inference and read the result

If you used the VIO approach from 4.4:
- Open `eew_bd_i/vio_0` in the Hardware Manager.
- Toggle `probe_out0[0]` to `1` then back to `0`. This pulses `ap_start` on the CNN.
- Within microseconds, `ap_done` fires → `LD0` lights briefly.
- `LD1` shows the binary classification: lit = "earthquake detected".

If you wired the user button instead (no VIO):
- Press `BTN0` on the board. Same behavior.

### Sanity-check against the Streamlit demo (from Phase 2.5)

Note the index in `test_quake.coe` (the script printed it). On the Mac:
```bash
source .venv-tf/bin/activate
TF_USE_LEGACY_KERAS=1 streamlit run app.py
# pick same index in the "By trace name" mode and confirm prediction matches
```

The Cmod's LED should agree with Streamlit's prediction to within rounding. **If they disagree by more than 0.05 probability, something's wrong** — start with the cosim mismatch FAQ in Section 6.

---

## Section 6 — Troubleshooting FAQ

**Q: `vitis_hls` says "WARNING: Cosimulation mismatch"**
A: Almost always a precision issue. Check `hls4ml_config.yml`: the accumulator precision (default `fixed<20,10>`) needs to be wider than the worst-case sum of products in any layer. Re-run `convert_to_hls.py` with the constants `ACCUM_PRECISION = "fixed<24,12>"` set higher.

**Q: Vivado says my IP doesn't fit (DSP > 100%)**
A: Re-run `convert_to_hls.py --reuse-factor 2` (or 4). This serializes multipliers — fewer DSPs, more cycles. Still well within the 200 ms budget.

**Q: Cmod A7 `DONE` LED doesn't light**
A: 
1. USB cable is charge-only, not data — try another cable.
2. Bitstream targeting wrong part — check `--board` flag matched the actual board.
3. Re-program; sometimes the first attempt times out.

**Q: Vivado doesn't see the Cmod A7 in Hardware Manager**
A:
- Linux: udev rules from 2.5 not applied. `sudo udevadm control --reload-rules`, log out + in.
- Windows: re-install Digilent Adept 2 Runtime.
- Both: `lsusb` (Linux) or Device Manager (Windows) should show `Digilent` device. If not, USB driver missing.

**Q: Synthesis takes forever and runs out of RAM**
A: 16 GB is the bare minimum. Close everything else. Or use the `c6i.4xlarge` AWS EC2 image with a Vivado AMI for a one-shot build — costs ~$1.

**Q: I'd rather use the ZedBoard than Cmod A7**
A: The Mac side: `python convert_to_hls.py --board zedboard --backend Vitis`. The teammate side: install Avnet ZedBoard board files instead of Digilent's, and Section 4.4's BRAM testbench style still works (the ZedBoard has the same BRAM/LED resources). The Zynq's ARM cores are an *additional* deployment option you can use if you want, not a requirement.

**Q: I'd rather use Vivado 2020.2 (the version this repo originally documented)**
A: Mac side: `python convert_to_hls.py --board cmod-a7-35 --backend Vivado` (NOT Vitis). Teammate side: use `vivado_hls` instead of `vitis_hls` in Section 3. Everything else identical.

---

## Section 7 — Files Produced This Phase

```
extract_weights.py        Keras-2 step: pulls trained weights from QAT model → .npz
convert_to_hls.py         Keras-3 step: rebuilds, loads weights, runs hls4ml
phase3.md                 this document

out/eew_cnn_float_weights.npz    weight tensors keyed by 'layer/idx' (~50 KB)
out/hls_project/                  Vivado/Vitis HLS project tree (~5 MB, 120 files)
```

To re-target a different board:
```bash
python convert_to_hls.py --board <preset> --backend <Vivado|Vitis>
```
Presets: `cmod-a7-35` (default), `arty-a7-35`, `arty-a7-100`, `nexys-a7-100`, `zedboard`. Or use `--part <xc7a...>` for a fully custom chip.

---

## Section 8 — What's Next

- **Phase 3a (you are here)**: bitstream that runs the CNN on demand from a button or VIO.
- **Phase 3b**: continuous mode — read 100 Hz samples from an ADC into a rolling 200-sample BRAM buffer, trigger the CNN every 20 samples (200 ms cadence per Phase 2.6), drive a GPIO pin high when an alert fires.
- **Phase 4**: latency budget characterisation on a real seismometer feed, false-alarm tuning under operational conditions.
