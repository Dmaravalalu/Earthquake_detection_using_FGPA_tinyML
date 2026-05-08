# Phase 3 — FPGA Synthesis & ZedBoard Deployment

This document is the **end-to-end deployment guide** for putting the trained earthquake-detection CNN onto a ZedBoard (Zynq-7020) FPGA. It assumes **no prior Vivado experience**.

**Read first:** `phase1.md`, `phase2.md`, `phase2_5.md`, `phase2_6.md` for the model and benchmark context.

---

## TL;DR for the Impatient

```
┌────────────────────┐    convert_to_hls.py    ┌──────────────────┐
│  Trained Keras     │  ───────────────────►   │  Vivado HLS C++  │
│  QAT model (Mac)   │                          │  project tree   │
│  out/eew_cnn_qat.. │                          │  out/hls_project │
└────────────────────┘                          └──────────────────┘
                                                          │
                              transfer to Linux/Windows   │
                              host (Vivado not on macOS)  ▼
                                                ┌──────────────────┐
                                                │  Vivado HLS:     │
                                                │  csim → csynth   │
                                                │  → cosim →       │
                                                │  export IP       │
                                                └──────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  Vivado IDE:     │
                                                │  block design,   │
                                                │  AXI plumbing,   │
                                                │  bitstream       │
                                                └──────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  ZedBoard:       │
                                                │  flash bitstream │
                                                │  feed waveform   │
                                                │  read prediction │
                                                └──────────────────┘
```

The script is on this Mac. **Vivado runs only on Linux or Windows** — you'll need a separate machine (or VM) for the synthesis steps.

---

## Section 0 — What Vivado Even Is

If you've never used FPGA tooling, here's the mental model.

### What is an FPGA?

An FPGA is a chip full of generic digital building blocks — lookup tables (LUTs), flip-flops (FFs), DSP slices for arithmetic, block RAM (BRAM) for storage. You write a configuration ("bitstream") that wires these blocks into a custom digital circuit — so the same chip can be a video decoder one day and an earthquake detector the next.

### What is Vivado?

**Vivado** is Xilinx's (now AMD's) software for designing FPGA bitstreams. It's actually a *suite* of tools, and the names matter because they do different things:

| Tool | What it does | When you use it |
|---|---|---|
| **Vivado HLS** (or **Vitis HLS** in newer releases) | Translates **C++** describing an algorithm into **synthesizable RTL** (Verilog/VHDL). Produces an "IP block" you can drop into a Vivado design. | This is where `hls4ml`'s output goes first. |
| **Vivado** (the IDE) | The main FPGA development environment. Lets you draw a **block design** (your IP + memory + AXI buses + processor + clocks), simulate, and **synthesize the whole system to a bitstream**. | After Vivado HLS produces the IP, you wire it up here. |
| **Vitis** (a.k.a. SDK) | C/C++ environment for the **ARM cores** that live next to the FPGA fabric on Zynq chips like the Z-7020 on ZedBoard. | If you want a small ARM program to feed data to the FPGA and read results. |

You'll touch all three, in that order: HLS → Vivado IDE → Vitis.

### What is hls4ml?

[hls4ml](https://fastmachinelearning.org/hls4ml/) is a Python library that automatically generates Vivado HLS C++ from a Keras model. It writes the C++ for every Conv2D, BatchNorm, ReLU, MaxPool, GlobalAveragePooling2D, and Dense layer in our network, including the trained weights as `static const` arrays. We use it so you don't have to hand-write HLS for an 8K-parameter network.

### Why does this involve so much software?

Going from a `.h5` file to a flashing FPGA is a multi-stage compilation pipeline, each stage solving a different abstraction problem:

```
Keras  →  C++  →  RTL (Verilog)  →  netlist  →  bitstream
       ↑       ↑                 ↑           ↑
     hls4ml  Vivado HLS       Vivado synth  Vivado
                              + place&route  bitgen
```

This document walks you through each stage. The good news: 80% of the work is automated by hls4ml + Vivado's tcl scripts. You mostly click through GUIs and wait.

---

## Section 1 — On Your Mac: Generate the HLS Project

### 1.1 Prerequisites

You should already have these from earlier phases:

```
~/Desktop/idp/
├── .venv-tf/                                  Phase 2 venv (Python 3.11 + TF 2.16)
├── out/eew_cnn_qat.keras                      Phase 2 trained QAT model (256 KB)
├── train_cnn.py                                builds the architecture
└── ~/Downloads/archive/merge.{csv,hdf5}        STEAD source (only needed for testbench data)
```

Plus, install hls4ml (one-time):
```bash
source .venv-tf/bin/activate
pip install hls4ml
```

### 1.2 Run the conversion

```bash
cd ~/Desktop/idp
source .venv-tf/bin/activate
python convert_to_hls.py
```

**What happens internally:**
1. The script needs the QAT weights, but `hls4ml` only understands clean Keras 3 models (no QAT wrappers). It first runs `extract_weights.py` as a subprocess in **Keras-2 mode** (with `TF_USE_LEGACY_KERAS=1`) to load the QAT checkpoint and dump the trained weights into `out/eew_cnn_float_weights.npz`.
2. The main script then runs in **Keras-3 mode**, rebuilds the same architecture, loads the weights from the npz, and feeds the model to `hls4ml`.
3. `hls4ml` writes a complete Vivado HLS project tree to `out/hls_project/`.

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
│   └── weights/                    .h files with the trained weights as
│                                   static const fixed<8,1> arrays
├── tb_data/                        test bench input/output samples
├── eew_cnn_test.cpp                C-simulation test bench
├── eew_cnn_bridge.cpp              Python ↔ HLS bridge
├── build_prj.tcl                   "do everything" tcl script for Vivado HLS
├── project.tcl                     Vivado HLS project setup
├── vivado_synth.tcl                post-HLS synthesis tcl
├── hls4ml_config.yml               human-readable config snapshot
└── keras_model.keras               the source Keras model (reference)
```

The `firmware/` directory is the heart of it. `eew_cnn.cpp` is roughly:

```cpp
void eew_cnn(input_t  input_1[N_INPUT_1_1*N_INPUT_2_1],
             result_t layer16_out[N_LAYER_16]) {
    #pragma HLS INTERFACE ap_vld port=input_1,layer16_out
    #pragma HLS DATAFLOW
    static input_t reshape_out[200*1*3];
    nnet::reshape<...>(input_1, reshape_out);
    static layer1_t layer1_out[200*1*16];
    nnet::conv_2d_cl<input_t, layer1_t, config1>(reshape_out, layer1_out, w1, b1);
    // ... BN, ReLU, MaxPool, Conv ×2 more, GAP, Dense, Sigmoid ...
}
```

Per the QAT settings: weights are `fixed<8,1>` (8-bit signed, 1 integer + 7 fraction), accumulators are `fixed<20,10>` (wide enough that MAC sums don't saturate).

### 1.3 Smoke-test on the Mac (optional but recommended)

Before transferring to the Vivado host, you can verify the rebuilt float Keras model is still classifying correctly. The conversion script does this automatically and prints a sanity range; you should also see ~98% accuracy on real samples:

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

Should print `accuracy: ~0.98`. If it doesn't, the weight extraction is broken — re-run `convert_to_hls.py`.

### 1.4 What gets copied to the Vivado host

Zip just the `out/hls_project/` tree (≈ 5 MB):
```bash
cd ~/Desktop/idp/out
zip -r hls_project.zip hls_project
# or tar:
tar czf hls_project.tar.gz hls_project
```

Transfer that single archive to your Linux/Windows machine. You don't need anything else from this Mac.

---

## Section 2 — Setting Up the Vivado Host

Vivado **does not run on macOS**. You have three realistic options:

| Option | Pros | Cons |
|---|---|---|
| **Linux PC / lab machine** | Easiest if you have access. Vivado is Linux-native. | Need separate hardware. |
| **Linux VM on the Mac** (UTM / Parallels) | Self-contained. | Vivado wants ≥ 16 GB RAM and ≥ 80 GB disk free; performance is slow under emulation on Apple Silicon. |
| **Windows PC** | Vivado has full Windows support. | Slightly different paths/scripts but everything works. |

For ZedBoard (Zynq-7020), you need:

### 2.1 Vivado / Vivado HLS Installation

- **Recommended version: Vivado 2020.2 ML Edition** (free WebPACK license is enough — Z-7020 is supported in WebPACK). This is the last major Vivado release that still ships **Vivado HLS** as a separate tool. Newer versions (2021.x+) replace Vivado HLS with **Vitis HLS**, which is mostly compatible but has small differences hls4ml may not have caught up to.
- Download from [https://www.xilinx.com/support/download.html](https://www.xilinx.com/support/download.html) (requires a free AMD/Xilinx account).
- Pick the **Web Installer** for your OS. During install, check:
  - Vivado Design Suite ✓
  - Vivado HLS ✓
  - SDK / Vitis (for the ARM-side host code)
  - Devices: Zynq-7000 (this includes the Z-7020 used by ZedBoard)
- Disk usage: ~50 GB.

After install, source the settings script in every shell where you'll use Vivado:

```bash
# Linux
source /opt/Xilinx/Vivado/2020.2/settings64.sh

# Windows (cmd)
"C:\Xilinx\Vivado\2020.2\settings64.bat"
```

Verify:
```bash
vivado_hls -version    # should print 2020.2
vivado -version
```

### 2.2 Cable Drivers (for talking to the ZedBoard)

- Linux: install Xilinx cable drivers (`/opt/Xilinx/Vivado/2020.2/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers`)
- Windows: drivers ship with the installer, but you may need to plug in the ZedBoard's `PROG-USB` port once and let Windows install them.

### 2.3 The ZedBoard

ZedBoard reference: [zedboard.org](http://zedboard.org/product/zedboard).

Connect three things:
1. **Power** — the wall adapter. Set `JP1` to `WALL`.
2. **PROG-USB** — micro-USB from the labeled `PROG-USB` jack to your host. This is how Vivado talks to the JTAG and how the ARM serial console comes out.
3. **Boot mode** — set the `MIO 4..0` jumpers to "JTAG" mode for now (means: get programmed by Vivado over USB, not from SD card). The ZedBoard's silkscreen shows the jumper pattern: `JTAG = ON ON ON ON ON` for `MIO4..MIO0`.

---

## Section 3 — On the Vivado Host: HLS → IP

You have the unzipped `hls_project/` tree on the Vivado-host machine. Now run Vivado HLS to turn the C++ into a synthesizable IP.

### 3.1 Run the all-in-one tcl script

```bash
cd hls_project
vivado_hls -f build_prj.tcl "csim=1 synth=1 cosim=1 export=1"
```

Four stages, each takes a few minutes:

| Stage | What it does | Why |
|---|---|---|
| `csim=1` | Compiles the HLS C++ + test bench, runs as a normal C++ program against `tb_data/`. Compares HLS-fixed-point output against Keras-float reference. | If C-sim fails, the model conversion is broken — fix that before going further. |
| `synth=1` | High-Level Synthesis: turns the C++ into RTL (Verilog), reports estimated LUTs / FFs / BRAM / DSPs / clock period. | Tells you whether the model fits on the chip and at what speed. |
| `cosim=1` | RTL co-simulation: runs the generated Verilog against the same test bench. Verifies that synthesis didn't break anything. | Catches HLS bugs (rare) and synthesis pragma issues. |
| `export=1` | Packages the RTL as a Vivado IP block. Output: `eew_cnn_prj/solution1/impl/ip/xilinx_com_hls_eew_cnn_1_0.zip` | This `.zip` is what you import into the Vivado block design. |

If any stage fails, look at `eew_cnn_prj/solution1/<stage>/report/eew_cnn_<stage>.rpt`. Common gotchas:
- "Cannot find vivado_hls" → forgot to `source settings64.sh`
- C-sim mismatch with reference → an unsupported layer (rare with our architecture). Double-check `hls4ml_config.yml` precision settings.
- Resource overflow ("LUT 120% of available") → re-run `convert_to_hls.py` with a higher `default_reuse_factor` (e.g., 4) which trades latency for fewer parallel multipliers. Edit the script, rebuild HLS project, redo this section.

For our 8K-param model on Z-7020, expected resource usage:
- LUTs: ~15-25% of 53,200
- FFs: ~10-15% of 106,400
- BRAM: ~5% of 140 blocks
- DSPs: ~30-50% of 220
- Latency: ~50-150 clock cycles per inference at 100 MHz = 0.5-1.5 µs/inference, well within the 200 ms (5 inferences/sec) target from Phase 2.6.

### 3.2 GUI alternative

If you'd rather click through:
```bash
vivado_hls
```
Then **File → Open Project**, point to `hls_project/eew_cnn_prj`. Use the toolbar buttons:
- "Run C Simulation" (green play)
- "Run C Synthesis" (yellow gear)
- "Run C/RTL Cosimulation"
- "Export RTL"

The IP `.zip` ends up at the same path as the tcl path above.

---

## Section 4 — On the Vivado Host: Build the Block Design

Now you have an IP block. Vivado IDE wires it up with everything else (the ARM core, AXI buses, DMA, GPIO, clocks) into a complete system that will run on the ZedBoard.

### 4.1 Create a new Vivado project

```bash
vivado
```
- **Create New Project** → Name: `eew_zedboard` → choose a project location.
- Project type: **RTL Project**
- Add Sources / Add Constraints: skip both for now.
- **Default Part**: search for "ZedBoard" in the **Boards** tab. If you don't see it, you may need to add the [ZedBoard board files](https://github.com/Avnet/bdf) — drop the unzipped `bdf/zedboard/A.04` folder into `<Vivado-install>/data/boards/board_files/`.

### 4.2 Add the HLS-generated IP to the IP catalog

- **Project Manager → IP Catalog → right-click → Add Repository**
- Browse to the `hls_project/eew_cnn_prj/solution1/impl/ip/` folder. Vivado will scan it and find the `eew_cnn` IP.

### 4.3 Build a Block Design

- **Flow Navigator → IP Integrator → Create Block Design** → Name: `eew_bd`
- Click `+` to add IPs. The minimum set you need:
  1. **ZYNQ7 Processing System** — the hard ARM cores on the chip. Click "Run Block Automation" when it appears (uses ZedBoard preset config — clocks, DDR, peripherals all set up correctly).
  2. **eew_cnn** — your generated CNN IP.
  3. **AXI DMA** — moves input waveforms from DDR memory into the CNN, and reads output back. Default config: scatter-gather disabled, simple mode, 32-bit data width.
  4. **AXI Interconnect** — Vivado will add this automatically when it auto-connects.

- Click **Run Connection Automation** to let Vivado wire up:
  - PS DDR ↔ AXI DMA M_AXI_MM2S/S2MM (memory-to-stream and stream-to-memory)
  - AXI DMA M_AXIS_MM2S ↔ eew_cnn input
  - eew_cnn output ↔ AXI DMA S_AXIS_S2MM
  - All AXI-Lite control buses ↔ Zynq M_AXI_GP0
  - All clocks ↔ FCLK_CLK0
  - All resets ↔ FCLK_RESET0_N

The resulting block design looks like:

```
┌─────────────────────────┐    AXI-Lite      ┌──────────────┐
│ Zynq Processing System  │◀────────────────▶│  AXI DMA     │
│  - 2 × ARM Cortex-A9    │                  │              │
│  - DDR controller       │  M_AXI_HP0       │  MM2S ─────▶ │── AXIS ──▶ ┌──────────┐
│  - peripheral I/O       │◀────────────────▶│              │             │ eew_cnn  │
│                         │                  │  S2MM ◀───── │── AXIS ──── │ (HLS IP) │
└─────────────────────────┘                  └──────────────┘             └──────────┘
```

- **Validate Design** (F6). Should pass with green ticks.
- Right-click the block design in Sources, **Create HDL Wrapper** → Let Vivado manage wrapper.

### 4.4 Add the constraints file

ZedBoard pin assignments are pre-known. Either use the official Avnet `.xdc` file (in the bdf repo) or create one with at least:
- 100 MHz clock from the PS — usually no constraint needed because we're using `FCLK_CLK0` from the Zynq.

### 4.5 Generate the bitstream

- **Flow Navigator → Generate Bitstream**.
- Vivado runs synthesis → implementation → bitstream generation. **This takes 15–60 minutes** depending on host speed. Grab coffee.
- Output: `eew_zedboard.runs/impl_1/eew_bd_wrapper.bit`.

---

## Section 5 — On the Vivado Host: Program & Test the ZedBoard

### 5.1 Plug in and program

With the ZedBoard powered on and `PROG-USB` connected:

- **Flow Navigator → Open Hardware Manager → Open Target → Auto Connect**. Vivado finds the JTAG.
- **Program Device** → select `eew_bd_wrapper.bit`. It uploads in a few seconds.
- The blue LED `DONE` (D5) lights up = bitstream is running.

The FPGA is now wired up but doing nothing — it's waiting for the ARM core to send it data. That's the next step.

### 5.2 Write a tiny ARM program (Vitis SDK)

In Vivado: **File → Export → Export Hardware** (include bitstream) → produces `eew_zedboard.xsa`.

Open Vitis (Vivado's SDK companion):
```bash
vitis &
```
- **Create Application Project**:
  - Hardware: select the `.xsa` you just exported.
  - Domain: standalone (no Linux, just bare metal).
  - Template: **Empty Application (C)**.
- Add a `main.c`:

```c
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include <stdio.h>

#define WINDOW 200
#define CHANNELS 3

// Test waveform — fill with real preprocessed sample at runtime via UART or
// hard-code one for quick smoke test.
fixed_t input_buffer[WINDOW * CHANNELS] __attribute__((aligned(64)));
fixed_t output_buffer[1] __attribute__((aligned(64)));

XAxiDma dma;

int main(void) {
    XAxiDma_Config *cfg = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
    XAxiDma_CfgInitialize(&dma, cfg);

    // ... fill input_buffer with a 200×3 normalized window ...

    Xil_DCacheFlushRange((UINTPTR)input_buffer, sizeof(input_buffer));
    XAxiDma_SimpleTransfer(&dma, (UINTPTR)input_buffer,
                           sizeof(input_buffer), XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_SimpleTransfer(&dma, (UINTPTR)output_buffer,
                           sizeof(output_buffer), XAXIDMA_DEVICE_TO_DMA);
    while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA));
    Xil_DCacheInvalidateRange((UINTPTR)output_buffer, sizeof(output_buffer));

    printf("Prediction: %f\r\n", (float)output_buffer[0]);
    return 0;
}
```

- **Build Project** (Ctrl-B).
- Open a serial console (Vitis has one built in — `Window → Show View → Terminal` → connect to the ZedBoard's UART, 115200 baud, 8N1).
- **Run As → Launch Hardware (Single Application Debug)**. Your `printf("Prediction: …")` should appear on the serial terminal.

### 5.3 Verify against the Streamlit demo

Pick a known earthquake trace from Phase 2.5, copy its 200×3 window into `input_buffer`, re-run, and compare the FPGA's prediction to the Streamlit app's prediction (which runs the int8 TFLite on CPU). They should agree to ~0.01.

### 5.4 Continuous operation (optional)

For a real EEW system, you'd:
1. Hook a 100 Hz I²S or SPI ADC to a PL pin on the ZedBoard.
2. Maintain a 200-sample rolling buffer in PL BRAM.
3. Trigger the CNN every 20 samples (200 ms cadence — Phase 2.6's recommended stride).
4. Apply threshold + sustain on the output stream in PL.
5. Drive a GPIO pin high when an alert fires.

That's a separate project — out of scope for this initial deployment doc. The bitstream you've built is the core compute; wiring it to a real sensor stream is the next round.

---

## Section 6 — Where Things Go Wrong (FAQ)

**Q: `vivado_hls` says "WARNING: Cosimulation mismatch"**
A: Almost always a precision issue. Check `hls4ml_config.yml`: the accumulator precision (default `fixed<20,10>`) needs to be wider than the worst-case sum of products in any layer. Try `fixed<24,12>`.

**Q: Vivado says my IP doesn't fit (LUT > 100%)**
A: The model's tiny — this should not happen with the default config. If it does, in `convert_to_hls.py` change `default_reuse_factor=1` to `4` or `8` and regenerate. This serializes the multipliers — fewer LUTs, more cycles.

**Q: ZedBoard `DONE` LED doesn't light up**
A: Power-cycle, re-program. Check that `MIO 4..0` jumpers are in JTAG position. Check that `JP1` is set to `WALL` (not `USB` — USB power can't drive the FPGA reliably).

**Q: ARM-side `XAxiDma_Busy` never returns**
A: The DMA isn't seeing data come back from the IP. In Vivado, double-check that the `eew_cnn` IP's `ap_done` and `ap_vld` signals are connected, and that the IP is enabled in the AXI-Lite control register. Check `Xil_Out32(XPAR_EEW_CNN_0_S_AXI_CONTROL_BASEADDR + 0x00, 1);` to start it.

**Q: My host machine can't run Vivado (Mac, low RAM)**
A: Use the AWS Marketplace AMIs (Xilinx publishes "Vivado Design Suite" images). Spin up a `c6i.4xlarge` for an hour, install your project, run synthesis, download the bitstream. Costs roughly $1–2 per build.

---

## Section 7 — Files Added in Phase 3

```
extract_weights.py        Keras-2 step: pulls trained weights from QAT model → .npz
convert_to_hls.py         Keras-3 step: rebuilds, loads weights, runs hls4ml
phase3.md                 this document

out/eew_cnn_float_weights.npz    weight tensors keyed by 'layer/idx'
out/hls_project/                  Vivado HLS project tree (~5 MB)
```

---

## Section 8 — What's Next

- **Phase 3a (you are here)**: bitstream that runs the CNN on demand from ARM.
- **Phase 3b**: real-time pipeline — ADC → PL rolling buffer → CNN → threshold → GPIO alert.
- **Phase 4**: latency budget characterisation on a real seismometer feed, false-alarm tuning under operational conditions.
