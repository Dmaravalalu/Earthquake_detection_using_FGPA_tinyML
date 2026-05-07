This document explains required info to work with the dataset.

# Project: Real-Time Earthquake Early Warning System (FPGA Implementation)

**Base Source:** STEAD (STanford Earthquake Dataset)

**Status:** Curated Subset (500,000 Instances)

| Component  | File Type | Description |
|------------|-----------|-------------|
| Metadata   | .csv      | 35 columns of labels (Arrival times, Magnitude, SNR, Station ID) |
| Waveforms  | .hdf5     | Raw 3-channel seismic signals (Vertical, North-South, East-West) |

---

# Remote Setup: Laptop → Desktop Workflow

The 90 GB STEAD dataset lives on the desktop. We SSH in from the laptop, run the preparation script there, then transfer only the final ~1.2 GB subset back.

## A. Enable SSH on the Desktop (Windows)

If your desktop is running Windows 10/11:

1. Go to **Settings > System > Optional Features**.
2. Search for **OpenSSH Server** and click **Install**.
3. Open PowerShell as Administrator and run:

```powershell
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
```

4. Find your desktop's local IP address by typing `ipconfig` (e.g., `192.168.1.35`).

## B. Connect from the Laptop (VS Code Method)

1. Install the **"Remote - SSH"** extension in VS Code on your laptop.
2. Click the green **"Remote Window"** icon in the bottom-left corner.
3. Select **Connect to Host > Add New SSH Host**.
4. Type:

```
ssh Maravalalu-PC@192.168.1.35
```

5. VS Code will open a new window. You are now "inside" your desktop. Open the 90 GB dataset folder and run Python scripts directly there.

## C. Transfer the Resulting Subset (SCP)

Once `prepare_dataset.py` finishes and creates `X_train.npy` + `y_train.npy` on the desktop, run this from your **laptop's** terminal:

```bash
scp Maravalalu-PC@192.168.1.35:C:/path/to/output/X_train.npy  ~/Desktop/idp/
scp Maravalalu-PC@192.168.1.35:C:/path/to/output/y_train.npy  ~/Desktop/idp/
```

Replace `C:/path/to/output/` with the actual directory where the script saved the files.

**Desktop Info:**
- **IP:** `192.168.1.35`
- **Name:** `Maravalalu-PC`