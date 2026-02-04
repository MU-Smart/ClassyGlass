# ClassyGlass Dataset

The **ClassyGlass Dataset** contains multimodal time-series data collected from wearable smart glasses used by real participants. The dataset includes synchronized measurements from multiple onboard sensors, including an accelerometer, gyroscope, magnetometer, and pressure sensor. The data are organized by subject and recording session to support reproducible research and machine learning workflows.

## Index

- [Overview](#overview)
- [Motivation and Research Use](#motivation-and-research-use)
- [Dataset Contents](#dataset-contents)
- [Directory Structure](#directory-structure)
- [File Format](#file-format)
- [Sensor Specifications](#sensor-specifications)
- [Data Collection Protocol](#data-collection-protocol)
- [Preprocessing](#preprocessing)
  - [Time Synchronization](#time-synchronization)
  - [Recommended Data Splits](#recommended-data-splits)
- [Usage Notes](#usage-notes)
  - [Metadata Description](#metadata-description)
- [Ethics and Consent](#ethics-and-consent)
- [License and Citation](#license-and-citation)

---

## Overview

ClassyGlass is a comprehensive wearable sensing dataset designed for activity recognition, behavioral analysis, and multimodal sensor fusion research. It provides temporally aligned sensor streams recorded during controlled and semi-naturalistic sessions.

This README describes the dataset structure, file formats, sensor specifications, and recommended usage practices.

## Motivation and Research Use

The ClassyGlass Dataset was collected to support research in wearable computing, human activity recognition, and multimodal sensor fusion using head-mounted devices. Unlike wrist- or phone-based datasets, this dataset captures motion and environmental signals from smart glasses, enabling analysis of head dynamics, fine-grained user behavior, and context-aware inference.

The dataset is suitable for tasks such as:

- Activity and gesture recognition
- Time-series classification and segmentation
- Multimodal sensor fusion
- Representation learning on wearable sensor data

---

## Dataset Contents

- **Multimodal sensor data:**
  - Accelerometer
  - Gyroscope
  - Magnetometer
  - Pressure sensor
- **Per-session CSV files** with columns (epoch, time, elapsed, x-axis, y-axis, z-axis)
- **Metadata** describing participants and recording sessions

---

## Directory Structure

```
ClassyGlass/
├── Datasets/
│   ├── Dataset_1A
│   │   └── *.csv
│   ├── Dataset_1B
│   │   └── *.csv
│   └── Dataset_2
│       └── *.csv
├── README.md

```

---

## File Format

All sensor data are stored in **CSV format** with header rows. Each row corresponds to a single timestamped observation.

### Filename Format

Each data file follows a structured naming convention that encodes metadata about
the device, recording time, sensor type, and sampling configuration.

```text
<experiment_id>_<device>_<timestamp>_<device_id>_<sensor_type>_<sampling rate>Hz_<firmware_version>.csv
```

Typical columns include:

- `timestamp` — Time of measurement (ISO 8601 or Unix epoch in milliseconds)
- `acc_x`, `acc_y`, `acc_z` — Linear acceleration (m/s²)
- `gyr_x`, `gyr_y`, `gyr_z` — Angular velocity (rad/s or deg/s)
- `mag_x`, `mag_y`, `mag_z` — Magnetic field strength (µT)
- `pressure` — Atmospheric pressure (hPa)

> **Note:** Please verify and update units or timestamp formats if they differ from those listed above.

---

## Sensor Specifications

- **Accelerometer**  
   Model: `<MODEL>`  
   Resolution: `<RESOLUTION>`  
   Sampling rate: `<SAMPLING_RATE>` Hz

- **Gyroscope**  
   Model: `<MODEL>`  
   Resolution: `<RESOLUTION>`  
   Sampling rate: `<SAMPLING_RATE>` Hz

- **Magnetometer**  
   Model: `<MODEL>`  
   Resolution: `<RESOLUTION>`  
   Sampling rate: `<SAMPLING_RATE>` Hz

- **Pressure Sensor**  
   Model: `<MODEL>`  
   Resolution: `<RESOLUTION>`  
   Sampling rate: `<SAMPLING_RATE>` Hz

---

## Data Collection Protocol

- **Number of Participants**: `<N_SUBJECTS>`
- **Sessions per Participant**: `<N_SESSIONS>`
- **Recording Duration**: `<MIN–MAX_DURATION>`
- **Device Placement**: Smart glasses worn on the face (describe exact sensor placement if relevant)
- **Activities**: `<LIST_OF_ACTIVITIES_OR_TASKS>`

---

## Preprocessing

### Time Synchronization

All sensor streams were synchronized using a common system clock on the wearable device. Timestamps were recorded at the time of acquisition in Unix epoch milliseconds. Sensor streams were aligned post-collection using linear interpolation to the highest sampling-rate sensor.

- **Filtering / Calibration**: `<LOW-PASS / HIGH-PASS / SENSOR CALIBRATION DETAILS>`

### Recommended Data Splits

For fair evaluation and subject-independent analysis, we recommend:

- **Subject-independent splits**, where subjects in the test set do not appear in training.
- A typical split of 70% train, 15% validation, and 15% test by subject.

Session-level splits are also possible for within-subject modeling.

---

## Usage Notes

- Ensure all timestamps are converted to a consistent format and time zone before analysis.
- When resampling or downsampling, preserve temporal alignment across sensor modalities.
- Example usage in Python:

```python
import pandas as pd
df = pd.read_csv("data/subject_01/session_01.csv")
```

### Metadata Description

- **participants.csv**
  - `subject_id`
  - `age`
  - `sex`
  - `handedness`
  - `consent`
- **sessions.csv**
  - `session_id`
  - `subject_id`
  - `activity`
  - `start_time`
  - `end_time`
  - `notes`

---

## Ethics and Consent

All participants provided informed consent prior to data collection. The study was conducted under the protocol `<PROTOCOL_ID_OR_DESCRIPTION>`. No personally identifiable information (PII) is included in this dataset.

---

## License and Citation

- **License**: `<LICENSE_NAME or URL>`

If you use this dataset in your research, please cite:

`<CITATION_TEXT or BIBTEX>`
