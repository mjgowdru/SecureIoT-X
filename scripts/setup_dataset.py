"""
SecureIoT-X Dataset Setup & Downloader Utility.

Usage:
    python scripts/setup_dataset.py --generate-sample   # Instantly generate sample RT_IOT2022.csv
    python scripts/setup_dataset.py --download-uci      # Download official RT-IoT2022 from UCI ML Repo
    python scripts/setup_dataset.py --help
"""

import argparse
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("secureiot.dataset_setup")

DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

UCI_RT_IOT2022_URL = "https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip"


def generate_sample_dataset(output_filename: str = "RT_IOT2022.csv", n_samples: int = 10000) -> Path:
    """Generate a realistic sample IoT dataset and save it to datasets directory."""
    import numpy as np
    import pandas as pd

    logger.info(f"Generating synthetic {output_filename} with {n_samples} samples...")
    np.random.seed(42)

    attack_types = [
        "Normal",
        "DOS_SYN_Hping",
        "Thing_Speak",
        "ARP_poisioning",
        "MQTT_Publish",
        "NMAP_UDP_SCAN",
        "NMAP_XMAS_TREE_SCAN",
        "Wipro_bulb",
    ]
    weights = [0.45, 0.15, 0.10, 0.08, 0.08, 0.06, 0.04, 0.04]
    labels = np.random.choice(attack_types, size=n_samples, p=weights)

    data = {
        "flow_duration": np.abs(np.random.exponential(10, n_samples)),
        "fwd_pkts_tot": np.random.randint(1, 500, n_samples),
        "bwd_pkts_tot": np.random.randint(0, 300, n_samples),
        "fwd_data_pkts_tot": np.random.randint(0, 400, n_samples),
        "bwd_data_pkts_tot": np.random.randint(0, 200, n_samples),
        "fwd_pkts_per_sec": np.abs(np.random.normal(50, 30, n_samples)),
        "bwd_pkts_per_sec": np.abs(np.random.normal(30, 20, n_samples)),
        "flow_pkts_per_sec": np.abs(np.random.normal(80, 40, n_samples)),
        "fwd_byts_per_sec": np.abs(np.random.normal(5000, 3000, n_samples)),
        "bwd_byts_per_sec": np.abs(np.random.normal(3000, 2000, n_samples)),
        "fwd_pkts_payload_avg": np.abs(np.normal(200, 100, n_samples) if hasattr(np, 'normal') else np.random.normal(200, 100, n_samples)),
        "bwd_pkts_payload_avg": np.abs(np.random.normal(150, 80, n_samples)),
        "proto": np.random.choice([6, 17, 1], n_samples, p=[0.6, 0.3, 0.1]),
        "dst_port": np.random.choice([80, 443, 1883, 8080, 22, 23, 53, 8883], n_samples),
        "conn_state_num": np.random.randint(0, 12, n_samples),
        "Attack_type": labels,
    }

    df = pd.DataFrame(data)
    out_path = DATASETS_DIR / output_filename
    df.to_csv(out_path, index=False)
    logger.info(f"Dataset successfully saved to: {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    return out_path


def download_uci_dataset() -> None:
    """Download and extract RT-IoT2022 from UCI Machine Learning Repository."""
    zip_path = DATASETS_DIR / "rt-iot2022.zip"
    logger.info(f"Downloading RT-IoT2022 dataset from {UCI_RT_IOT2022_URL}...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, block_num * block_size / total_size * 100)
            sys.stdout.write(f"\rDownloading: {percent:5.1f}% ({block_num * block_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(UCI_RT_IOT2022_URL, zip_path, reporthook=_progress)
        print("\nDownload complete! Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATASETS_DIR)
        logger.info(f"Extracted to {DATASETS_DIR}")

        # Check for extracted csv
        csv_files = list(DATASETS_DIR.glob("*.csv"))
        logger.info(f"Found CSV files: {[f.name for f in csv_files]}")
    except Exception as e:
        logger.error(f"Failed to download automatically: {e}")
        logger.info("Alternative: Download manually from UCI ML Repository or Kaggle.")


def main():
    parser = argparse.ArgumentParser(description="SecureIoT-X Dataset Setup Tool")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a sample RT_IOT2022.csv with 10,000 samples for immediate training",
    )
    parser.add_argument(
        "--download-uci",
        action="store_true",
        help="Download RT-IoT2022 from UCI Machine Learning Repository",
    )

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset()
    elif args.download_uci:
        download_uci_dataset()
    else:
        print("SecureIoT-X Dataset Manager")
        print("-" * 50)
        print("Available actions:")
        print("  1. Generate sample dataset for quick testing:")
        print("     python scripts/setup_dataset.py --generate-sample\n")
        print("  2. Download official RT-IoT2022 from UCI:")
        print("     python scripts/setup_dataset.py --download-uci\n")
        print("  3. Manual Download Links:")
        print("     - RT-IoT2022 (Primary): https://archive.ics.uci.edu/dataset/942/rt-iot2022")
        print("       or Kaggle: https://www.kaggle.com/datasets/mssmartypants/rt-iot2022")
        print("       -> Place as: datasets/RT_IOT2022.csv\n")
        print("     - IoT-23: https://www.stratosphereips.org/datasets-iot23")
        print("       -> Place as: datasets/IoT23.csv\n")
        print("     - BoT-IoT: https://research.unsw.edu.au/projects/bot-iot-dataset")
        print("       -> Place as: datasets/BoT_IoT.csv\n")


if __name__ == "__main__":
    main()
