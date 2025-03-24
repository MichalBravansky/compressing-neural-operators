import sys
sys.path.append("C:\\Users\\ahmed\\UCL-Ibsa-Ahmed\\Year_3\\COMP0030-31\\COMP0031-Model-Compression-on-Neural-Operator")
from pathlib import Path
from neuralop.data.datasets.web_utils import download_from_zenodo_record
data_root = Path("neuralop/data/datasets/data")
data_root.mkdir(parents=True, exist_ok=True) 
files_to_download = ["darcy_128.tgz"]
zenodo_record_id = "12784353"
download_from_zenodo_record(record_id=zenodo_record_id,
                            root=data_root,
                            files_to_download=files_to_download)
print("✅ darcy_128.tgz downloaded successfully!")