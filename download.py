import kagglehub
import pandas as pd
import os


save_dir = r"D:\occ\cafe-sales-cleaning-trainnig"

# Download kaggle dataset
print("📥 Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download(
    "ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training",
    force_download=False  # pass if already exist
)

print(f"✅ Dataset downloaded to cache: {dataset_path}")

os.makedirs(save_dir, exist_ok=True)


csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in the dataset folder!")

# Take the first csv file
csv_filename = csv_files[0]
src_path = os.path.join(dataset_path, csv_filename)
dst_path = os.path.join(save_dir, csv_filename)

import shutil
shutil.copy(src_path, dst_path)
print(f"✅ Copied {csv_filename} to {save_dir}")

# Load csv dataset
df = pd.read_csv(dst_path)
print("✅ Successfully loaded the dataset!\n")

# print the top 5 record
print("First 5 records:")
print(df.head())

# print basic information of the dataset
print("\n📊 Dataset Info:")
print(df.info())
