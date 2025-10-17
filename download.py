import kagglehub
import pandas as pd
import os

# 1️⃣ 指定本地保存目录（推荐使用原始字符串 r"" 以避免转义问题）
save_dir = r"D:\occ\cafe-sales-cleaning-trainnig"

# 2️⃣ 下载 Kaggle 数据集
print("📥 Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download(
    "ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training",
    force_download=False  # 若已存在则跳过下载
)

print(f"✅ Dataset downloaded to cache: {dataset_path}")

# 3️⃣ 将数据集文件复制到指定目录
os.makedirs(save_dir, exist_ok=True)

# 获取 CSV 文件名
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in the dataset folder!")

# 取第一个 CSV 文件
csv_filename = csv_files[0]
src_path = os.path.join(dataset_path, csv_filename)
dst_path = os.path.join(save_dir, csv_filename)

# 拷贝到目标路径
import shutil
shutil.copy(src_path, dst_path)
print(f"✅ Copied {csv_filename} to {save_dir}")

# 4️⃣ 加载 CSV 数据
df = pd.read_csv(dst_path)
print("✅ Successfully loaded the dataset!\n")

# 5️⃣ 打印前 5 条记录
print("First 5 records:")
print(df.head())

# 6️⃣ 可选：打印数据集基本信息
print("\n📊 Dataset Info:")
print(df.info())
