import gdown
import os
import yaml
import zipfile

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    raw_dir = params["data"]["raw"]
    os.makedirs(raw_dir, exist_ok=True)

    url = "https://drive.google.com/uc?id=1Zy99cVdudB3NYr-VqWg89YJRmcKIwCQr"  # Your zip link
    output_zip = os.path.join(raw_dir, "dataset.zip")

    print("⬇️ Downloading dataset...")
    gdown.download(url, output_zip, quiet=False)

    print("📦 Extracting zip file...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    print("✅ Dataset extracted successfully!")

if __name__ == "__main__":
    main()
