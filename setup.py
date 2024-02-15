import subprocess
import os
import zipfile
import shutil

def install_packages():
    # Install required Python packages
    subprocess.check_call(["pip", "install", "requests"])

def download_and_extract(download_url, extraction_path):
    import requests

    # Download the zip file
    print("Downloading zip file...")
    r = requests.get(download_url, stream=True)
    zip_path = os.path.join(extraction_path, 'downloaded.zip')
    with open(zip_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

    # Unzip the file
    print("Unzipping the file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    # Optionally, remove the zip file after extraction
    os.remove(zip_path)

    print("Setup completed.")

def main():
    # Install required packages
    install_packages()

    # Define the download URL
    download_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"

    # Determine the extraction path dynamically based on the location of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Download and extract
    download_and_extract(download_url, script_dir)

if __name__ == "__main__":
    main()
