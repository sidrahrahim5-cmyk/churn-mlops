# src/data/download_data.py

# This file downloads our dataset.We will use the built-in python library

import urllib.request
import os
import sys


def download_dataset():
    """
    This function downloads the dataset
    """
    # URL for datasource
    url=(
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/" 
        "master/data/Telco-Customer-Churn.csv"
    )

    # Path to save data
    save_path="data/raw/churn_data.csv"

    # Create if folder does not exist
    os.makedirs("data/raw",exist_ok=True)

    # Don't download if file already exists
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return save_path
    
    # Download data
    print("Data is downloading...")

    try:
        urllib.request.urlretrieve(url, save_path)

        file_size=os.path.getsize(save_path)
        file_size_kb=file_size/1024

        print(f" Data has been downloaded.")
        print(f" Location: {save_path}")
        print(f" Size: {file_size_kb:.1f} KB")

        return save_path
    
    except Exception as e:
        print(f" Data not downloaded: {e}")
        sys.exit(1)

if __name__=="__main__":
    path= download_dataset()
    print(f" Ready: {path}")