import requests
import zipfile
import io

def download_uci_air_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    print("Downloading UCI Air Quality dataset...")
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("AirQualityUCI")
        print("Dataset downloaded and extracted to the 'AirQualityUCI/' folder.")
    else:
        print("Failed to download dataset. Status code:", response.status_code)

if __name__ == '__main__':
    download_uci_air_quality()