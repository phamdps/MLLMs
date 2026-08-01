import os
import urllib.request

DATA_DIR = "data/raw/METR-LA"
os.makedirs(DATA_DIR, exist_ok=True)

# Active direct GitHub raw download links
URLS = {
    "metr_la.h5": "https://raw.githubusercontent.com/transpaper/GCNN/master/data/METR-LA_traffic_speed/metr-la.h5",
    "adj_mx.pkl": "https://raw.githubusercontent.com/chnsh/DCRNN_PyTorch/pytorch_scratch/data/sensor_graph/adj_mx.pkl"
}

def download_metr_la():
    # Set standard User-Agent header to avoid 403/404 CDN blocking from urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64)')]
    urllib.request.install_opener(opener)

    for fname, url in URLS.items():
        out_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(out_path):
            print(f"Downloading {fname} from {url}...")
            try:
                urllib.request.urlretrieve(url, out_path)
                print(f"Successfully downloaded {fname} to {out_path}.")
            except Exception as e:
                print(f"Failed to download {fname}: {e}")
        else:
            print(f"File {fname} already exists at {out_path}.")

if __name__ == "__main__":
    download_metr_la()