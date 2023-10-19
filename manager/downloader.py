import os
import zipfile
from uuid import uuid4
from tqdm import tqdm
from pytube import YouTube
import wget
from urllib.parse import urlparse

from utils.io import load_configs
from utils.helpers import exists
from config import settings

class Downloader:
    def __init__(self, configs=None) -> None:
        if exists(configs):
            self.configs = [load_configs(config) for config in configs]
        else:
            self.configs = None

    def download_from_youtube(self, url, save_dir):
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        print(save_dir)
        save_path = os.path.join(save_dir, f"{str(uuid4())}.mp4")
        stream.download(output_path=save_path)
        metadata = {"video": save_path, "source": url}
        return (metadata, True)

    def download_from_url(self, url, save_dir):
        save_path = os.path.join(save_dir, f"{str(uuid4())}.zip")
        wget.download(url, save_path)
        metadata = {"file": save_path, "source": url}
        return (metadata, True)

    def unzip_file(self, path, save_dir):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)

    def walk_files(self, save_dir=None):
        for config in tqdm(self.configs):
            for path in tqdm(config.sources):
                save_dir = save_dir or config.get("save_dir", [])
                if "youtube.com" in path:
                    yield from self.download_from_youtube(path, save_dir)
                else:
                    yield from self.download_from_url(path, save_dir)

    def download_file(self, metadata, save_dir):
        if "source" in metadata:
            parsed_url = urlparse(metadata["source"])
            if "youtube.com" in parsed_url.netloc:
                return self.download_from_youtube(metadata["source"], save_dir)
            else:
                metadata, status = self.download_from_url(metadata["source"], save_dir)
                if status and metadata['file'].endswith('.zip'):
                    self.unzip_file(metadata['file'], save_dir)
                return (metadata, status)

    def __call__(self, metadata, save_dir=None):
        if not exists(save_dir):
            save_dir = os.path.join(settings.CACHE_DIR, "tmp", "downloads")
        return self.download_file(metadata, save_dir)
