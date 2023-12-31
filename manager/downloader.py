import os
import re
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
    def __init__(self, configs=None, save_dir="data/") -> None:
        if exists(configs):
            self.configs = load_configs(configs[0])
        else:
            self.configs = None
        self.save_dir = save_dir
    def download_from_youtube(self, url):
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        save_path = os.path.join(self.save_dir, f"{str(uuid4())}")
        file_path = stream.download(output_path=save_path)

        directory, filename = os.path.split(file_path)
        file_root, file_extension = os.path.splitext(filename)
        sanitized_root = re.sub(r"[^a-zA-Z0-9 ]", "", file_root)
        sanitized_root = sanitized_root.replace(" ", "_")
        
        new_filename = f"{sanitized_root}{file_extension}"
        new_file_path = os.path.join(directory, new_filename)
        
        os.rename(file_path, new_file_path)
        metadata = {"video": new_file_path, "source": url}
        return (metadata, True)

    def download_from_url(self, url):
        save_path = os.path.join(self.save_dir, f"{str(uuid4())}.zip")
        wget.download(url, save_path)
        metadata = {"file": save_path, "source": url}
        return (metadata, True)

    def unzip_file(self, path, save_dir):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)

    def walk_files(self, save_dir=None):
        for path in tqdm(self.configs.sources):
            save_dir = self.save_dir or self.configs.get("save_dir", [])
            if "youtube.com" in path:
                metadata, _ = self.download_from_youtube(path)
                yield metadata
            else:
                metadata, _ = self.download_from_url(path, save_dir)
                yield metadata

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
