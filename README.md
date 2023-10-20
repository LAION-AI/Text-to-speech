## Audio pipeline for TTS Datasets

This project aims to provide high level APIs for various feature engineering techniques for processing audio files. The implementation follows a modularized and config based approach, so any dataset processing pipeline can be built and managed using the same.

### Creating new pipeline 

Creat an yaml file under `config/pipelines/` directory with the following structure

```
pipeline:
  loader:
    target: manager.Downloader
    args:
      configs:
        - config/datasets/data-config.yaml
      save_dir: raw_data/yt_data
  manager:
    target: manager.YoutubeRunner
  processors:
    - name: chunking
      target: modules.AudioChunking
      args:
        model_choice: pydub_chunking
    - name: denoise_audio
      target: modules.DenoiseAudio
      args:
        model_choice: meta_denoiser_dns48
    - name: audio_superres
      target: modules.SuperResAudio
      args:
        model_choice: voicefixer
    ...
```

**Pipeline Schema**

**Loader**: The entry point for fetching data from various sources like S3, local systems, or blob storage.
**Manager**: Specifies the manager class responsible for running the pipeline.
**Processors**: An ordered list of processors to apply for feature extraction or other manipulations.


If new feature extractors or manager are required for your needs, check the `modules/` directory for understanding the structure and create or update the objects as needed.

### Run pipleines

```
python workers/pipeline.py --configs <space separated path to config(s)> 
```
## Acknowledgements
credit a few of the amazing folks in the community that have helped to this happen:
- [bud-studio](https://bud.studio/) - For providing a initial framework
