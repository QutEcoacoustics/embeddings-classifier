# resultsExplorer

Install from the monorepo subdirectory:

```r
install.packages("remotes")
remotes::install_github(
  "QutEcoacoustics/embeddings-classifier/r/resultsExplorer"
)
```

Launch it:

```r
resultsExplorer::launch()
```

The browser UI reads CSV files locally and does not upload them. It expects:

- `label`
- `score`
- `audio_recording_id`
- `start_offset_seconds`
- `end_offset_seconds`

The package serves the bundled files on `127.0.0.1:8000` with the isolation
headers required by `oe-spectrogram`.
