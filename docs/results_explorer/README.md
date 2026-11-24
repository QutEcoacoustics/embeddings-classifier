# Classifier results explorer

This is a static, client-only results explorer. It reads a CSV selected with a
file picker; the CSV is not uploaded.

## Publish

Deploy only this directory to a static host that supports custom headers, such
as Cloudflare Pages or Netlify. The included `_headers` file enables
cross-origin isolation, which is required by `oe-spectrogram`.

The Ecoacoustics Web Components bundle is loaded from jsDelivr. Audio segments
are loaded from the EcoSounds API.

## Local development

Opening `index.html` directly from `file://` is useful for checking the CSV
and histogram interface, but `oe-spectrogram` may fail because a local file
cannot provide the required cross-origin isolation headers.

For full local testing, use a static server that can send the headers in
`_headers`; a basic `python -m http.server` does not send them.

## Expected CSV columns

Required:

- `label`
- `score`
- `audio_recording_id`
- `start_offset_seconds`
- `end_offset_seconds`

Audio URLs are constructed as:

```text
https://api.ecosounds.org/audio_recordings/{audio_recording_id}/media.wav
  ?start_offset={start_offset_seconds}
  &end_offset={end_offset_seconds}
```

EcoSounds links are constructed as:

```text
https://www.ecosounds.org/listen/{audio_recording_id}/
  ?start={start_offset_seconds}
  &end={end_offset_seconds}
```

The explorer intentionally limits selected CSV files to 15 MB.
