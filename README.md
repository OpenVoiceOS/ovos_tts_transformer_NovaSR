# OVOS NovaSR TTS Transformer

Audio super-resolution for OpenVoiceOS speech synthesis. `NovaSR` upsamples the
audio produced by any TTS plugin from 16 kHz to 48 kHz just before playback,
recovering high-frequency detail that low-sample-rate voices discard. The result
is brighter, less muffled speech without retraining or replacing your existing
voice.

This is a [TTS transformer](https://openvoiceos.github.io/ovos-technical-manual/tts_transformers/)
plugin (`opm.transformer.tts`): it runs after the TTS stage and before playback,
operating on the generated waveform rather than on text.

## How it works

- The [NovaSR](https://github.com/ysharma3501/NovaSR) `FastSR` upsampler loads
  the generated `.wav`.
- `FastSR` reconstructs a 48 kHz waveform from the low-resolution input.
- The upsampled audio is written alongside the original and handed back for
  playback.
- If the incoming audio is already 48 kHz, the transform is skipped and the file
  is returned untouched.

The model weights are fetched from the Hugging Face Hub on first use and cached
locally. No manual download step is required.

`NovaSR` runs on PyTorch and uses the GPU when one is available. On CPU-only
systems, enabling half precision (`half=True`) yields a 3–4× speedup.

> **Note:** this transformer carries a Torch runtime. For a lighter,
> dependency-free alternative covering the same role, see
> [`ovos-tts-transformer-FlashSR`](https://github.com/OpenVoiceOS/ovos_tts_transformer_FlashSR),
> which runs on `onnxruntime`.

## Installation

```bash
pip install ovos-tts-transformer-NovaSR
```

## Configuration

Enable the transformer in `mycroft.conf` under the `tts_transformers` section,
keyed by the plugin name:

```json
"tts_transformers": {
  "ovos-tts-transformer-NovaSR": {}
}
```

The plugin takes no configuration of its own; once enabled it applies to the
output of whichever TTS plugin is active. Multiple TTS transformers can be
chained. Execution order follows each plugin's `priority` (NovaSR defaults to
`50`).

## Requirements

- [`NovaSR`](https://github.com/ysharma3501/NovaSR) (PyTorch)
- `huggingface-hub`
- `einops`

## Related

- [`ovos-tts-transformer-FlashSR`](https://github.com/OpenVoiceOS/ovos_tts_transformer_FlashSR):
  an ONNX-runtime super-resolution transformer covering the same role without a
  Torch dependency.
- [`ovos-tts-transformer-sox-plugin`](https://github.com/OpenVoiceOS/ovos-tts-transformer-sox-plugin):
  general-purpose audio effects (pitch, reverb, EQ, ...) for TTS output.

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
