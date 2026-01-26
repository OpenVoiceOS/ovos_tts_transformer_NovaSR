# TODO - convert to onnx to drop dependency on torch
from typing import Tuple, Dict, Any

from NovaSR import FastSR
from ovos_plugin_manager.templates.transformers import TTSTransformer
from ovos_plugin_manager.templates.tts import TTS
from ovos_plugin_manager.utils.audio import AudioData, AudioFile


class NovaSRTTSTransformer(TTSTransformer):
    """ runs after TTS stage but before playback"""

    def __init__(self, name="ovos-tts-transformer-NovaSR", priority=50, config=None):
        super().__init__(name, priority, config)
        half = False  ## Use this instead for CPUs as it leads to 3-4x speedup.
        self.upsampler = FastSR(half=half)  ## downloads from hf

    def transform(self, wav_file: str, context: dict | None = None) -> Tuple[str, Dict[str, Any]]:
        """
        Optionally transform passed wav_file and return path to transformed file
        :param wav_file: path to wav file generated in TTS stage
        :returns: path to transformed wav file for playback
        """
        context = context or {}
        sr = context.get("sr", 16000)
        if sr == 48000:
            # skip - already hi-res
            return wav_file, context

        lowres_audio = self.upsampler.load_audio(wav_file)

        highres_audio = self.upsampler.infer(lowres_audio).cpu()

        # Save output audio at 48kHz
        outpath = wav_file.replace(".wav", "_sr.wav")
        data = AudioData.from_array(highres_audio.squeeze(0).numpy(), sample_rate=48000, sample_width=2)
        with open(outpath, "wb") as f:
            f.write(data.get_wav_data())

        return outpath, context


if __name__ == "__main__":
    tx = NovaSRTTSTransformer()
    tx.transform("test.wav")
    # listen to test_sr.wav and compare
