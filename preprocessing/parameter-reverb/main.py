from typing import Any
from pedalboard import load_plugin
from pedalboard.io import AudioFile
import random
import time
import constants


def live_reverberate(effect, chunk, samplerate):
    return effect(chunk, samplerate, reset=False)


def offline_reverberate(effect, audio_file_path, output_file_path):
    with AudioFile(audio_file_path) as f:
        with AudioFile(output_file_path, "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = effect(chunk, f.samplerate, reset=False)
                o.write(effected)


def set_effect_attrib(choice, effect, parameter_dict: dict[str, Any]):
    for key in parameter_dict.keys():
        setattr(effect, key, choice(parameter_dict[key]))


effect = load_plugin("vst3/ValhallaSupermassive.vst3")


random.seed(42)
ts = time.perf_counter()
set_effect_attrib(random.choice, effect, constants.VALLHALLA_DICT)
offline_reverberate(
    effect, "test-data/-0atYHAfGHA ('Male singing',).wav", "test-data/out.wav"
)
print(time.perf_counter() - ts)


random.seed(100)
ts = time.perf_counter()
set_effect_attrib(random.choice, effect, constants.VALLHALLA_DICT)
offline_reverberate(
    effect, "test-data/-0atYHAfGHA ('Male singing',).wav", "test-data/out.wav"
)
print(time.perf_counter() - ts)
