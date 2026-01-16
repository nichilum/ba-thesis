from typing import Any
from pedalboard import load_plugin
from pedalboard.io import AudioFile
import random
import constants
import os

from dotenv import load_dotenv

load_dotenv()

effect = load_plugin(os.getenv("VALHALLA_VST_PATH"))


def live_reverberate(chunk, samplerate):
    set_effect_attrib(random.choice, effect, constants.VALLHALLA_DICT)
    return effect(chunk, samplerate, reset=False)


def offline_reverberate(audio_file_path, output_file_path):
    with AudioFile(audio_file_path) as f:
        with AudioFile(output_file_path, "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = effect(chunk, f.samplerate, reset=False)
                o.write(effected)


def set_effect_attrib(choice, effect, parameter_dict: dict[str, Any]):
    for key in parameter_dict.keys():
        setattr(effect, key, choice(parameter_dict[key]))
