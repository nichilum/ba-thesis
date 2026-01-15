from typing import Any
from pedalboard import load_plugin
from pedalboard.io import AudioFile
import random
import time


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
valhalla_dict = {
    "mix": range(0, 100),
    "delay_ms": range(0, 2000),
    "delaywarp": range(0, 100),
    "feedback": range(0, 100),
    "density": range(0, 100),
    "width": range(-100, 100),
    "lowcut": range(10, 2000, 10),
    "highcut": range(200, 20000, 15),
    "modrate": [x / 100.0 for x in range(1, 1000, 2)],
    "moddepth": range(0, 100),
    "mode": [
        "Gemini",
        "Hydra",
        "Centaurus",
        "Sagittarius",
        "Great Annihilator",
        "Andromeda",
        "Large Magellanic Cloud",
        "Triangulum",
        "Lyra",
        "Capricorn",
        "Cirrus Major",
        "Cirrus Minor",
        "Cassiopeia",
        "Orion",
        "Aquarius",
        "Pisces",
        "Scorpio",
        "Libra",
        "Leo",
        "Virgo",
        "Pleiades",
        "Sirius",
    ],
}

random.seed(42)
ts = time.perf_counter()
set_effect_attrib(random.choice, effect, valhalla_dict)
print(effect.parameters)
offline_reverberate(
    effect, "test-data/-0atYHAfGHA ('Male singing',).wav", "test-data/out.wav"
)
print(time.perf_counter() - ts)


random.seed(43)
ts = time.perf_counter()
set_effect_attrib(random.choice, effect, valhalla_dict)
print(effect.parameters)
offline_reverberate(
    effect, "test-data/-0atYHAfGHA ('Male singing',).wav", "test-data/out.wav"
)
print(time.perf_counter() - ts)
