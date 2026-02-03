#!/usr/bin/env python3
import gi
import os
import csv
import time
import argparse

gi.require_version("Gst", "1.0")
from gi.repository import Gst

REFS_DIRECTORY = "./refs"
TESTS_DIRECTORY = "./tests"
EXPORT_DIRECTORY = "./export"


def create_branch(filename, pipeline):
    """
    Create a branch for WAV
    to convert to 48kHz mono audio
    """
    src = Gst.ElementFactory.make("filesrc")
    src.set_property("location", filename)

    dec = Gst.ElementFactory.make("decodebin")
    conv = Gst.ElementFactory.make("audioconvert")
    res = Gst.ElementFactory.make("audioresample")
    caps = Gst.ElementFactory.make("capsfilter")
    caps.set_property(
        "caps", Gst.Caps.from_string("audio/x-raw,format=F32LE,rate=48000,channels=1")
    )

    for e in (src, dec, conv, res, caps):
        pipeline.add(e)

    src.link(dec)
    conv.link(res)
    res.link(caps)

    def on_pad_added(decoder, pad):
        sink_pad = conv.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    dec.connect("pad-added", on_pad_added)

    return caps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="optional file name")
    file_name = parser.parse_args().file_name

    Gst.init(None)

    refs_files = set(os.listdir(REFS_DIRECTORY))
    tests_files = set(os.listdir(TESTS_DIRECTORY))
    intersection = refs_files.intersection(tests_files)
    wav_files = {f for f in intersection if f.lower().endswith(".wav")}

    odgs = ["odg"]  # Objective Difference Grade
    dis = ["di"]  # Distortion Index

    print(wav_files)
    for file in wav_files:
        pipeline = Gst.Pipeline.new(None)

        ref_filepath = os.path.join(REFS_DIRECTORY, file)
        test_filepath = os.path.join(TESTS_DIRECTORY, file)

        ref_caps = create_branch(ref_filepath, pipeline)
        test_caps = create_branch(test_filepath, pipeline)

        peaq = Gst.ElementFactory.make("peaq", "peaq")
        peaq.set_property("console-output", False)
        pipeline.add(peaq)

        ref_caps.get_static_pad("src").link(peaq.get_static_pad("ref"))
        test_caps.get_static_pad("src").link(peaq.get_static_pad("test"))

        pipeline.set_state(Gst.State.PLAYING)

        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )

        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print("Error:", err, debug)

        pipeline.set_state(Gst.State.NULL)

        odgs.append(peaq.get_property("odg"))
        dis.append(peaq.get_property("di"))

    # write to csv
    filenames = list(wav_files)
    filenames.insert(0, "")
    with open(
        os.path.join(
            EXPORT_DIRECTORY,
            file_name
            if file_name
            else f"peaq-export-{time.strftime('%Y%m%d-%H%M%S')}.csv",
        ),
        "a",
    ) as export:
        wr = csv.writer(export, quoting=csv.QUOTE_ALL)
        if not file_name:
            wr.writerow(filenames)
        wr.writerows([odgs, dis])
