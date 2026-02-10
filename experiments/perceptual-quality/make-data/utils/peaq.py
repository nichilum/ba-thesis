#!/usr/bin/env python3
import gi
import argparse
import sys

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def create_branch(filename, pipeline):
    src = Gst.ElementFactory.make("filesrc")
    src.set_property("location", filename)

    dec = Gst.ElementFactory.make("decodebin")
    conv = Gst.ElementFactory.make("audioconvert")
    res = Gst.ElementFactory.make("audioresample")
    caps = Gst.ElementFactory.make("capsfilter")
    caps.set_property(
        "caps",
        Gst.Caps.from_string("audio/x-raw,format=F32LE,rate=48000,channels=1"),
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


def main():
    parser = argparse.ArgumentParser(description="Compute PEAQ ODG and DI")
    parser.add_argument("--ref", required=True, help="reference WAV file")
    parser.add_argument("--test", required=True, help="test WAV file")
    args = parser.parse_args()

    Gst.init(None)
    pipeline = Gst.Pipeline.new(None)

    ref_caps = create_branch(args.ref, pipeline)
    test_caps = create_branch(args.test, pipeline)

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
        print(f"ERROR: {err}", file=sys.stderr)
        if debug:
            print(debug, file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    pipeline.set_state(Gst.State.NULL)

    odg = peaq.get_property("odg")
    di = peaq.get_property("di")

    # Subprocess-friendly output (easy to parse)
    print(f"ODG={odg}")
    print(f"DI={di}")


if __name__ == "__main__":
    main()
