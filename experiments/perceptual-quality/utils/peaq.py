#!/usr/bin/env python3
import gi
import sys

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


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


def run_peaq(ref, test):
    pipeline = Gst.Pipeline.new(None)

    ref_caps = create_branch(ref, pipeline)
    test_caps = create_branch(test, pipeline)

    peaq = Gst.ElementFactory.make("peaq")
    peaq.set_property("console-output", False)
    pipeline.add(peaq)

    ref_caps.get_static_pad("src").link(peaq.get_static_pad("ref"))
    test_caps.get_static_pad("src").link(peaq.get_static_pad("test"))

    pipeline.set_state(Gst.State.PLAYING)

    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE,
        Gst.MessageType.EOS | Gst.MessageType.ERROR,
    )

    if msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError(err)

    pipeline.set_state(Gst.State.NULL)

    odg = peaq.get_property("odg")
    di = peaq.get_property("di")

    # Explicit deref (important for long runs)
    pipeline = None
    peaq = None
    bus = None

    return odg, di


def main():
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            break

        ref, test = line.split("\t")
        try:
            odg, di = run_peaq(ref, test)
            print(f"{odg}\t{di}", flush=True)
        except Exception as e:
            print("nan\tnan", flush=True)


if __name__ == "__main__":
    main()
