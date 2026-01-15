import yt_dlp
import os
import csv

import pandas as pd

d = pd.read_csv(
    "class_labels_indices.csv",
    header=None,
    usecols=[1, 2],
    index_col=[0],
    sep=",",
)[2].to_dict()

dest = "output"
if not os.path.exists(dest):
    os.makedirs(dest)

with open("./unbalanced_train_segments.csv") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        URL = "https://www.youtube.com/watch?v=" + row[0]

        start = int(row[1])
        end = int(row[2])

        filename = f"{row[0]} {(*list(map(d.get, row[3].split(','))),)}"

        if os.path.isfile(os.path.join(dest, filename + ".wav")):
            continue

        ydl_opts = {
            "format": "wav/bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "external_downloader": "ffmpeg",
            "external_downloader_args": {
                "ffmpeg_i": ["-ss", str(start), "-to", str(end)]
            },
            "outtmpl": os.path.join(dest, filename),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([URL])
            except:  # noqa: E722
                pass
