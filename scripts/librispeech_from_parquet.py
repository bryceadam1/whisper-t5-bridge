"""Fetch LibriSpeech shards from the Hugging Face mirror (openslr/librispeech_asr, CC BY 4.0) and write them
in LibriSpeech's own layout (<subset>/<speaker>/<chapter>/<id>.flac + .trans.txt), so
librispeech_prepare.py can read them. Used when openslr.org is slow.
"""

from __future__ import annotations

import argparse
import collections
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-split", default="clean/train.360")
    ap.add_argument("--subset-name", default="train-clean-360")
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--root", required=True, help="output LibriSpeech root")
    args = ap.parse_args()

    out = Path(args.root) / args.subset_name
    n = 0
    for shard in args.shards:
        path = hf_hub_download(REPO, f"{args.hf_split}/{shard:04d}.parquet", repo_type="dataset", revision=REVISION,
                               local_dir=Path(args.root) / "_download")
        table = pq.read_table(path, columns=["audio", "text", "speaker_id", "chapter_id", "id"]).to_pylist()
        trans = collections.defaultdict(list)
        for row in table:
            d = out / str(row["speaker_id"]) / str(row["chapter_id"])
            d.mkdir(parents=True, exist_ok=True)
            data = row["audio"]["bytes"]
            sf.info(io.BytesIO(data))  # must be readable audio
            (d / f"{row['id']}.flac").write_bytes(data)
            trans[d].append(f"{row['id']} {row['text']}")
            n += 1
        for d, lines in trans.items():
            name = f"{d.parent.name}-{d.name}.trans.txt"
            with open(d / name, "a") as fh:
                fh.write("\n".join(sorted(lines)) + "\n")
        Path(path).unlink()  # keep only the extracted copy
        print(f"shard {shard}: {len(table)} utterances ({n} total)", flush=True)


if __name__ == "__main__":
    main()
