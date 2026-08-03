"""Merge LightFC search/fusion/heads into one two-input NCNN graph.

The weight binary is concatenated by the ncnnmerge tool. This script then
rewires the prefixed subgraphs in the merged param so that intermediate blobs
do not remain external inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


STEMS = ("lightfc_search", "lightfc_fusion", "lightfc_score", "lightfc_size", "lightfc_offset")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=root / "model")
    parser.add_argument("--ncnnmerge", type=Path, default=root / "ncnn" / "x64" / "bin" / "ncnnmerge.exe")
    parser.add_argument("--output-stem", type=Path, default=root / "model" / "lightfc_tracking.ncnn")
    return parser.parse_args()


def graph_counts(lines: list[str]) -> tuple[int, int]:
    blobs: set[str] = set()
    for line in lines:
        fields = line.split()
        bottom_count = int(fields[2])
        top_count = int(fields[3])
        blobs.update(fields[4 : 4 + bottom_count + top_count])
    return len(lines), len(blobs)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_stem = args.output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        for suffix in (".ncnn.param", ".ncnn.bin"):
            path = model_dir / f"{stem}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix="lightfc_merge_", dir=output_stem.parent) as temporary:
        temporary = Path(temporary)
        raw_param = temporary / "raw.param"
        raw_bin = temporary / "raw.bin"
        command = [str(args.ncnnmerge)]
        for stem in STEMS:
            command += [f"{stem}.ncnn.param", f"{stem}.ncnn.bin"]
        command += [str(raw_param), str(raw_bin)]
        subprocess.run(command, cwd=model_dir, check=True)

        source_lines = raw_param.read_text(encoding="utf-8").splitlines()
        if source_lines[0].strip() != "7767517":
            raise RuntimeError("unexpected NCNN param magic")
        layers = source_lines[2:]
        remove_inputs = {
            "lightfc_fusion.ncnn/in1",
            "lightfc_score.ncnn/in0",
            "lightfc_size.ncnn/in0",
            "lightfc_offset.ncnn/in0",
        }
        rewired: list[str] = []
        replacements = {
            "lightfc_search.ncnn/in0": "search",
            "lightfc_search.ncnn/out0": "search_features",
            "lightfc_fusion.ncnn/in0": "template_features",
            "lightfc_fusion.ncnn/in1": "search_features",
            "lightfc_fusion.ncnn/out0": "fused_features",
            "lightfc_score.ncnn/in0": "score_features",
            "lightfc_size.ncnn/in0": "size_features",
            "lightfc_offset.ncnn/in0": "offset_features",
            "lightfc_score.ncnn/out0": "score_map",
            "lightfc_size.ncnn/out0": "size_map",
            "lightfc_offset.ncnn/out0": "offset_map",
        }
        for line in layers:
            fields = line.split()
            if fields[0] == "Input" and fields[1] in remove_inputs:
                continue
            for old, new in replacements.items():
                line = line.replace(old, new)
            rewired.append(line)
            if fields[1] == "lightfc_fusion.ncnn/cat_0":
                rewired.append(
                    "Split lightfc_tracking/split_heads 1 3 "
                    "fused_features score_features size_features offset_features"
                )

        layer_count, blob_count = graph_counts(rewired)
        output_param = Path(str(output_stem) + ".param")
        output_bin = Path(str(output_stem) + ".bin")
        output_param.write_text(
            "7767517\n" + f"{layer_count} {blob_count}\n" + "\n".join(rewired) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(raw_bin, output_bin)

    print(f"generated {output_stem}.param")
    print(f"generated {output_stem}.bin")


if __name__ == "__main__":
    main()
