"""Assemble this paper's local reproducibility package without changing source data."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper/reproducibility_package"
ARCHIVE = ROOT / "paper/reproducibility_package.zip"


def main():
    if DEST.exists() or ARCHIVE.exists():
        raise FileExistsError("Preserve an existing package before preparing another one")
    selected = {ROOT / "README.md", ROOT / "requirements.txt"}
    for folder in ("common_code", "duffing_code", "sine_gordon_code", "double_sine_gordon_code", "legacy"):
        selected.update(p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".md"))
    excluded = []
    for folder in ("common_data", "duffing_data", "sine_gordon_data", "double_sine_gordon_data"):
        for p in (ROOT / folder).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.suffix in (".json", ".csv", ".log"):
                selected.add(p)
            elif p.suffix in (".npz", ".npy"):
                if folder != "double_sine_gordon_data" or p.stat().st_size <= 1024**2 or p.relative_to(ROOT).as_posix() == "double_sine_gordon_data/sg128/reference/E4_reference_lambda0.3_N128.npz":
                    selected.add(p)
                else:
                    excluded.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                         reason="intermediate trajectory not required for figures or Gaussian comparison"))
    for name in ("main_reog_marked_tocheck.tex", "main_reog_marked_tocheck.pdf", "main_reog_clean.tex", "main_reog_clean.pdf",
                 "ref.bib", "arxiv_abstract.txt", "reproducibility.md"):
        selected.add(ROOT / "paper" / name)
    selected.update(p for p in (ROOT / "paper/figures").iterdir() if p.suffix in (".tex", ".pdf", ".png"))
    selected.update(p for p in ROOT.glob("LICENSE*") if p.is_file())
    records = []
    for source in sorted(selected):
        target = DEST / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(dict(path=str(source.relative_to(ROOT)), bytes=source.stat().st_size,
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (DEST / "release_manifest.json").write_text(json.dumps(dict(files=records, exclusions=excluded), indent=2) + "\n")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                z.write(path, arcname=path.relative_to(DEST))
    print(f"{len(records)} files, {ARCHIVE.stat().st_size / 1024**2:.1f} MiB: {ARCHIVE}")


if __name__ == "__main__":
    main()
