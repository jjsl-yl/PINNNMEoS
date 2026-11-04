"""Create a deterministic-content GitHub source archive and checksums."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and
                 not any(part in ('__pycache__','.git','.venv','outputs') for part in p.relative_to(ROOT).parts)
                 and p.name not in ('SHA256SUMS.json','.DS_Store') and p.suffix not in ('.pyc','.zip'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=ROOT/'SHA256SUMS.json'
    manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    target=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in [*files,manifest]: archive.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    print(json.dumps(dict(path=str(target),files=len(files)+1,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__': main()
