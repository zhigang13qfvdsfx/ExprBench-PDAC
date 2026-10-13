"""Verify every file in the lightweight package SHA256 inventory."""
from pathlib import Path
import argparse, csv, hashlib
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    with (root / 'file_manifest_sha256.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = (root / row['path']).resolve()
        assert path.is_relative_to(root)
        data = path.read_bytes()
        assert len(data) == int(row['bytes']) and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    print(f'Verified all {len(rows)} listed file hashes.')
if __name__ == '__main__':
    main()
