"""Download the four original exploration images; verify exact file identity."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import shutil


def prepare_exploration(root, photos, bundled_dir=None):
    root = Path(root)
    failures = []
    for photo in photos:
        group = 'korea' if photo['country'] == 'korea' else 'other'
        target = root / 'inputs' / group / photo['name']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != photo['sha256']:
                raise ValueError(f'Existing image differs from the original: {target}')
            print(f'INPUT_VERIFIED {photo["name"]} (cached)', flush=True)
            continue
        bundled = Path(bundled_dir) / photo['name'] if bundled_dir is not None else None
        if bundled is not None and bundled.is_file():
            if hashlib.sha256(bundled.read_bytes()).hexdigest() != photo['sha256']:
                raise ValueError(f'Bundled image differs from the original: {bundled}')
            shutil.copyfile(bundled, target)
            print(f'INPUT_VERIFIED {photo["name"]} (repository copy)', flush=True)
            continue
        errors = []
        downloaded = False
        for attempt in range(2):
            for url in photo['urls']:
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        payload = response.read(20 * 1024 * 1024 + 1)
                    if hashlib.sha256(payload).hexdigest() != photo['sha256']:
                        raise ValueError('Downloaded bytes differ from the original image')
                    temporary = target.with_suffix(target.suffix + '.download')
                    temporary.write_bytes(payload)
                    temporary.replace(target)
                    print(f'INPUT_VERIFIED {photo["name"]} ({len(payload)} bytes)', flush=True)
                    downloaded = True
                    break
                except Exception as error:
                    errors.append(f'{url}: {type(error).__name__}: {error}')
                    print(f'DOWNLOAD_RETRY {photo["name"]}: {error}', flush=True)
            if downloaded:
                break
            if attempt == 0:
                time.sleep(1)
        if not downloaded:
            failures.append(f'{photo["name"]}\n' + '\n'.join(errors))
    if failures:
        raise RuntimeError('Exploration input download failed; inference was not started.\n' + '\n'.join(failures))
    (root / 'input_sources.json').write_text(json.dumps(photos, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'EXPLORATION_INPUTS_READY {len(photos)} images: {root / "inputs"}', flush=True)
    return root


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'exploration_photos.json'
    prepare_exploration(args.root, json.loads(manifest.read_text(encoding='utf8')), manifest.parent / 'assets/exploration')
