"""Select matching AI Hub Surface JPG/MASK PNG pairs without unpacking the archive.

Uses only Python's standard library. Folder groups are suggestions, not verified split IDs.
Original mask values and XML are preserved; no class mapping/review is inferred.
"""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import random
import shutil
import struct
import xml.etree.ElementTree as ET
import zipfile


def prepare(source, output, count=350, seed=42):
    source, output = Path(source), Path(output)
    if count < 1:
        raise ValueError('count must be positive')
    if output.exists():
        raise FileExistsError('Output already exists; choose a new output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        if len(infos) != sum(not i.is_dir() for i in archive.infolist()):
            raise ValueError('Archive has duplicate names')
        for name in infos:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe archive entry')
        groups = defaultdict(list)
        missing = 0
        for name in sorted(infos):
            path = PurePosixPath(name)
            if path.suffix.lower() not in ('.jpg', '.jpeg'):
                continue
            mask = str(path.parent / 'MASK' / (path.stem + '.png'))
            if mask not in infos:
                missing += 1
                continue
            groups[str(path.parent)].append((name, mask))
        if not groups:
            raise ValueError('No matching JPG and sibling MASK/*.png pairs')
        rng = random.Random(seed)
        order = sorted(groups)
        rng.shuffle(order)
        for pairs in groups.values():
            rng.shuffle(pairs)
        # Round-robin avoids taking 350 adjacent frames from the first folder.
        selected = []
        while len(selected) < count:
            changed = False
            for group in order:
                if groups[group] and len(selected) < count:
                    selected.append(groups[group].pop())
                    changed = True
            if not changed:
                break
        selected_groups = {str(PurePosixPath(image).parent) for image, _ in selected}
        xmls = [name for name in sorted(infos) if PurePosixPath(name).suffix.lower() == '.xml'
                and str(PurePosixPath(name).parent) in selected_groups]
        names = [name for pair in selected for name in pair] + xmls
        estimate = sum(infos[name].file_size for name in names)
        print(f'Selected {len(selected)} pairs from {len(selected_groups)} folder groups + {len(xmls)} XML files', flush=True)
        print(f'Uncompressed selected files: {estimate / 2**20:.1f} MiB', flush=True)
        if shutil.disk_usage(output.parent).free < estimate + 16 * 2**20:
            raise ValueError('Insufficient disk space for subset ZIP')
        report = {
            'source_archive_name': source.name, 'seed': seed, 'requested_pairs': count,
            'selected_pairs': len(selected), 'folder_groups': sorted(selected_groups),
            'folder_groups_confirmed': False, 'unmatched_jpg_count': missing,
            'source_extensions': dict(Counter(PurePosixPath(n).suffix.lower() for n in infos)),
            'label_status': 'official_unreviewed', 'class_mapping': 'not_verified',
            'pairs': [{'image': image, 'mask': mask, 'group_hint': str(PurePosixPath(image).parent)}
                      for image, mask in selected],
        }
        # Inspect the file header only. Pixel semantics cannot be established from a filename.
        with archive.open(selected[0][1]) as mask:
            header = mask.read(33)
        if header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR':
            width, height, depth, color_type, *_ = struct.unpack('>IIBBBBB', header[16:29])
            report['sample_png_header'] = dict(width=width, height=height, bit_depth=depth, color_type=color_type)
        if xmls and infos[xmls[0]].file_size < 4 * 2**20:
            try:
                root = ET.fromstring(archive.read(xmls[0]))
                report['sample_xml'] = {
                    'path': xmls[0], 'root_tag': root.tag,
                    'root_children': list(dict.fromkeys(child.tag for child in root)),
                    'label_names': [node.text for node in root.findall('.//labels/label/name')],
                }
            except ET.ParseError:
                report['sample_xml'] = {'path': xmls[0], 'parse_status': 'unrecognized_xml'}
        pending = output.with_name(output.name + '.partial')
        try:
            with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as target:
                hashes = {}
                for name in names:
                    digest = hashlib.sha256()
                    with archive.open(name) as src, target.open(name, 'w', force_zip64=True) as dst:
                        while chunk := src.read(1024 * 1024):
                            dst.write(chunk)
                            digest.update(chunk)
                    hashes[name] = digest.hexdigest()
                report['sha256'] = hashes
                target.writestr('subset_manifest.json', json.dumps(report, ensure_ascii=False, indent=2))
            pending.rename(output)
        except BaseException:
            # Keep partial output for diagnosis; never report it as complete.
            raise
    print(f'Created: {output.resolve()} ({output.stat().st_size / 2**20:.1f} MiB)', flush=True)
    print(json.dumps({key: report.get(key) for key in ['sample_png_header', 'sample_xml', 'unmatched_jpg_count']}, ensure_ascii=False, indent=2))
    print('Keep images, MASK and XML together; verify class mapping and real sequence/location groups before training.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=350)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.count, args.seed)
