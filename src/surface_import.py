"""Bounded, resumable AI Hub Surface CVAT XML preparation. No automatic review/split."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import random
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from .schema import init, sha, load_json, save_json, CLASSES, COLORS, active_class_ids
from .dataset import rows, write_rows, FIELDS

MAPPING = {'sidewalk': 2, 'roadway:normal': 1, 'alley:normal': 1, 'alley:speed_bump': 1, 'alley:damaged': 1, 'bike_lane': 8}
EXCLUDED = ['alley:crosswalk', 'braille_guide_blocks', 'caution_zone', 'roadway:crosswalk', 'unannotated']


def train_id(shape):
    label = shape.attrib.get('label', '')
    attrs = {a.attrib.get('name'): (a.text or '').strip() for a in shape.findall('attribute')}
    if label in ('roadway','alley'):
        return MAPPING.get(label+':'+attrs.get('attribute',''),255)
    return MAPPING.get(label, 255)


def rasterize(image_element, native_ids):
    width, height = int(image_element.attrib['width']), int(image_element.attrib['height'])
    if not (0 < width <= 20000 and 0 < height <= 20000):
        raise ValueError('Invalid annotated dimensions')
    training = Image.new('L', (width, height), 255)
    native = Image.new('L', (width, height), 255)
    td, nd = ImageDraw.Draw(training), ImageDraw.Draw(native)
    shapes = list(image_element)
    if any(s.tag not in ('polygon', 'tag') for s in shapes):
        raise ValueError('Unsupported annotation shape; refusing an incomplete mask')
    for shape in sorted((s for s in shapes if s.tag == 'polygon'), key=lambda s: int(s.attrib.get('z_order', '0'))):
        label = shape.attrib['label']
        if label not in native_ids:
            raise ValueError('Undeclared source label: ' + label)
        points = [tuple(map(float, pair.split(','))) for pair in shape.attrib['points'].split(';')]
        if len(points) < 3 or any(len(p) != 2 or not all(np.isfinite(p)) for p in points):
            raise ValueError('Invalid polygon points')
        # An ignored polygon must erase known classes below it as well.
        td.polygon(points, fill=train_id(shape))
        nd.polygon(points, fill=native_ids[label])
    return np.asarray(training), np.asarray(native)


def prepare_surface(root, archive_path, count=350, seed=42):
    root, archive_path = init(root), Path(archive_path).resolve()
    if count < 1:
        raise ValueError('count must be positive')
    if (root / 'data/splits/locked.json').exists():
        raise ValueError('Dataset is frozen; prepare a new dataset version')
    data = rows(root)
    state_path = root / 'data/surface_preparation.json'
    if data and not state_path.exists():
        raise ValueError('Existing unrelated dataset; prepare into a new root')
    exclusions = set(load_json(root / 'data/excluded_hashes.json')) if (root / 'data/excluded_hashes.json').exists() else set()
    with zipfile.ZipFile(archive_path) as archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        byname = {i.filename: i for i in infos}
        if len(byname) != len(infos):
            raise ValueError('Duplicate archive paths')
        for name in byname:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe archive path')
        fingerprint = hashlib.sha256(json.dumps([(i.filename, i.file_size, i.CRC) for i in infos]).encode()).hexdigest()
        config = dict(source_path=str(archive_path), central_directory_sha256=fingerprint,
                      count=count, seed=seed, mapping=MAPPING, excluded=EXCLUDED)
        if state_path.exists() and load_json(state_path)['config'] != config:
            raise ValueError('Preparation settings/source changed; choose a new dataset root')
        save_json(state_path, {'config': config, 'status': 'preparing', 'reviewed': False})
        labels, indexed, groups, xml_info = set(), {}, defaultdict(list), {}
        for name in sorted(n for n in byname if n.lower().endswith('.xml')):
            if byname[name].file_size > 16 * 2**20:
                raise ValueError('Unexpectedly large XML')
            payload = archive.read(name)
            element = ET.fromstring(payload)
            if element.tag != 'annotations' or element.find('track') is not None:
                raise ValueError('Expected CVAT image annotations, not video tracks')
            labels.update(e.text for e in element.findall('.//labels/label/name') if e.text)
            parent = PurePosixPath(name).parent
            xml_info[name] = {'bytes': payload, 'task_name': element.findtext('meta/task/name')}
            for image in element.findall('image'):
                photo = str(parent / PurePosixPath(image.attrib['name']).name)
                mask = str(parent / 'MASK' / (PurePosixPath(photo).stem + '.png'))
                if photo not in byname or mask not in byname:
                    continue
                if photo in indexed:
                    raise ValueError('Multiple XML records for one image')
                indexed[photo] = (image, mask, name)
                # Record the selection criterion; this is a supported-label subset, not a random test set.
                if any(s.tag == 'polygon' and train_id(s) != 255 for s in image):
                    groups[str(parent)].append(photo)
        native_ids = {label: i for i, label in enumerate(sorted(labels))}
        if len(native_ids) >= 255:
            raise ValueError('Too many source labels')
        rng = random.Random(seed)
        group_order = sorted(groups)
        rng.shuffle(group_order)
        for photos in groups.values():
            rng.shuffle(photos)
        candidate_order = []
        while any(groups.values()):
            for group in group_order:
                if groups[group]:
                    candidate_order.append(groups[group].pop())
        known_paths = {r['original_path'] for r in data}
        known_pixels = {r['pixel_sha'] for r in data}
        expected_source = 'AI Hub Surface XML (official, unreviewed)'
        if any(r['label_source'] != expected_source for r in data):
            raise ValueError('Existing rows have another label source; refusing to overwrite')
        # Resume checks every previously written image and mask, not only the row count.
        for row in data:
            if sha(root / row['image_path']) != row['image_sha'] or sha(root / row['mask_path']) != row['mask_sha']:
                raise ValueError('Previously prepared file changed')
        for folder in ['data/official_labels', 'data/source_rgb_masks', 'data/native_labels', 'data/source_xml', 'data/source_annotations', 'data/review/surface_overlays']:
            (root / folder).mkdir(parents=True, exist_ok=True)
        skipped = Counter()
        for photo in candidate_order:
            if len(data) >= count:
                break
            if photo in known_paths:
                continue
            element, mask_name, xml_name = indexed[photo]
            if any(shape.tag not in ('polygon', 'tag') for shape in element):
                skipped['unsupported_annotation_shape'] += 1
                continue
            train, native = rasterize(element, native_ids)
            if not (train != 255).any():
                skipped['all_ignore_after_layering'] += 1
                continue
            jpeg = archive.read(photo)
            with Image.open(io.BytesIO(jpeg)) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError('EXIF orientation requires explicit annotation alignment')
                image = image.convert('RGB'); image.load()
            if image.size != (train.shape[1], train.shape[0]):
                raise ValueError('Image/XML dimensions disagree')
            pixel = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
            if pixel in known_pixels or pixel in exclusions or hashlib.sha256(jpeg).hexdigest() in exclusions:
                skipped['duplicate_or_exploration'] += 1
                continue
            image_id = 'im_' + pixel[:20]
            image_path = root / 'data/raw' / (image_id + '.jpg')
            train_path = root / 'data/official_labels' / (image_id + '.png')
            image_path.write_bytes(jpeg)
            Image.fromarray(train).save(train_path)
            Image.fromarray(native).save(root / 'data/native_labels' / (image_id + '.png'))
            (root / 'data/source_rgb_masks' / (image_id + '.png')).write_bytes(archive.read(mask_name))
            save_json(root / 'data/source_annotations' / (image_id + '.json'), {
                'source_image': photo, 'source_rgb_mask': mask_name, 'source_xml': xml_name,
                'task_name': xml_info[xml_name]['task_name'], 'image_attributes': element.attrib,
                'shapes': [{'type': shape.tag, 'attributes': shape.attrib,
                            'properties': {a.attrib.get('name'): a.text for a in shape.findall('attribute')}}
                           for shape in element], 'reviewed': False})
            xml_dest = root / 'data/source_xml' / xml_name
            if not xml_dest.exists():
                xml_dest.parent.mkdir(parents=True, exist_ok=True)
                xml_dest.write_bytes(xml_info[xml_name]['bytes'])
            colors = np.zeros((*train.shape, 3), np.uint8)
            colors[:] = [80, 80, 80]
            valid = train != 255
            colors[valid] = COLORS[train[valid]]
            overlay = Image.blend(image, Image.fromarray(colors), .72)
            overlay.thumbnail((960, 540))
            overlay.save(root / 'data/review/surface_overlays' / (image_id + '.jpg'), quality=88)
            small = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
            bits = (small[:, 1:] > small[:, :-1]).flatten()
            values, counts = np.unique(train[valid], return_counts=True)
            row = dict.fromkeys(FIELDS, '')
            row.update(image_id=image_id, source='AI Hub sidewalk dataset 189 / Surface',
                       country='korea', camera_view='unknown', image_width=image.width, image_height=image.height,
                       label_source=expected_source, label_verified='false', status='needs_review',
                       image_path=str(image_path.relative_to(root)), image_sha=sha(image_path), pixel_sha=pixel,
                       mask_path=str(train_path.relative_to(root)), mask_sha=sha(train_path),
                       dhash=f'{sum(int(x)<<i for i,x in enumerate(bits)):016x}',
                       dominant_class=CLASSES[int(values[counts.argmax()])],
                       original_path=photo, group_hint=str(PurePosixPath(photo).parent),
                       direction_verified='false', license='AI Hub dataset terms; user must confirm')
            data.append(row); known_paths.add(photo); known_pixels.add(pixel)
            # Persist each sample for safe resume if the Colab runtime is interrupted.
            write_rows(root, data)
            if len(data) % 10 == 0:
                print(f'Surface preparation: {len(data)}/{count}', flush=True)
        pixel_counts = Counter()
        for row in data:
            values, counts = np.unique(np.asarray(Image.open(root / row['mask_path'])), return_counts=True)
            pixel_counts.update({str(int(v)): int(c) for v, c in zip(values, counts)})
        summary = dict(pixel_counts=dict(pixel_counts), config=config, status='prepared' if len(data) == count else 'insufficient_eligible_images',
                       prepared_count=len(data), native_ids=native_ids, source_labels=sorted(labels),
                       source_image_count=len(indexed), eligible_candidate_count=len(candidate_order), skipped=dict(skipped),
                       review_complete=False, group_confirmation=False, split_created=False,
                       selection_policy='seeded folder round-robin among images with supported XML polygons; all-ignore/duplicates excluded',
                       supported_train_ids=sorted(set(MAPPING.values())),
                       notes='Unannotated=255, never Other; crosswalk attribute not equated with crosswalk markings. Native/RGB/XML preserved.')
        save_json(state_path, summary)
        save_json(root / 'data/surface_class_mapping.json', dict(mapping=MAPPING, excluded=EXCLUDED, native_ids=native_ids,
                                                              background=255, reviewed=False, classes=CLASSES))
        with (root / 'data/review/surface_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'original_path', 'group_hint', 'mask_path', 'status', 'reviewer', 'notes'])
            writer.writeheader()
            for row in data:
                writer.writerow({k: row.get(k, '') for k in writer.fieldnames})
        return summary


def refresh_review_overlays(root):
    """Refresh visualization only; annotation IDs, reviews, split and checkpoints stay intact."""
    root=Path(root);data=rows(root)
    folder=root/'data/review/surface_overlays';folder.mkdir(parents=True,exist_ok=True)
    road_examples=[];road_images=0;road_pixels=0;total_pixels=0
    for index,row in enumerate(data):
        with Image.open(root/row['image_path']) as source:image=source.convert('RGB')
        mask=np.asarray(Image.open(root/row['mask_path']))
        if mask.shape!=(image.height,image.width):raise ValueError('Image/mask size mismatch')
        colors=np.full((*mask.shape,3),80,np.uint8);valid=mask!=255
        colors[valid]=COLORS[mask[valid]]
        overlay=Image.blend(image,Image.fromarray(colors),.72)
        overlay.thumbnail((960,540))
        # A persistent legend on every review image prevents confusing source RGB colors.
        canvas=Image.new('RGB',(overlay.width,overlay.height+32),'white');canvas.paste(overlay,(0,32))
        draw=ImageDraw.Draw(canvas)
        for x,(cid,label) in enumerate([(1,'ROAD'),(2,'SIDEWALK'),(8,'BIKE LANE')]):
            pos=x*max(150,overlay.width//4);color=tuple(int(c) for c in COLORS[cid])
            draw.rectangle((pos+6,8,pos+22,24),fill=color);draw.text((pos+28,10),label,fill='black')
        road_count=int((mask==1).sum());fraction=road_count/mask.size
        draw.text((max(0,overlay.width-200),10),f'Road: {fraction:.1%}',fill='black')
        canvas.save(folder/(row['image_id']+'.jpg'),quality=93)
        road_pixels+=road_count;total_pixels+=mask.size
        if road_count:
            road_images+=1;road_examples.append((fraction,row))
        if (index+1)%50==0:print('Overlays refreshed:',index+1,flush=True)
    # Include images with visible road labels; never infer model accuracy from these annotations.
    examples=[];groups=set()
    for fraction,row in sorted(road_examples,key=lambda value:-value[0]):
        if row['group_hint'] in groups:continue
        examples.append((fraction,row));groups.add(row['group_hint'])
        if len(examples)==6:break
    preview=Image.new('RGB',(1152,max(1,len(examples))*270+30),'white');draw=ImageDraw.Draw(preview)
    for x,title in enumerate(['Original image','Source RGB mask','Converted labels: orange = ROAD']):draw.text((x*384+6,8),title,fill='black')
    for y,(fraction,row) in enumerate(examples):
        paths=[root/row['image_path'],root/'data/source_rgb_masks'/(row['image_id']+'.png'),folder/(row['image_id']+'.jpg')]
        for x,path in enumerate(paths):
            with Image.open(path) as source:
                thumb=source.convert('RGB');thumb.thumbnail((384,240));preview.paste(thumb,(x*384,30+y*270))
        draw.text((8,30+y*270+242),f"{row['original_path']} | road label {fraction:.1%}",fill='black')
    dest=root/'reports/surface_road_preview.jpg';dest.parent.mkdir(parents=True,exist_ok=True);preview.save(dest,quality=92)
    # Main review preview uses the same new palette.
    preview.save(root/'reports/surface_preparation_preview.jpg',quality=92)
    result={'images':len(data),'road_label_images':road_images,'road_pixels':road_pixels,
            'total_pixels':total_pixels,'road_color':COLORS[1].tolist(),'overlay_alpha':.72,
            'preview_path':str(dest),'data_kind':'official XML labels, not model predictions',
            'road_example_paths':[row['original_path'] for _,row in examples]}
    save_json(root/'reports/surface_visualization.json',result)
    return result


def include_alley_as_road(root,confirmed=False):
    """Explicit, audited migration of unreviewed XML labels before split/training."""
    if not confirmed:raise ValueError('Explicit confirmation of the broader Road definition is required')
    root=Path(root)
    if active_class_ids(root)!=[1,2,8]:raise ValueError('Alley merge is only defined for the current three-class task')
    if (root/'data/splits/locked.json').exists() or any((root/'checkpoints').glob('*/run.json')):
        raise ValueError('Create a new dataset version after splitting or training')
    data=rows(root)
    if any(r['label_verified']=='true' or r['label_source']!='AI Hub Surface XML (official, unreviewed)' for r in data):
        raise ValueError('Migration is limited to original, unreviewed Surface XML masks')
    map_path=root/'data/surface_class_mapping.json';old=load_json(map_path)
    native_ids=old['native_ids']
    audit_path=root/'reports/alley_road_migration.json'
    audit=load_json(audit_path) if audit_path.exists() else {'previous_mapping':old,'status':'in_progress','changes':{},'user_confirmed':True}
    save_json(audit_path,audit)
    changed=0
    for index,row in enumerate(data):
        image_id=row['image_id'];mask_path=root/row['mask_path']
        if sha(mask_path)!=row['mask_sha']:raise ValueError('Source mask changed outside migration: '+image_id)
        payload=load_json(root/'data/source_annotations'/(image_id+'.json'))
        element=ET.Element('image',payload['image_attributes'])
        for source_shape in payload['shapes']:
            shape=ET.SubElement(element,source_shape['type'],source_shape['attributes'])
            for name,value in source_shape['properties'].items():ET.SubElement(shape,'attribute',{'name':name}).text=value
        mask,_=rasterize(element,native_ids)
        before=np.asarray(Image.open(mask_path));pixels=int((before!=mask).sum())
        if pixels:
            previous_sha=row['mask_sha']
            temp=mask_path.with_suffix('.tmp');Image.fromarray(mask).save(temp,format='PNG');temp.replace(mask_path)
            row['mask_sha']=sha(mask_path)
            values,counts=np.unique(mask[mask!=255],return_counts=True)
            row['dominant_class']=CLASSES[int(values[counts.argmax()])]
            row['status']='needs_review';row['label_verified']='false';row['reviewer']='';row['reviewed_at']=''
            audit['changes'][image_id]={'before_sha':previous_sha,'after_sha':row['mask_sha'],'changed_pixels':pixels}
            write_rows(root,data);save_json(audit_path,audit);changed+=1
        if (index+1)%50==0:print('Alley-to-Road labels updated:',index+1,flush=True)
    summary=load_json(root/'data/surface_preparation.json')
    summary['config']['mapping']=MAPPING;summary['config']['excluded']=EXCLUDED
    pixel_counts=Counter()
    for row in data:
        values,counts=np.unique(np.asarray(Image.open(root/row['mask_path'])),return_counts=True)
        pixel_counts.update({str(int(v)):int(c) for v,c in zip(values,counts)})
    summary['pixel_counts']=dict(pixel_counts)
    summary['road_definition']='General roadway plus alley normal/speed_bump/damaged; crosswalk and caution zones excluded'
    save_json(root/'data/surface_preparation.json',summary)
    save_json(map_path,{'mapping':MAPPING,'excluded':EXCLUDED,'native_ids':native_ids,'background':255,'reviewed':False,'classes':CLASSES,'road_definition':summary['road_definition']})
    audit.update(status='completed',updated_images=len(audit['changes']),last_run_changed_images=changed,reviewed=False,pixel_counts=dict(pixel_counts))
    save_json(audit_path,audit)
    return audit
