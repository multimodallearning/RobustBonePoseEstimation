from pathlib import Path
import xmltodict
from shutil import copy

src_path = Path('/home/ron/Documents/FractureAngle/dataset/data/img8bit')
dst_path = Path('/home/ron/Downloads/FractureAngle')
xml_files = list(Path('/home/ron/Documents/FractureAngle/dataset/data/cvat_annotations').glob('*.xml'))
for xml_file in xml_files:

    with open(xml_file, 'r') as file:
        current_dst = dst_path / xml_file.stem
        current_dst.mkdir(exist_ok=True)
        images = xmltodict.parse(file.read())['annotations']['image']
    for img in images:
        copy(src_path / img['@name'], current_dst / img['@name'])
