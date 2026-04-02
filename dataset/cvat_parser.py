import math
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Union

import torch
from xmltodict import parse


class LocationCode(Enum):
    PROXIMAL = 0
    DISTAL = 1
    METASQUARE = 2
    NA = -1


class BoneCode(Enum):
    ULNA = 0
    RADIUS = 1
    NA = -1


@dataclass
class Box:
    location: LocationCode
    bone: BoneCode
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]

    @cached_property
    def bbox(self) -> torch.Tensor:
        return torch.tensor([self.top_left, self.top_right, self.bottom_right, self.bottom_left])


class CVATBoxParser:
    def __init__(self, path2xml, filter_loc: Union[List[LocationCode], LocationCode] = None):
        """
        Parser to extract rotated bounding boxes from CVAT XML annotations
        :param path2xml: string or list of strings holding the path to the XML files
        :param filter_loc: which location code should be included. None=keep everything
        """
        super().__init__()
        img_dicts = list()

        if not isinstance(path2xml, list):
            path2xml = [path2xml]
        for xml in path2xml:
            with open(xml) as fd:
                img_dicts.extend(parse(fd.read())['annotations']['image'])
        if len(img_dicts) == 0:
            raise FileNotFoundError('Parser found no annotations.')
        self.boxes = dict()
        # extract boxes
        for img_dict in img_dicts:
            if 'box' in img_dict:
                if isinstance(img_dict['box'], list):
                    self.boxes[img_dict['@name'].split('.')[0]] = [self.create_box(b) for b in img_dict['box']]
                else:
                    self.boxes[img_dict['@name'].split('.')[0]] = [self.create_box(img_dict['box'])]

        # filter to wanted boxes
        if filter_loc is not None:
            if not isinstance(filter_loc, list):
                filter_loc = [filter_loc]
            for img, box_list in self.boxes.items():
                self.boxes[img] = list(filter(lambda b: b.location in filter_loc, box_list))
            # drop now empty entries
            self.boxes = {k: v for k, v in self.boxes.items() if len(v) > 0}

    def __call__(self, img_id: str):
        return self.boxes[img_id]

    @staticmethod
    def create_box(d: dict):
        rotation = float(d['@rotation']) if '@rotation' in d else 0.
        tl, tr, br, bl = CVATBoxParser.rotated_box_points(float(d['@xtl']), float(d['@ytl']), float(d['@xbr']),
                                                          float(d['@ybr']), rotation)
        match d['@label'].split(' ')[0].lower():
            case 'proximales':
                loc = LocationCode.PROXIMAL
            case 'distales':
                loc = LocationCode.DISTAL
            case 'metaphysäre':
                loc = LocationCode.METASQUARE
            case _:
                loc = LocationCode.NA

        if 'attribute' in d:
            bone_str = d['attribute']['#text'].upper()
            bone = BoneCode[bone_str]
        else:
            bone = BoneCode.NA

        b = Box(loc, bone, tl, tr, br, bl)
        return b

    @staticmethod
    def rotated_box_points(xtl, ytl, xbr, ybr, rotation_deg):
        cx = (xtl + xbr) / 2
        cy = (ytl + ybr) / 2
        w = xbr - xtl
        h = ybr - ytl
        angle = math.radians(rotation_deg)
        corners = [
            (-w / 2, -h / 2),  # top left
            (w / 2, -h / 2),  # top right
            (w / 2, h / 2),  # bottom right
            (-w / 2, h / 2),  # bottom left
        ]
        rotated_corners = [(cx + x * math.cos(angle) - y * math.sin(angle),
                            cy + x * math.sin(angle) + y * math.cos(angle)) for x, y in corners]

        return rotated_corners


if __name__ == '__main__':
    from pathlib import Path

    # test
    import matplotlib.pyplot as plt
    from random import sample

    img_path = Path('dataset/data/img8bit')

    parser = CVATBoxParser('dataset/data/cvat_annotations/annotations1.xml')
    img_ids = ['0003_0662359226_01_WRI-R1_M011.png', '0215_0914656950_03_WRI-L2_F003.png']
    img_ids = sample(list(parser.boxes.keys()), 5)
    c = {
        LocationCode.PROXIMAL: 'r',
        LocationCode.DISTAL: 'g',
        LocationCode.METASQUARE: 'b',
    }
    for img_id in img_ids:
        plt.figure(img_id)
        img = plt.imread(img_path.joinpath(img_id).with_suffix('.png'))
        plt.imshow(img, cmap='gray', origin='upper')  # origin='upper' matches CVAT
        for b in parser(img_id):
            bbox = b.bbox
            plt.plot(bbox[:, 0], bbox[:, 1], c=c[b.location])
            # coord = [b.top_left, b.top_right, b.bottom_right, b.bottom_left]
            # plt.scatter([c[0] for c in coord], [c[1] for c in coord], c='b')

    plt.show()
