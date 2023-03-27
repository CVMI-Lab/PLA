import numpy as np

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168],[75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])


# scannet
SCANNET_SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 40])

SCANNET_SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator',
                                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'unannotated'])

SCANNET_CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [82, 84, 163]
}

SCANNET_SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table',
                             8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture', 12: 'counter', 14: 'desk',
                             16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink',
                             36: 'bathtub', 39: 'otherfurniture', 40: 'unannotated'}

SCANNET_COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255


# s3dis
S3DIS_SEMANTIC_NAMES = np.array(["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair",
                                 "sofa", "bookshelf", "board", "clutter"])

S3DIS_SEMANTIC_IDX2NAME = {1: 'ceiling', 2: 'floor', 3: 'wall', 4: 'beam', 5: 'column', 6: 'window', 7: 'door',
                           8: 'table', 9: 'chair', 10: 'sofa', 11:'bookshelf', 12: 'board', 13: 'clutter'}

S3DIS_CLASS_COLOR = {
    'ceiling': [0, 255, 0],
    'floor': [0, 0, 255],
    'wall': [0, 255, 255],
    'beam': [255, 255, 0],
    'column': [255, 0, 255],
    'window': [100, 100, 255],
    'door': [200, 200, 100],
    'table': [170, 120, 200],
    'chair': [255, 0, 0],
    'sofa': [200, 100, 100],
    'bookshelf': [10, 200, 100],
    'board': [200, 200, 200],
    'clutter': [50, 50, 50],
    'ignore': [0, 0, 0]
}

### ScanNet200 Benchmark constants ###
VALID_CLASS_IDS_200 = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
    488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191])

CLASS_LABELS_200 = np.array([
    'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
    'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
    'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
    'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
    'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
    'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
    'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
    'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
    'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress'])

SCANNET_CLASS_COLOR_200 = {'wall': (174.0, 199.0, 232.0), 'chair': (188.0, 189.0, 34.0), 'floor': (152.0, 223.0, 138.0), 'table': (255.0, 152.0, 150.0), 'door': (214.0, 39.0, 40.0), 'couch': (91.0, 135.0, 229.0), 'cabinet': (31.0, 119.0, 180.0), 'shelf': (229.0, 91.0, 104.0), 'desk': (247.0, 182.0, 210.0), 'office chair': (91.0, 229.0, 110.0), 'bed': (255.0, 187.0, 120.0), 'pillow': (141.0, 91.0, 229.0), 'sink': (112.0, 128.0, 144.0), 'picture': (196.0, 156.0, 148.0), 'window': (197.0, 176.0, 213.0), 'toilet': (44.0, 160.0, 44.0), 'bookshelf': (148.0, 103.0, 189.0), 'monitor': (229.0, 91.0, 223.0), 'curtain': (219.0, 219.0, 141.0), 'book': (192.0, 229.0, 91.0), 'armchair': (88.0, 218.0, 137.0), 'coffee table': (58.0, 98.0, 137.0), 'box': (177.0, 82.0, 239.0), 'refrigerator': (255.0, 127.0, 14.0), 'lamp': (237.0, 204.0, 37.0), 'kitchen cabinet': (41.0, 206.0, 32.0), 'towel': (62.0, 143.0, 148.0), 'clothes': (34.0, 14.0, 130.0), 'tv': (143.0, 45.0, 115.0), 'nightstand': (137.0, 63.0, 14.0), 'counter': (23.0, 190.0, 207.0), 'dresser': (16.0, 212.0, 139.0), 'stool': (90.0, 119.0, 201.0), 'cushion': (125.0, 30.0, 141.0), 'plant': (150.0, 53.0, 56.0), 'ceiling': (186.0, 197.0, 62.0), 'bathtub': (227.0, 119.0, 194.0), 'end table': (38.0, 100.0, 128.0), 'dining table': (120.0, 31.0, 243.0), 'keyboard': (154.0, 59.0, 103.0), 'bag': (169.0, 137.0, 78.0), 'backpack': (143.0, 245.0, 111.0), 'toilet paper': (37.0, 230.0, 205.0), 'printer': (14.0, 16.0, 155.0), 'tv stand': (196.0, 51.0, 182.0), 'whiteboard': (237.0, 80.0, 38.0), 'blanket': (138.0, 175.0, 62.0), 'shower curtain': (158.0, 218.0, 229.0), 'trash can': (38.0, 96.0, 167.0), 'closet': (190.0, 77.0, 246.0), 'stairs': (208.0, 49.0, 84.0), 'microwave': (208.0, 193.0, 72.0), 'stove': (55.0, 220.0, 57.0), 'shoe': (10.0, 125.0, 140.0), 'computer tower': (76.0, 38.0, 202.0), 'bottle': (191.0, 28.0, 135.0), 'bin': (211.0, 120.0, 42.0), 'ottoman': (118.0, 174.0, 76.0), 'bench': (17.0, 242.0, 171.0), 'board': (20.0, 65.0, 247.0), 'washing machine': (208.0, 61.0, 222.0), 'mirror': (162.0, 62.0, 60.0), 'copier': (210.0, 235.0, 62.0), 'basket': (45.0, 152.0, 72.0), 'sofa chair': (35.0, 107.0, 149.0), 'file cabinet': (160.0, 89.0, 237.0), 'fan': (227.0, 56.0, 125.0), 'laptop': (169.0, 143.0, 81.0), 'shower': (42.0, 143.0, 20.0), 'paper': (25.0, 160.0, 151.0), 'person': (82.0, 75.0, 227.0), 'paper towel dispenser': (253.0, 59.0, 222.0), 'oven': (240.0, 130.0, 89.0), 'blinds': (123.0, 172.0, 47.0), 'rack': (71.0, 194.0, 133.0), 'plate': (24.0, 94.0, 205.0), 'blackboard': (134.0, 16.0, 179.0), 'piano': (159.0, 32.0, 52.0), 'suitcase': (213.0, 208.0, 88.0), 'rail': (64.0, 158.0, 70.0), 'radiator': (18.0, 163.0, 194.0), 'recycling bin': (65.0, 29.0, 153.0), 'container': (177.0, 10.0, 109.0), 'wardrobe': (152.0, 83.0, 7.0), 'soap dispenser': (83.0, 175.0, 30.0), 'telephone': (18.0, 199.0, 153.0), 'bucket': (61.0, 81.0, 208.0), 'clock': (213.0, 85.0, 216.0), 'stand': (170.0, 53.0, 42.0), 'light': (161.0, 192.0, 38.0), 'laundry basket': (23.0, 241.0, 91.0), 'pipe': (12.0, 103.0, 170.0), 'clothes dryer': (151.0, 41.0, 245.0), 'guitar': (133.0, 51.0, 80.0), 'toilet paper holder': (184.0, 162.0, 91.0), 'seat': (50.0, 138.0, 38.0), 'speaker': (31.0, 237.0, 236.0), 'column': (39.0, 19.0, 208.0), 'bicycle': (223.0, 27.0, 180.0), 'ladder': (254.0, 141.0, 85.0), 'bathroom stall': (97.0, 144.0, 39.0), 'shower wall': (106.0, 231.0, 176.0), 'cup': (12.0, 61.0, 162.0), 'jacket': (124.0, 66.0, 140.0), 'storage bin': (137.0, 66.0, 73.0), 'coffee maker': (250.0, 253.0, 26.0), 'dishwasher': (55.0, 191.0, 73.0), 'paper towel roll': (60.0, 126.0, 146.0), 'machine': (153.0, 108.0, 234.0), 'mat': (184.0, 58.0, 125.0), 'windowsill': (135.0, 84.0, 14.0), 'bar': (139.0, 248.0, 91.0), 'toaster': (53.0, 200.0, 172.0), 'bulletin board': (63.0, 69.0, 134.0), 'ironing board': (190.0, 75.0, 186.0), 'fireplace': (127.0, 63.0, 52.0), 'soap dish': (141.0, 182.0, 25.0), 'kitchen counter': (56.0, 144.0, 89.0), 'doorframe': (64.0, 160.0, 250.0), 'toilet paper dispenser': (182.0, 86.0, 245.0), 'mini fridge': (139.0, 18.0, 53.0), 'fire extinguisher': (134.0, 120.0, 54.0), 'ball': (49.0, 165.0, 42.0), 'hat': (51.0, 128.0, 133.0), 'shower curtain rod': (44.0, 21.0, 163.0), 'water cooler': (232.0, 93.0, 193.0), 'paper cutter': (176.0, 102.0, 54.0), 'tray': (116.0, 217.0, 17.0), 'shower door': (54.0, 209.0, 150.0), 'pillar': (60.0, 99.0, 204.0), 'ledge': (129.0, 43.0, 144.0), 'toaster oven': (252.0, 100.0, 106.0), 'mouse': (187.0, 196.0, 73.0), 'toilet seat cover dispenser': (13.0, 158.0, 40.0), 'furniture': (52.0, 122.0, 152.0), 'cart': (128.0, 76.0, 202.0), 'storage container': (187.0, 50.0, 115.0), 'scale': (180.0, 141.0, 71.0), 'tissue box': (77.0, 208.0, 35.0), 'light switch': (72.0, 183.0, 168.0), 'crate': (97.0, 99.0, 203.0), 'power outlet': (172.0, 22.0, 158.0), 'decoration': (155.0, 64.0, 40.0), 'sign': (118.0, 159.0, 30.0), 'projector': (69.0, 252.0, 148.0), 'closet door': (45.0, 103.0, 173.0), 'vacuum cleaner': (111.0, 38.0, 149.0), 'candle': (184.0, 9.0, 49.0), 'plunger': (188.0, 174.0, 67.0), 'stuffed animal': (53.0, 206.0, 53.0), 'headphones': (97.0, 235.0, 252.0), 'dish rack': (66.0, 32.0, 182.0), 'broom': (236.0, 114.0, 195.0), 'guitar case': (241.0, 154.0, 83.0), 'range hood': (133.0, 240.0, 52.0), 'dustpan': (16.0, 205.0, 144.0), 'hair dryer': (75.0, 101.0, 198.0), 'water bottle': (237.0, 95.0, 251.0), 'handicap bar': (191.0, 52.0, 49.0), 'purse': (227.0, 254.0, 54.0), 'vent': (49.0, 206.0, 87.0), 'shower floor': (48.0, 113.0, 150.0), 'water pitcher': (125.0, 73.0, 182.0), 'mailbox': (229.0, 32.0, 114.0), 'bowl': (158.0, 119.0, 28.0), 'paper bag': (60.0, 205.0, 27.0), 'alarm clock': (18.0, 215.0, 201.0), 'music stand': (79.0, 76.0, 153.0), 'projector screen': (134.0, 13.0, 116.0), 'divider': (192.0, 97.0, 63.0), 'laundry detergent': (108.0, 163.0, 18.0), 'bathroom counter': (95.0, 220.0, 156.0), 'object': (98.0, 141.0, 208.0), 'bathroom vanity': (144.0, 19.0, 193.0), 'closet wall': (166.0, 36.0, 57.0), 'laundry hamper': (212.0, 202.0, 34.0), 'bathroom stall door': (23.0, 206.0, 34.0), 'ceiling light': (91.0, 211.0, 236.0), 'trash bin': (79.0, 55.0, 137.0), 'dumbbell': (182.0, 19.0, 117.0), 'stair rail': (134.0, 76.0, 14.0), 'tube': (87.0, 185.0, 28.0), 'bathroom cabinet': (82.0, 224.0, 187.0), 'cd case': (92.0, 110.0, 214.0), 'closet rod': (168.0, 80.0, 171.0), 'coffee kettle': (197.0, 63.0, 51.0), 'structure': (175.0, 199.0, 77.0), 'shower head': (62.0, 180.0, 98.0), 'keyboard piano': (8.0, 91.0, 150.0), 'case of water bottles': (77.0, 15.0, 130.0), 'coat rack': (154.0, 65.0, 96.0), 'storage organizer': (197.0, 152.0, 11.0), 'folded chair': (59.0, 155.0, 45.0), 'fire alarm': (12.0, 147.0, 145.0), 'power strip': (54.0, 35.0, 219.0), 'calendar': (210.0, 73.0, 181.0), 'poster': (221.0, 124.0, 77.0), 'potted plant': (149.0, 214.0, 66.0), 'luggage': (72.0, 185.0, 134.0), 'mattress': (42.0, 94.0, 198.0)}

### For instance segmentation the non-object categories ###
VALID_PANOPTIC_IDS = (1, 3)

CLASS_LABELS_PANOPTIC = ('wall', 'floor')
