from typing import Dict, List
import numpy as np

# use for index 2 rgb
floorplan_room_map = {
	0: [  203,  203,  203], # background
	1: [255, 255, 255], # Free
	2: [224,224,224], # not used
	3: [224,224,128]  # not used
}

# boundary label
floorplan_boundary_map = {
	0: [ 203, 203, 203], # background
	1: [255,60,128],  # opening (door&window)
	2: [0,0,0]  # wall line
}

# boundary label for presentation
floorplan_boundary_map_figure = {
	0: [255,255,255], # background
	1: [255, 60,128],  # opening (door&window)
	2: [  0,  0,  0]  # wall line
}

# merge all label into one multi-class label
floorplan_fuse_map = {
	0: [  0,  0,  0], # background
	1: [203, 203, 203], # Free
	2: [224,224,224], # not used
	3: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [255,255,255]  # extra label for wall line
}

# invert the color of wall line and background for presentation
floorplan_fuse_map_figure = {
	0: [255,255,255], # background
	1: [203, 203, 203], # Free
	2: [224,224,224], # not used
	3: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [ 0, 0,  0]  # extra label for wall line
}

def rgb2ind(
    im: np.ndarray, color_map: Dict[int, List[int]] = floorplan_room_map
) -> np.ndarray:
    ind = np.zeros((im.shape[0], im.shape[1]))

    for i, rgb in color_map.items():
        ind[(im == rgb).all(2)] = i

    return ind.astype(np.uint8)  # force to uint8

def ind2rgb(
    ind_im: np.ndarray, color_map: Dict[int, List[int]] = floorplan_room_map
) -> np.ndarray:
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb
    return rgb_im.astype(int)
