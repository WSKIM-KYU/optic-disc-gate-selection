"""Fundus-aware crop R1m, fixed on Chaksu Train (development, 2026-11-13): largest bright component (gray > 15) after
binary opening (3 iterations); square box around its bounding box with a 1.02 margin; pixels outside the component blanked;
resized to 384 with bilinear interpolation."""
import numpy as np
from PIL import Image
from scipy import ndimage
from fe_external_check import crop
def fundus_component(img):
    a = np.asarray(img.convert("L"), dtype=np.float32); m = ndimage.binary_opening(a > 15, iterations=3); lab, n = ndimage.label(m)
    return lab == (int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1)
def box_from(comp):
    ys, xs = np.nonzero(comp); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * 1.02; return cx - half, cy - half, 2 * half
def r1m(img):
    comp = fundus_component(img); b = box_from(comp); arr = np.asarray(img).copy(); arr[~comp] = 0
    return np.asarray(crop(Image.fromarray(arr), b, Image.BILINEAR)), b
