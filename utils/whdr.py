#!/usr/bin/env python2.7
#
# This is an implementation of the WHDR metric proposed in this paper:
#
#     Sean Bell, Kavita Bala, Noah Snavely. "Intrinsic Images in the Wild". ACM
#     Transactions on Graphics (SIGGRAPH 2014). http://intrinsic.cs.cornell.edu.
#
# Please cite the above paper if you find this code useful.  This code is
# released under the MIT license (http://opensource.org/licenses/MIT).
#


import sys
import json
import argparse
import numpy as np
from PIL import Image


def compute_whdr(reflectance, judgements, delta=0.10):
    """ Return the WHDR score for a reflectance image, evaluated against human
    judgements.  The return value is in the range 0.0 to 1.0, or None if there
    are no judgements for the image.  See section 3.5 of our paper for more
    details.

    :param reflectance: a numpy array containing the linear RGB
    reflectance image.

    :param judgements: a JSON object loaded from the Intrinsic Images in
    the Wild dataset.

    :param delta: the threshold where humans switch from saying "about the
    same" to "one point is darker."
    """

    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    weight_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
            int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
            int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    if weight_sum:
        return error_sum / weight_sum
    else:
        return None


def load_image(filename, is_srgb=True):
    """ Load an image that is either linear or sRGB-encoded. """

    if not filename:
        raise ValueError("Empty filename")
    image = np.asarray(Image.open(filename)).astype(np.float) / 255.0
    if is_srgb:
        return srgb_to_rgb(image)
    else:
        return image


def srgb_to_rgb(srgb):
    """ Convert an sRGB image to a linear RGB image """

    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate an intrinsic image decomposition using the WHDR metric presented in:\n'
            '    Sean Bell, Kavita Bala, Noah Snavely. "Intrinsic Images in the Wild".\n'
            '    ACM Transactions on Graphics (SIGGRAPH 2014).\n'
            '    http://intrinsic.cs.cornell.edu.\n'
            '\n'
            'The output is in the range 0.0 to 1.0.'
        )
    )

    parser.add_argument(
        'reflectance', metavar='<reflectance.png>',
        help='reflectance image to be evaluated')

    parser.add_argument(
        'judgements', metavar='<judgements.json>',
        help='human judgements JSON file')

    parser.add_argument(
        '-l', '--linear', action='store_true', required=False,
        help='assume the reflectance image is linear, otherwise assume sRGB')

    parser.add_argument(
        '-d', '--delta', metavar='<float>', type=float, required=False, default=0.10,
        help='delta threshold (default 0.10)')

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    reflectance = load_image(filename=args.reflectance, is_srgb=(not args.linear))
    judgements = json.load(open(args.judgements))

    whdr = compute_whdr(reflectance, judgements, args.delta)
    print(whdr)
