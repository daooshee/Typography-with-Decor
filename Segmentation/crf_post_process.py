# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import warnings
import cv2

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

parser = argparse.ArgumentParser()
parser.add_argument('--img')

opt = parser.parse_args()

def dense_crf(img, output_probs):
    H = output_probs.shape[0]
    W = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    U = unary_from_softmax(output_probs)  # note: num classes is first dim
    d = dcrf.DenseCRF2D(H, W, 2)
    d.setUnaryEnergy(U)

    Q_unary = d.inference(5)

    # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
    map_soln_unary = np.argmax(Q_unary, axis=0)

    # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
    map_soln_unary = map_soln_unary.reshape((H,W))

    pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img.reshape([256,256,1]), chdim=2)

    d.addPairwiseEnergy(pairwise_energy, compat=10)  # `compat` is the "strength" of this potential.

    Q, tmp1, tmp2 = d.startInference()
    for _ in range(5):
        d.stepInference(Q, tmp1, tmp2)
    kl1 = d.klDivergence(Q) / (H*W)
    map_soln1 = np.argmax(Q, axis=0).reshape((H,W))

    return map_soln1

warnings.filterwarnings("ignore")


###### Parameters ######
ngf = 64
###### Parameters ######

input_image = Image.open(opt.img).convert('L')
input_image = input_image.resize((256, 256), Image.BILINEAR)
input_image = np.array(input_image).astype(np.uint8)

result_image = Image.open("temp/mask_ori.jpg").convert('RGB')
result_image = np.array(result_image.resize((256, 256), Image.BILINEAR))[:,:,0] / 255.

result_image = cv2.blur(np.array(result_image), (3, 3))
result_image = (result_image ** 0.5).astype(np.float32)

result = dense_crf(input_image,result_image).astype(np.float32)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

result_image = Image.open("temp/mask_ori.jpg").convert('RGB')
result_image = result_image.resize((256, 256), Image.BILINEAR)
result_image = ((np.array(result_image)[:,:,0]/(255.)) ** 0.5).astype(np.float32)
result_image[result_image<0.5] = 0
final = 1-((1-result)*(1-result_image))
final[final > 1] = 1

Final = np.zeros([256,256,3])

for i in range(3):
    Final[:,:,i] = final

plt.imsave("temp/mask_final.jpg",Final)
