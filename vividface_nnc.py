# ------------------------------------------------------------------ import(s)
import sys
import os
import random
import csv

import numpy as np

# -------------------------------------------------------------------- nnabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.logger as L
import nnabla.utils as U
import nnabla.utils.image_utils
import nnabla.utils.nnp_graph

try:
    import nnabla.ext_utils
    ctx = nnabla.ext_utils.get_extension_context("cudnn")
    nn.set_default_context(ctx)
except:
    pass

TARGET_CHARA = {
    0: "corona",
    1: "einhald",
    2: "fuka",
    3: "miura",
    4: "rinne",
    5: "rio",
    6: "vivio"
}
MODEL_PATHNAME = "vividface_nnc_model/model.nnp"
DATASET_DIR = "./dataset"
IMAGE_W = 48
IMAGE_H = 48
IMAGE_D = 3
LOAD_SIZE = 100
BATCH_SIZE = len(TARGET_CHARA) * 20


# ------------------------------------------------------------------- class(s)
class CDataset(object):
    def __init__(self, chara_name, filename, nnl_image, nnl_onehot):
        self.chara_name = chara_name
        self.filename = filename
        self.nnl_image = nnl_image
        self.nnl_onehot = nnl_onehot


# ---------------------------------------------------------------- function(s)
def valid(list_valid):

    nn.clear_parameters()
    nnc_nnp = U.nnp_graph.NnpLoader(MODEL_PATHNAME)
    nnc_net = nnc_nnp.get_network("MainRuntime", len(list_valid))

    x = nnc_net.inputs["Input"]
    f = nnc_net.outputs["Softmax"]

    list_size = [0] * len(TARGET_CHARA)
    list_true = [0] * len(TARGET_CHARA)
    list_ans = [0] * len(TARGET_CHARA)

    list_x = [o.nnl_image for o in list_valid]
    list_y = [o.nnl_onehot for o in list_valid]

    x.d = list_x

    f.forward(clear_buffer=True)

    for y, result in zip(list_y, f.d):
        v_max = max(result)
        n_idx = result.tolist().index(v_max)
        list_onehot = [0] * len(TARGET_CHARA)
        list_onehot[n_idx] = 1

        list_size[y.index(1)] += 1
        list_ans[n_idx] += 1
        if list_onehot == y:
            list_true[n_idx] += 1

    list_result = []
    for size, true in zip(list_size, list_true):
        if size > 0:
            list_result.append(true / size)
        else:
            list_result.append(0)

    L.info("size     " + " ".join(["%6.2f" % v for v in list_size]))
    L.info("true     " + " ".join(["%6.2f" % v for v in list_true]))
    L.info("ans      " + " ".join(["%6.2f" % v for v in list_ans]))
    L.info("result   " + " ".join(["%6.2f" % v for v in list_result]))

    return list_result


def inference(nnl_image):

    nn.clear_parameters()
    nnc_nnp = U.nnp_graph.NnpLoader(MODEL_PATHNAME)
    nnc_net = nnc_nnp.get_network("MainRuntime", 1)

    x = nnc_net.inputs["Input"]
    f = nnc_net.outputs["Softmax"]

    x.d = [nnl_image]

    f.forward()

    return f.d[0].tolist()


def main():

    list_train = []
    list_valid = []

    for dir_name, _, list_filename in os.walk(DATASET_DIR):
        _, chara_name = os.path.split(dir_name)
        if chara_name in TARGET_CHARA.values():
            list_image = []
            for filename in list_filename:
                if os.path.splitext(filename)[1].lower() in (".png", ".jpg",
                                                             ".jpeg"):

                    nnl_image = U.image_utils.imread(os.path.join(
                        dir_name, filename),
                                                     size=(IMAGE_W, IMAGE_H),
                                                     channel_first=False)

                    nnl_image = nnl_image.transpose(2, 0, 1)

                    list_onehot = [0] * len(TARGET_CHARA)
                    dict_chara = {v: k for k, v in TARGET_CHARA.items()}
                    list_onehot[dict_chara[chara_name]] = 1

                    list_image.append(
                        CDataset(chara_name, filename, nnl_image / 255.0,
                                 list_onehot))

                if len(list_image) == LOAD_SIZE:
                    break

            list_train += list_image[0:-20]
            list_valid += list_image[-20:]

            L.info("%s %4d %4d" %
                   (chara_name, len(list_train), len(list_valid)))

    valid(list_valid)
    L.info(inference(list_valid[0].nnl_image))


if __name__ == "__main__":
    main()

# [EOF]