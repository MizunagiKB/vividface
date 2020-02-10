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

DATASET_DIR = "./dataset"
IMAGE_W = 48
IMAGE_H = 48
IMAGE_D = 3
LOAD_SIZE = 100
BATCH_SIZE = len(TARGET_CHARA) * 20
MIN_SAVE_SCORE = 0.75

VIVIDFACE_TRAIN = "vividface_train_%dx%d" % (IMAGE_W, IMAGE_H)
VIVIDFACE_VALID = "vividface_valid_%dx%d" % (IMAGE_W, IMAGE_H)


# ------------------------------------------------------------------- class(s)
class CDataset(object):
    def __init__(self, chara_name, filename, nnl_image, nnl_onehot):
        self.chara_name = chara_name
        self.filename = filename
        self.nnl_image = nnl_image
        self.nnl_onehot = nnl_onehot


# ---------------------------------------------------------------- function(s)
def build(in_x, in_y, train=True):

    if train is True:
        h = F.image_augmentation(in_x, (IMAGE_D, IMAGE_W, IMAGE_H), (0, 0),
                                 1.0, 1.0, 0.0, 1.0, 0.0, False, False, 0.0,
                                 False, 1.1, 0.5, False, 0.0, 0)
    else:
        h = F.image_augmentation(in_x)

    with nn.parameter_scope("conv1"):
        h = PF.convolution(h, 8, (2, 2), stride=(2, 2), pad=(0, 0))
        h = PF.batch_normalization(h,
                                   axes=(1, ),
                                   decay_rate=0.9,
                                   eps=0.0001,
                                   batch_stat=train)
        h = F.relu(h, True)

    with nn.parameter_scope("conv2"):
        h = PF.convolution(h, 16, (2, 2), stride=(2, 2), pad=(0, 0))
        h = PF.batch_normalization(h,
                                   axes=(1, ),
                                   decay_rate=0.9,
                                   eps=0.0001,
                                   batch_stat=train)
        h = F.relu(h, True)

    with nn.parameter_scope("conv3"):
        h = PF.convolution(h, 64, (3, 3), stride=(2, 2), pad=(1, 1))
        h = PF.batch_normalization(h,
                                   axes=(1, ),
                                   decay_rate=0.9,
                                   eps=0.0001,
                                   batch_stat=train)
        h = F.relu(h, True)

    with nn.parameter_scope("affine4"):
        h = PF.affine(h, len(TARGET_CHARA) * 20)
        h = F.relu(h, True)

    with nn.parameter_scope("affine5"):
        h = PF.affine(h, len(TARGET_CHARA) * 10)
        h = F.relu(h, True)

    with nn.parameter_scope("affine6"):
        h = PF.affine(h, len(TARGET_CHARA) * 3)
        h = F.relu(h, True)

    with nn.parameter_scope("affine7"):
        h = PF.affine(h, len(TARGET_CHARA))
        h = F.softmax(h)

    return h


def train(list_train, list_valid, epoch_limit=10000):

    x = nn.Variable(shape=(BATCH_SIZE, IMAGE_D, IMAGE_W, IMAGE_H))
    y = nn.Variable(shape=(BATCH_SIZE, len(TARGET_CHARA)))
    f = build(x, None)

    h = F.squared_error(f, y)

    loss = F.mean(h)

    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

    for _ in range(10):
        random.shuffle(list_train)

    epoch = 1
    while True:

        for n in range(0, len(list_train), BATCH_SIZE):

            x.d = [o.nnl_image for o in list_train[n:n + BATCH_SIZE]]
            y.d = [o.nnl_onehot for o in list_train[n:n + BATCH_SIZE]]

            loss.forward()
            solver.zero_grad()
            loss.backward()
            solver.update()

        if (epoch % 10) == 0:
            list_result = valid(list_valid)
            min_score = min(list_result)
            L.info("epoch(s): [%6d]  score: [%.3f] loss: [%.12f]" %
                   (epoch, min_score, loss.d))
            if min_score > MIN_SAVE_SCORE:
                model_filename = "vividface_nnl_model/train_model_%06d_%03d_%6f.h5" % (
                    epoch, int(min_score * 100), loss.d)
                nn.save_parameters(model_filename)

        epoch += 1
        if epoch_limit > 0:
            if epoch > epoch_limit:
                break


def valid(list_valid):

    x = nn.Variable(shape=(len(list_valid), IMAGE_D, IMAGE_W, IMAGE_H))
    f = build(x, None, False)

    list_size = [0] * len(TARGET_CHARA)
    list_true = [0] * len(TARGET_CHARA)
    list_ans = [0] * len(TARGET_CHARA)

    list_x = [o.nnl_image for o in list_valid]
    list_y = [o.nnl_onehot for o in list_valid]

    x.d = list_x

    f.forward()

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

    x = nn.Variable(shape=(1, IMAGE_D, IMAGE_W, IMAGE_H))
    x.d = [nnl_image]

    f = build(x, None, False)
    f.forward()

    return f.d[0].tolist()


def dataset_save(dir_name, basename, list_data):

    csv_file = os.path.join(dir_name, basename) + ".csv"
    print(csv_file)

    with open(csv_file, "w") as hw:
        csv_w = csv.writer(hw)
        csv_w.writerow(["x"] + [
            "y__%d:%s" % (n, TARGET_CHARA[n]) for n in range(len(TARGET_CHARA))
        ])
        for o in list_data:
            csv_w.writerow(["./%s/%s" % (o.chara_name, o.filename)] +
                           o.nnl_onehot)


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

    dataset_save(DATASET_DIR, VIVIDFACE_TRAIN, list_train)
    dataset_save(DATASET_DIR, VIVIDFACE_VALID, list_valid)

    train(list_train, list_valid)


if __name__ == "__main__":
    main()

# [EOF]
