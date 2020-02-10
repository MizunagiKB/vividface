# ------------------------------------------------------------------ import(s)
import sys
import os
import hashlib
import io
import time

# -------------------------------------------------------------------- bottle
import bottle
import numpy as np
import PIL.Image
import cv2

# -------------------------------------------------------------------- nnabla
import nnabla as nn
import nnabla.logger as L
import nnabla.utils as U
import nnabla.utils.image_utils
import nnabla.utils.nnp_graph

import vividface_nnl

IMAGE_W = vividface_nnl.IMAGE_W
IMAGE_H = vividface_nnl.IMAGE_H
IMAGE_D = vividface_nnl.IMAGE_D
TARGET_CHARA = vividface_nnl.TARGET_CHARA

MODEL_PATHNAME = "vividface_nnc_model/model.nnp"

EDGE_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 255)

IMWORK_DIR = "./imwork"
AUTO_IMWORK_CLEAN = False
IMWORK_EXPIRE_SEC = 1 * 60 * 60


# ------------------------------------------------------------------- class(s)
# ---------------------------------------------------------------- function(s)
def imwork_clean(expire_sec):

    current_time = time.time()

    for dir_name, _, list_filename in os.walk(IMWORK_DIR):
        for filename in list_filename:
            if os.path.splitext(filename)[1] in (".png", ".jpg", ".jpeg"):
                pathname = os.path.join(dir_name, filename)
                oss = os.stat(pathname)
                if (current_time - oss.st_mtime) > expire_sec:
                    os.remove(pathname)


@bottle.route("/")
def html_index():
    return bottle.template("index")


@bottle.route("/imwork/<img_filepath:path>", name="imwork")
def res_image(img_filepath):
    return bottle.static_file(img_filepath, root=IMWORK_DIR)


@bottle.route("/decide")
def html_decide():
    bottle.redirect("/")


@bottle.route("/decide", method="POST")
def do_upload():
    try:
        upload = bottle.request.files.get("upload", "")
        if os.path.splitext(upload.filename)[1].lower() not in (".png", ".jpg",
                                                                ".jpeg"):
            bottle.redirect("/")
    except AttributeError:
        bottle.redirect("/")

    data_raw = upload.file.read()

    image_hash = hashlib.sha1(data_raw).hexdigest()

    data_pil = PIL.Image.open(io.BytesIO(data_raw))
    if data_pil.mode != "RGB":
        data_pil = data_pil.convert("RGB")

    clip_cv2 = cv2.cvtColor(np.asarray(data_pil), cv2.COLOR_RGB2BGR)
    data_cv2 = cv2.cvtColor(np.asarray(data_pil), cv2.COLOR_RGB2BGR)

    cv2_cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")

    gry = cv2.cvtColor(data_cv2, cv2.COLOR_BGR2GRAY)
    gry = cv2.equalizeHist(gry)

    list_face = cv2_cascade.detectMultiScale(gry,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(IMAGE_W, IMAGE_H))

    nn.clear_parameters()
    nnc_nnp = U.nnp_graph.NnpLoader(MODEL_PATHNAME)
    nnc_net = nnc_nnp.get_network("MainRuntime", 1)
    net_x = nnc_net.inputs["Input"]
    net_y = nnc_net.outputs["Softmax"]

    list_result = []
    for idx, tpl_region in enumerate(list_face):
        ix, iy, iw, ih = tpl_region

        o_ref = cv2.cvtColor(clip_cv2[iy:iy + ih, ix:ix + iw],
                             cv2.COLOR_BGR2RGB)
        o_ref = U.image_utils.imresize(o_ref, (IMAGE_W, IMAGE_H))

        face_pathname = os.path.join(IMWORK_DIR,
                                     "%s_%02d.png" % (image_hash, idx))
        U.image_utils.imsave(face_pathname, o_ref)

        net_x.d = [(o_ref.transpose(2, 0, 1) / 255)]
        net_y.forward()
        list_detect_rate = net_y.d[0].tolist()

        rate_max = max(list_detect_rate)
        n_idx = list_detect_rate.index(rate_max)
        chracter_name = TARGET_CHARA[n_idx]

        cv2.rectangle(data_cv2, (ix, iy), (ix + iw, iy + ih), EDGE_COLOR, 3)
        cv2.rectangle(data_cv2, (ix, iy), (ix + iw, iy + ih), LINE_COLOR, 1)
        for txt_y in (-2, -1, 0, 1, 2):
            for txt_x in (-2, -1, 0, 1, 2):
                cv2.putText(data_cv2, "%d) %s" % (idx, chracter_name),
                            (ix + 3 + txt_x, iy + ih + 16 + txt_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, EDGE_COLOR)
        cv2.putText(data_cv2, "%d) %s" % (idx, chracter_name),
                    (ix + 3, iy + ih + 16), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    LINE_COLOR)

        list_result.append([face_pathname, chracter_name, list_detect_rate])

    if AUTO_IMWORK_CLEAN is True:
        imwork_clean(IMWORK_EXPIRE_SEC)

    pathname = os.path.join(IMWORK_DIR, "%s.jpg" % (image_hash, ))
    cv2.imwrite(pathname, data_cv2)

    return bottle.template("index", pathname=pathname, list_result=list_result)


if __name__ == "__main__":
    bottle.run(host="localhost", port=8001, debug=True, reloader=True)

# [EOF]
