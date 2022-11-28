from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3


from yolov3_tf2.dataset import transform_images
import yolov3_tf2.dataloader as dataloader
from yolov3_tf2.utils import draw_outputs, draw_outputs_custom

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


flags.DEFINE_boolean("isValidation", True, "validation dateset or custom image")
flags.DEFINE_string("image", "./data/objects365_ship.jpg", "path to input image")


def main(_argv):
    PATH_WEIGHTS = "./checkpoints/yolov3_train.tf"
    OUTPUT_PATH = "./output"

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=7)

    yolo.load_weights(PATH_WEIGHTS).expect_partial()
    logging.info("weights loaded")

    classes_name_file = "./data/ship.names"
    class_names = [c.strip() for c in open(classes_name_file).readlines()]
    logging.info("classes loaded")

    dataset = dataloader.Dataloader_all(
        "./ship1000_data/validation"
    )

    if FLAGS.isValidation :
        cnt = 0
        for _, (images, labels) in enumerate(dataset):
            img_raw = images
            img_raw = tf.image.resize(img_raw, (416, 416)) / 255
            img_input = tf.expand_dims(img_raw, 0)
            boxes, scores, classes, nums = yolo(img_input)

            single_label = labels

            img_norm = cv2.normalize(img_raw.numpy(), None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)

            img = draw_outputs_custom(
                img, (boxes, scores, classes, nums), class_names, True
            )

            boxes = []
            scores = []
            classes = []
            for x1, y1, x2, y2, label in single_label:
                if x1 == 0 and x2 == 0:
                    continue
                boxes.append((x1, y1, x2, y2))
                scores.append(1)
                classes.append(label)
            nums = [len(boxes)]
            boxes = [boxes]
            scores = [scores]
            classes = [classes]

            img = draw_outputs_custom(
                img, (boxes, scores, classes, nums), class_names, False
            )
            # img = draw_outputs_True(img, pred_outputs, labels, class_names)
            cv2.imwrite(f"{OUTPUT_PATH}/detect_vd{cnt}.jpg", img)
            cnt = cnt + 1
    else:
        img_raw = tf.image.decode_image(open(FLAGS.image, "rb").read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, 416)

        boxes, scores, classes, nums = yolo(img)

        logging.info("detections:")
        for i in range(nums[0]):
            logging.info(
                "\t{}, {}, {}".format(
                    class_names[int(classes[0][i])],
                    np.array(scores[0][i]),
                    np.array(boxes[0][i]),
                )
            )

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(OUTPUT_PATH + "/detect_output.jpg", img)
        logging.info("output saved to: {}/detect_output.jpg".format(OUTPUT_PATH))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
