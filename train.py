from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import pickle
import cv2
import time

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    LearningRateScheduler,
)

import tensorflow.keras.metrics as metrics
from tensorflow.keras.layers import Lambda

from yolov3_tf2.models import YoloV3, YoloLoss, yolo_anchors, yolo_anchor_masks
from yolov3_tf2.utils import freeze_all, draw_outputs, draw_outputs_predTrue
import yolov3_tf2.dataloader as dataloader

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  


import wandb
flags.DEFINE_string("dataset_path", "./ship1000_data/", "path to image and labels file")
flags.DEFINE_enum(
    "mode",
    "fit",
    ["fit", "eager_fit", "eager_tf"],
    "fit: model.fit, "
    "eager_fit: model.fit(run_eagerly=True), "
    "eager_tf: custom GradientTape",
)
flags.DEFINE_enum(
    "transfer",
    "none",
    ["none", "darknet", "no_output", "frozen", "fine_tune"],
    "none: Training from scratch, "
    "darknet: Transfer darknet, "
    "no_output: Transfer all but output, "
    "frozen: Transfer and freeze all, "
    "fine_tune: Transfer all and freeze darknet only",
)


SIZE = 416
EPOCHS = 1
BATCH_SIZE = 64
NUM_CLASSES = 7

LEARNING_RATE = 0.5e-2

MODE = "fit"  # ["fit", "eager_fit", "eager_tf"],
TRANSFER = "no_output"  # ["none", "darknet", "no_output", "frozen", "fine_tune"]

PATH_WEIGHTS = "./checkpoints/yolov3.tf"
WEIGHTS_NUM_CLASSES = 80

CLASSES = "./data/ship.names"
OUTPUT_PATH = "./output/"


def setup_model():
    model = YoloV3(SIZE, training=True, classes=NUM_CLASSES)
    model_vd = YoloV3(SIZE, classes=NUM_CLASSES)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if TRANSFER == "none":
        pass  # Nothing to do
    elif TRANSFER in ["darknet", "no_output"]:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        model_pretrained = YoloV3(
            SIZE, training=True, classes=WEIGHTS_NUM_CLASSES or NUM_CLASSES
        )
        model_pretrained.load_weights(PATH_WEIGHTS)

        if TRANSFER == "darknet":
            model.get_layer("yolo_darknet").set_weights(
                model_pretrained.get_layer("yolo_darknet").get_weights()
            )
            freeze_all(model.get_layer("yolo_darknet"))
        elif TRANSFER == "no_output":
            for l in model.layers:
                if not l.name.startswith("yolo_output"):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(PATH_WEIGHTS)

        if TRANSFER == "fine_tune":
            # freeze darknet and fine tune other layers
            darknet = model.get_layer("yolo_darknet")
            freeze_all(darknet)
        elif TRANSFER == "frozen":
            # freeze everything
            freeze_all(model)

    cos_decay = tf.keras.optimizers.schedules.CosineDecay(LEARNING_RATE, 100)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cos_decay)
    loss = [YoloLoss(anchors[mask], classes=NUM_CLASSES) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=(MODE == "eager_fit"))

    return model, model_vd, optimizer, loss, anchors, anchor_masks


def main(_argv):
    wandb.init("king")
    wandb.run.name = wandb.run.id  # Generated run ID
    wandb.run.save()
    wandb.config.update(
        {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "SIZE": SIZE, "lr": LEARNING_RATE}
    )

    wandb.run.log_code(".")

    # Setup
    model, model_vd, optimizer, loss, anchors, anchor_masks = setup_model()

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    train_dataset = dataloader.Dataloader(
        FLAGS.dataset_path + "train", anchors, anchor_masks, BATCH_SIZE, shuffle=True
    )

    val_dataset = dataloader.Dataloader(
        FLAGS.dataset_path + "validation",
        anchors,
        anchor_masks,
        BATCH_SIZE,
        shuffle=True,
        isVal=True,
    )

    ############
    if MODE == "eager_tf":
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = metrics.Mean("loss", dtype=tf.float32)
        avg_val_loss = metrics.Mean("val_loss", dtype=tf.float32)

        for epoch in range(1, EPOCHS + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    # regularization_loss = 0
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                logging.info(
                    "{}_train_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_loss.update_state(total_loss)

            train_dataset.on_epoch_end()

            ### draw box code by detect ######
            model.save_weights("./checkpoints/tmpWeights.tf")
            model_vd.load_weights("./checkpoints/tmpWeights.tf").expect_partial()

            class_names = [c.strip() for c in open(CLASSES).readlines()]
            for i in range(2):
                img_raw = images[i]
                img_input = tf.expand_dims(img_raw, 0)
                boxes, scores, classes, nums = model_vd(img_input)

                normalizedImg = np.zeros((SIZE, SIZE))
                img_norm = cv2.normalize(
                    img_raw.numpy(), normalizedImg, 0, 255, cv2.NORM_MINMAX
                )

                img = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(f"{OUTPUT_PATH}/output_tr{epoch}_{i}.jpg", img)
                logging.info(
                    "output saved to: {} tr_{}_{}".format(OUTPUT_PATH, epoch, i)
                )
                wandb.log({"tr_output": [wandb.Image(img, caption=f"tr_{epoch}_{i}")]})

            #############################################
            #############   val_dataset    ##############
            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info(
                    "{}_val_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_val_loss.update_state(total_loss)

            ### draw box code by detect ######
            for i in range(len(images)):
                img_raw = images[i]
                img_input = tf.expand_dims(img_raw, 0)
                boxes, scores, classes, nums = model_vd(img_input)

                img_norm = cv2.normalize(img_raw.numpy(), None, 0, 255, cv2.NORM_MINMAX)

                img = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
                # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                pred_outputs = (boxes, scores, classes, nums)
                trues = (boxes, scores, classes, nums)
                img = draw_outputs_predTrue(img, pred_outputs, trues, class_names)
                cv2.imwrite(f"{OUTPUT_PATH}/output_vd{epoch}_{i}.jpg", img)
                logging.info(
                    "output saved to: {} vd_{}_{}".format(OUTPUT_PATH, epoch, i)
                )
                wandb.log({"vd_output": [wandb.Image(img, caption=f"vd_{epoch}_{i}")]})

            val_dataset.on_epoch_end()
            ####################################
            logging.info(
                "{}, train: {}, val: {}".format(
                    epoch, avg_loss.result().numpy(), avg_val_loss.result().numpy()
                )
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "trainig_loss": avg_loss.result().numpy(),
                    "valid_loss": avg_val_loss.result().numpy(),
                }
            )
            avg_loss.reset_states()
            avg_val_loss.reset_states()

        model.save_weights(f"./checkpoints/yolov3_train.tf")
        ############

    else:

        class MyCustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                tr_imgs, _ = next(iter(train_dataset))
                val_imgs, _ = next(iter(val_dataset))

                model.save_weights("./checkpoints/tmpWeights.tf")
                model_vd.load_weights("./checkpoints/tmpWeights.tf").expect_partial()
                ### draw box code by detect ######
                class_names = [c.strip() for c in open(CLASSES).readlines()]
                for i in range(2):
                    img_raw = tr_imgs[i]
                    img_input = tf.expand_dims(img_raw, 0)
                    boxes, scores, classes, nums = model_vd(img_input)

                    normalizedImg = np.zeros((SIZE, SIZE))
                    img_norm = cv2.normalize(
                        img_raw.numpy(), normalizedImg, 0, 255, cv2.NORM_MINMAX
                    )

                    img = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
                    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                    cv2.imwrite(f"{OUTPUT_PATH}/output_tr{epoch}_{i}.jpg", img)
                    logging.info(
                        "output saved to: {} tr_{}{}".format(OUTPUT_PATH, epoch, i)
                    )

                    wandb.log(
                        {"tr_output": [wandb.Image(img, caption=f"tr_{epoch}_{i}")]}
                    )
                ####################################
                for i in range(2):
                    img_raw = val_imgs[i]
                    img_input = tf.expand_dims(img_raw, 0)
                    boxes, scores, classes, nums = model_vd(img_input)

                    normalizedImg = np.zeros((SIZE, SIZE))
                    img_norm = cv2.normalize(
                        img_raw.numpy(), normalizedImg, 0, 255, cv2.NORM_MINMAX
                    )

                    img = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
                    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                    cv2.imwrite(f"{OUTPUT_PATH}/output_vd{epoch}_{i}.jpg", img)
                    logging.info(
                        "output saved to: {} vd_{}{}".format(OUTPUT_PATH, epoch, i)
                    )

                    wandb.log(
                        {"tr_output": [wandb.Image(img, caption=f"vd_{epoch}_{i}")]}
                    )
                ####################################

        callbacks = [
            # ReduceLROnPlateau(verbose=1),
            # EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint(
                "./checkpoints/train_{epoch}.tf",
                verbose=1,
                save_weights_only=True,
            ),
            TensorBoard(log_dir="logs"),
            MyCustomCallback(),
        ]

        start_time = time.time()
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=val_dataset,
        )
        end_time = time.time() - start_time
        print(f"Total Training Time: {end_time}")

        model.save_weights(f"./checkpoints/yolov3_train.tf")
        with open(f"./output_hist/histoty", "wb") as file:
            pickle.dump(history.history, file)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
