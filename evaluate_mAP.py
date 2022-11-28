from absl import app

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Lambda

from yolov3_tf2.models import (
    YoloV3,
    YoloLoss,
    yolo_boxes,
    yolo_nms,
    yolo_anchors,
    yolo_anchor_masks,
)
import yolov3_tf2.dataloader as dataloader
import yolov3_tf2.dataset as dataset

import shutil
import json
import time
import tqdm

import os

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ###############


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:  for i=numel(mpre)-1:-1:1
                                mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def get_mAP(Yolo, ts_dataset):
    MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

    CLASSES = "./data/ship.names"
    class_names = [c.strip() for c in open(CLASSES).readlines()]
    NUM_CLASS = class_names

    ground_truth_dir_path = "./mAP/ground-truth"
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists("mAP"):
        os.mkdir("mAP")
    os.mkdir(ground_truth_dir_path)

    # print(f"\ncalculating mAP{int(iou_threshold*100)}...\n")

    gt_counter_per_class = {}
    # for index in range(dataset.num_samples):
    for index, (images, labels) in tqdm(enumerate(ts_dataset)):
        original_image = images
        bbox_data_gt = labels

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            bboxes_gt = np.array(bboxes_gt)
            classes_gt = np.array(classes_gt)
        # ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + ".txt")
        num_bbox_gt = len(bboxes_gt)

        bounding_boxes = []
        for i in range(num_bbox_gt):
            class_name = NUM_CLASS[int(classes_gt[i])]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            bounding_boxes.append(
                {"class_name": class_name, "bbox": bbox, "used": False}
            )

            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
            bbox_mess = " ".join([class_name, xmin, ymin, xmax, ymax]) + "\n"
        with open(
            f"{ground_truth_dir_path}/{str(index)}_ground_truth.json", "w"
        ) as outfile:
            json.dump(bounding_boxes, outfile)

    gt_counter_per_class[NUM_CLASS[-1]] = 0
    gt_classes = list(gt_counter_per_class.keys())
    # if len(gt_classes) < len(NUM_CLASS):
    #     gt_classes.append(NUM_CLASS[-1])
    # sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    times = []
    json_pred = [[] for i in range(n_classes)]

    for index, (images, labels) in tqdm(enumerate(ts_dataset)):
        img_raw = images

        t1 = time.time()
        img_resize_scale = tf.image.resize(img_raw, (416, 416)) / 255
        img_input = tf.expand_dims(img_resize_scale, 0)
        boxes, scores, classes, nums = Yolo(img_input)

        t2 = time.time()
        times.append(t2 - t1)

        # bboxes = postprocess_boxes(boxes, scores, classes, img_raw, TEST_INPUT_SIZE, score_threshold)
        # bboxes = nms(bboxes, iou_threshold, method='nms')
        boxes = boxes[0]
        scores = scores[0]
        classes = classes[0]
        for bbox, score, class_ in zip(boxes, scores, classes):
            # coor = np.array(bbox[:4], dtype=np.int32)
            coor = bbox[:4].numpy()
            score = np.float(score)
            class_ind = int(class_)
            class_name = NUM_CLASS[class_ind]
            score = "%.4f" % score
            xmin, ymin, xmax, ymax = list(map(str, coor))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            json_pred[gt_classes.index(class_name)].append(
                {"confidence": str(score), "file_id": str(index), "bbox": str(bbox)}
            )

    ms = sum(times) / len(times) * 1000
    fps = 1000 / ms

    for class_name in gt_classes:
        json_pred[gt_classes.index(class_name)].sort(
            key=lambda x: float(x["confidence"]), reverse=True
        )
        with open(
            f"{ground_truth_dir_path}/{class_name}_predictions.json", "w"
        ) as outfile:
            json.dump(json_pred[gt_classes.index(class_name)], outfile)

    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    with open("./mAP/results.txt", "w") as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_file = f"{ground_truth_dir_path}/{class_name}_predictions.json"
            predictions_data = json.load(open(predictions_file))

            # Assign predictions to ground truth objects
            nd = len(predictions_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = f"{ground_truth_dir_path}/{str(file_id)}_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [
                    float(x) for x in prediction["bbox"].split()
                ]  # bounding box of prediction
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [
                            float(x) for x in obj["bbox"].split()
                        ]  # bounding box of ground truth
                        bi = [
                            max(bb[0], bbgt[0]),
                            max(bb[1], bbgt[1]),
                            min(bb[2], bbgt[2]),
                            min(bb[3], bbgt[3]),
                        ]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (
                                (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                                + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                                - iw * ih
                            )
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:  # if ovmax > minimum overlap
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, "w") as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / (gt_counter_per_class[class_name] + 1e-12)
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / ((fp[idx] + tp[idx]) + 1e-12)
            # print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = (
                "{0:.3f}%".format(ap * 100) + " = " + class_name + " AP  "
            )  # class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = ["%.3f" % elem for elem in prec]
            rounded_rec = ["%.3f" % elem for elem in rec]
            # Write to results.txt
            results_file.write(
                text
                + "\n Precision: "
                + str(rounded_prec)
                + "\n Recall   :"
                + str(rounded_rec)
                + "\n\n"
            )

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP * 100, fps)
        results_file.write(text + "\n")
        print(text)

        return mAP * 100, text


def main(_argv):
    SIZE = 416
    NUM_CLASSES = 7

    path_weight = "./checkpoints/yolov3_train.tf"

    yolo = YoloV3(SIZE, classes=NUM_CLASSES)
    yolo.load_weights(path_weight).expect_partial()
    LEARNING_RATE = 1e-2
    optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
    loss = [
        YoloLoss(yolo_anchors[mask], classes=NUM_CLASSES) for mask in yolo_anchor_masks
    ]

    yolo.compile(optimizer=optimizer, loss=loss)

    testset = dataloader.Dataloader_all("./ship1000_data/validation")

    result, txt = get_mAP(
        yolo,
        testset,
    )

    print(result, txt)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
