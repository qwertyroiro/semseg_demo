import argparse
import os
from mmdet.core.visualization.image import imshow_det_bboxes
import time
from collections import deque
from threading import Thread
import traceback

import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
import spacy
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2



import warnings
warnings.filterwarnings("ignore")


def oneformer_coco_segmentation(
        image, oneformer_coco_processor, oneformer_coco_model, rank
):
    inputs = oneformer_coco_processor(
        images=image, task_inputs=["semantic"], return_tensors="pt"
    ).to(rank)
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_map = (
        oneformer_coco_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
    )
    return predicted_semantic_map

def oneformer_ade20k_segmentation(
    image, oneformer_ade20k_processor, oneformer_ade20k_model, rank
):
    inputs = oneformer_ade20k_processor(
        images=image, task_inputs=["semantic"], return_tensors="pt"
    ).to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = (
        oneformer_ade20k_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
    )
    return predicted_semantic_map

def oneformer_cityscapes_segmentation(
        image, oneformer_cityscapes_processor, oneformer_cityscapes_model, rank
):
    inputs = oneformer_cityscapes_processor(
        images=image, task_inputs=["semantic"], return_tensors="pt"
    ).to(rank)
    outputs = oneformer_cityscapes_model(**inputs)
    predicted_semantic_map = (
        oneformer_cityscapes_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
    )
    return predicted_semantic_map

nlp = spacy.load("en_core_web_sm")


def get_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    return noun_phrases


def clip_classification(image, class_list, top_k, clip_processor, clip_model, rank):
    inputs = clip_processor(
        text=class_list, images=image, return_tensors="pt", padding=True
    ).to(rank)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    if top_k == 1:
        class_name = class_list[probs.argmax().item()]
        return class_name
    else:
        top_k_indices = probs.topk(top_k, dim=1).indices[0]
        top_k_class_names = [class_list[index] for index in top_k_indices]
        return top_k_class_names


def clipseg_segmentation(image, class_list, clipseg_processor, clipseg_model, rank):
    if isinstance(class_list, str):
        print(len(image), len(class_list))
        print(type(image), len(class_list))
        print(image.shape)
        class_list = [
            class_list,
        ]
    inputs = clipseg_processor(
        text=class_list,
        images=[image] * len(class_list),
        padding=True,
        return_tensors="pt",
    ).to(rank)
    # resize inputs['pixel_values'] to the longest side of inputs['pixel_values']
    h, w = inputs["pixel_values"].shape[-2:]
    fixed_scale = (512, 512)
    inputs["pixel_values"] = F.interpolate(
        inputs["pixel_values"], size=fixed_scale, mode="bilinear", align_corners=False
    )
    outputs = clipseg_model(**inputs)
    try:
        logits = F.interpolate(
            outputs.logits[None], size=(h, w), mode="bilinear", align_corners=False
        )[0]
    except Exception as e:
        logits = F.interpolate(
            outputs.logits[None, None, ...],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]
    return logits


def open_vocabulary_classification_blip(raw_image, blip_processor, blip_model, rank):
    # unconditional image captioning
    captioning_inputs = blip_processor(raw_image, return_tensors="pt").to(rank)
    out = blip_model.generate(**captioning_inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    ov_class_list = get_noun_phrases(caption)
    return ov_class_list


def load_filename_with_extensions(data_path, filename):
    """
    Returns file with corresponding extension to json file.
    Raise error if such file is not found.

    Args:
        filename (str): Filename (without extension).

    Returns:
        filename with the right extension.
    """
    full_file_path = os.path.join(data_path, filename)
    # List of image file extensions to attempt
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]
    # Iterate through image file extensions and attempt to upload the file
    for ext in image_extensions:
        # Check if the file with current extension exists
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext  # Return True if file is successfully uploaded
    raise FileNotFoundError(
        f"No such file {full_file_path}, checked for the following extensions {image_extensions}"
    )

rank = "cuda"

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(rank)

checkpoint_large = "ckp/sam_vit_l_0b3195.pth"
checkpoint_huge = "ckp/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_huge).to(rank)

oneformer_ade20k_processor = OneFormerProcessor.from_pretrained(
    "shi-labs/oneformer_ade20k_swin_tiny"
)
oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_ade20k_swin_tiny"
).to(rank)

oneformer_coco_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(rank)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(rank)

clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd16")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd16").to(rank)
clipseg_processor.image_processor.do_resize = False

pred_iou_thresh = 0.62
stability_score_thresh = 0.80

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=pred_iou_thresh, # 0.86 by default
    stability_score_thresh=stability_score_thresh, # 0.92 by default
    crop_n_layers=0,  # 1 by default
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
    output_mode="coco_rle",
)

def semantic_annotation_pipeline(img):
    try:
        scale_small = 1.2
        scale_large = 1.6
        scale_huge = 1.6
        with torch.no_grad():
            img = img
            anns = {"annotations": mask_generator.generate(img)}
            bitmasks, class_names = [], []

            # ade20kの結果を取得
            class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(img),
                                                                        oneformer_coco_processor,
                                                                        oneformer_coco_model, rank)
            class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(img),
                                                                            oneformer_ade20k_processor,
                                                                            oneformer_ade20k_model, rank)
            from tqdm import tqdm
            for ann in tqdm(anns["annotations"], leave=False):
                # SAEの各アノテーションのone-hotベクトルを取得
                valid_mask = torch.tensor(maskUtils.decode(ann["segmentation"])).bool()

                # valid_maskごとにade20kの結果を取得
                coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
                ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]

                # 結果の中から、最も多く出現したクラスを1つだけ取得
                top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
                top_k_ade20k_propose_classes_ids = (
                    torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
                )
                # cocoとade20kの和集合を取得
                local_class_names = set()
                local_class_names = set.union(local_class_names,
                                                set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for
                                                    class_id in top_k_ade20k_propose_classes_ids]))
                local_class_names = set.union(local_class_names, set(([
                    CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in
                    top_k_coco_propose_classes_ids])))

                # 今後の処理で使う1.2倍(patch_small), 1.6倍(patch_large), 1.6倍(patch_huge、なぜかlargeと同じサイズ)のsamで切り取った画像を取得
                # valid_mask_huge_cropはSAMの領域を切り取った画像??
                patch_small = mmcv.imcrop(
                    img,
                    np.array(
                        [
                            ann["bbox"][0],
                            ann["bbox"][1],
                            ann["bbox"][0] + ann["bbox"][2],
                            ann["bbox"][1] + ann["bbox"][3],
                        ]
                    ),
                    scale=scale_small,
                )
                patch_large = mmcv.imcrop(
                    img,
                    np.array(
                        [
                            ann["bbox"][0],
                            ann["bbox"][1],
                            ann["bbox"][0] + ann["bbox"][2],
                            ann["bbox"][1] + ann["bbox"][3],
                        ]
                    ),
                    scale=scale_large,
                )
                patch_huge = mmcv.imcrop(
                    img,
                    np.array(
                        [
                            ann["bbox"][0],
                            ann["bbox"][1],
                            ann["bbox"][0] + ann["bbox"][2],
                            ann["bbox"][1] + ann["bbox"][3],
                        ]
                    ),
                    scale=scale_huge,
                )
                valid_mask_huge_crop = mmcv.imcrop(
                    valid_mask.numpy(),
                    np.array(
                        [
                            ann["bbox"][0],
                            ann["bbox"][1],
                            ann["bbox"][0] + ann["bbox"][2],
                            ann["bbox"][1] + ann["bbox"][3],
                        ]
                    ),
                    scale=scale_huge,
                )

                # 1.6倍の画像(patch_large)を用いて、画像の中に含まれるクラスを取得。 画像から状況を文で説明し、その文からget_noun_phrases関数で名詞句を抽出し、返している
                op_class_list = open_vocabulary_classification_blip(patch_large, blip_processor, blip_model, rank)

                # cocoとade20kの和集合と、名詞句を抽出した結果の和集合を取得
                local_class_list = list(set.union(local_class_names, set(op_class_list))) # , set(refined_imagenet_class_names)
                # local_class_list = list(local_class_names)

                # openai/clip-vitを使って、和集合のlocal_class_listから、画像の中に含まれるクラスを3つ取得？
                mask_categories = clip_classification(patch_small, local_class_list,
                                                        3 if len(local_class_list) > 3 else len(local_class_list),
                                                        clip_processor, clip_model, rank)

                # mask_categories(clip_classificationの戻り値)がstr型の場合、list型に変換
                if isinstance(mask_categories, str):
                    mask_categories = [mask_categories]

                # 得られた3つのクラス(mask_categories)から、それがどこにあるのかをCIDAS/clipseg-rd16を使って推定？
                class_ids_patch_huge = clipseg_segmentation(patch_huge, mask_categories, clipseg_processor,
                                                            clipseg_model, rank).argmax(0)

                # テンソルに変換
                valid_mask_huge_crop = torch.tensor(valid_mask_huge_crop)

                # ????
                if valid_mask_huge_crop.shape != class_ids_patch_huge.shape:
                    valid_mask_huge_crop = F.interpolate(
                        valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(),
                        size=(class_ids_patch_huge.shape[-2], class_ids_patch_huge.shape[-1]),
                        mode='nearest').squeeze(0).squeeze(0).bool()

                top_1_patch_huge = torch.bincount(class_ids_patch_huge[valid_mask_huge_crop].flatten()).topk(
                    1).indices
                top_1_mask_category = mask_categories[top_1_patch_huge.item()]

                append_classname = str(top_1_mask_category)
                ann["class_name"] = append_classname
                class_names.append(append_classname)

                del ade20k_propose_classes_ids
                del top_k_ade20k_propose_classes_ids

            for ann in anns["annotations"]:
                bitmasks.append(maskUtils.decode(ann["segmentation"]))

            result = imshow_det_bboxes(
                img,
                bboxes=None,
                labels=np.arange(len(bitmasks)),
                segms=np.stack(bitmasks),
                class_names=class_names,
                mask_color="random",
                font_size=11,
                show=False,
                # out_file=os.path.join(output_path, filename+'_semantic.png')
            )
            
            # Delete variables that are no longer needed
            del img
            del anns
            del class_ids_from_oneformer_ade20k
            
            return result
            



    except Exception as e:
        traceback.print_exc()


def main():
    start_time = time.time()
    filename = "test_road.mp4"
    name, ext = os.path.splitext(filename)
    cap = cv2.VideoCapture(filename)
    # 総フレーム数を取得
    total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(frame_width), int(frame_height))
    # videowriterでmp4で保存
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{name}_{pred_iou_thresh}_{stability_score_thresh}_out.mp4', fourcc, 29.0, size)
    
    while True:
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(f"残り{int(frame_count/total_frame_count * 100)}%: {int(frame_count)} / {int(total_frame_count)}")
        try:
            ret, img = cap.read()
            if not ret:
                break
            
            result_img = semantic_annotation_pipeline(img)
            print("\033[1A", end="")
            out.write(result_img)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        except Exception as e:
            print(f"「KeyboardInterrupt」 or 「{e}」")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    print(f"処理時間: {end_time - start_time}")

main()
