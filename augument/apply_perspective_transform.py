# -*- coding: utf-8 -*-
# 文件描述: 图像透视变换

import json
import os
import cv2
import numpy as np 
import albumentations as A
from tqdm import tqdm

# 1.视角转换
# 2.bbox/text可能也会发生变化


def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_annotations(gpt_value):
    try:
        if isinstance(gpt_value, str):
            return json.loads(gpt_value)
        return gpt_value
    except:
        return []

def normalize_bbox(bbox, img_w, img_h, target_size=1000):
    """将像素坐标归一化到 target_size (1000)"""
    x1, y1, x2, y2 = bbox
    x1_norm = int(np.clip(x1 * target_size / img_w, 0, target_size))
    y1_norm = int(np.clip(y1 * target_size / img_h, 0, target_size))
    x2_norm = int(np.clip(x2 * target_size / img_w, 0, target_size))
    y2_norm = int(np.clip(y2 * target_size / img_h, 0, target_size))
    return [x1_norm, y1_norm, x2_norm, y2_norm]

def denormalize_bbox(bbox, img_w, img_h, target_size=1000):
    """将归一化坐标反转为像素坐标，并强制确保宽高 > 0"""
    x1, y1, x2, y2 = bbox
    x1_denorm = x1 * img_w / target_size
    y1_denorm = y1 * img_h / target_size
    x2_denorm = x2 * img_w / target_size
    y2_denorm = y2 * img_h / target_size

    if x2_denorm <= x1_denorm:
        x2_denorm = x1_denorm + 1.0
    if y2_denorm <= y1_denorm:
        y2_denorm = y1_denorm + 1.0

    return [x1_denorm, y1_denorm, x2_denorm, y2_denorm]

def apply_perspective_transform(image, annotations, max_retries=30):
    """
    应用随机透视变换，并严格保证标签文本的视觉完整性。
    """
    h, w = image.shape[:2]

    # 1. 准备原始 bbox
    valid_bboxes = []
    valid_labels = []
    for ann in annotations:
        denorm = denormalize_bbox(ann['bbox_2d'], w, h)
        if denorm[2] <= denorm[0] or denorm[3] <= denorm[1]:
            continue
        valid_bboxes.append(denorm)
        valid_labels.append(ann['text_content'])

    if not valid_bboxes:
        return image, annotations

    num_obj = len(valid_bboxes)
    # 提高可见度阈值到极高水平，减少被切断风险
    visibility_threshold = 1.0 if num_obj == 1 else 0.99

    transform = A.Compose([
        A.Perspective(
            scale=(0.05, 0.1), # 控制变换强度，避免过度形变
            keep_size=True,
            p=1.0
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=visibility_threshold,
        clip=True,
        min_width=1.0,
        min_height=1.0
    ))

    for _ in range(max_retries):
        try:
            transformed = transform(
                image=image,
                bboxes=valid_bboxes,
                class_labels=valid_labels
            )

            t_bboxes = transformed['bboxes']
            t_labels = transformed['class_labels']

            # 关键修复：除了面积检查，增加“物理边缘”检查
            # 如果任何一个保留下来的 bbox 紧贴图像边缘，认为其文本可能不完整，重新尝试
            if len(t_bboxes) > 0:
                is_safe = True
                for bbox in t_bboxes:
                    # 如果 bbox 已经触碰到画面的物理边缘 (0 或 w, h)
                    # 在 Pascal VOC 格式下: [x1, y1, x2, y2]
                    if bbox[0] <= 1 or bbox[2] >= w - 1 or bbox[1] <= 1 or bbox[3] >= h - 1:
                        is_safe = False
                        break
                
                if not is_safe:
                    continue

                # 构造返回结果
                new_annotations = []
                for bbox, label in zip(t_bboxes, t_labels):
                    bbox_norm = normalize_bbox(bbox, w, h, 1000)
                    # 最终合法性加固
                    if bbox_norm[2] <= bbox_norm[0]: bbox_norm[2] = min(1000, bbox_norm[0] + 1)
                    if bbox_norm[3] <= bbox_norm[1]: bbox_norm[3] = min(1000, bbox_norm[1] + 1)

                    new_annotations.append({
                        'text_content': label,
                        'bbox_2d': bbox_norm
                    })
                return transformed['image'], new_annotations
                
        except (ValueError, Exception):
            continue

    # 如果重试次数耗尽，返回原图以保证数据质量
    return image, annotations

def process_dataset(dataset, output_image_dir, output_json_path):
    os.makedirs(output_image_dir, exist_ok=True)
    new_dataset = []

    for item in tqdm(dataset, desc="处理图像"):
        image_path = item['images'][0]
        image_path = "/root/autodl-tmp/.autodl/data/" + image_path
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        gpt_value = item['conversations'][1]['value']
        annotations = parse_annotations(gpt_value)

        if annotations:
            transformed_image, new_annotations = apply_perspective_transform(image, annotations)
        else:
            transformed_image, new_annotations = image, []

        image_name = "persp_" + os.path.basename(image_path)
        output_image_full_path = os.path.join(output_image_dir, image_name)
        cv2.imwrite(output_image_full_path, transformed_image)

        new_item = {
            'conversations': [
                item['conversations'][0],
                {
                    'from': 'gpt',
                    'value': json.dumps(new_annotations, ensure_ascii=False)
                }
            ],
            'images': [os.path.join("train_images_perspective", image_name)]
        }
        new_dataset.append(new_item)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

def main():
    input_json = '/root/autodl-tmp/.autodl/data/train_sft_correct.json'
    output_json = '/root/autodl-tmp/.autodl/data/train_sft_perspective_correct.json'
    output_image_dir = '/root/autodl-tmp/.autodl/data/train_images_perspective'

    print(f"任务启动，图片将保存至: {output_image_dir}")
    if os.path.exists(input_json):
        dataset = load_dataset(input_json)
        process_dataset(dataset, output_image_dir, output_json)
        print(f"处理成功！新标注文件: {output_json}")
    else:
        print(f"错误：未找到输入文件 {input_json}")

if __name__ == '__main__':
    main()
