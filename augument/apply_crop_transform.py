# -*- coding: utf-8 -*-
# 1.crop之后车位号可能减少
# 2.crop之后图片不合理，丢弃
# 3.bbox，text都需要做相应的调整


import json
import os
import random
import numpy as np
from PIL import Image
import albumentations as A

def parse_conversations(value):
    """从conversations的value中解析出标注数据"""
    try:
        return json.loads(value)
    except:
        return []

def get_safe_crop_params(annotations, orig_w, orig_h, min_crop_ratio=0.6):
    """针对单框情况：计算一个包含该框且不改变宽高比的裁剪区域"""
    ann = annotations[0]
    bbox = ann['bbox_2d']
    x1, y1, x2, y2 = (bbox[0] * orig_w / 1000, bbox[1] * orig_h / 1000, 
                      bbox[2] * orig_w / 1000, bbox[3] * orig_h / 1000)
    
    bw, bh = x2 - x1, y2 - y1
    scale = random.uniform(min_crop_ratio, 0.95)
    crop_w, crop_h = orig_w * scale, orig_h * scale

    if bw > crop_w or bh > crop_h:
        crop_w, crop_h = min(orig_w, bw * 1.2), min(orig_h, bh * 1.2)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x_min = int(max(0, min(orig_w - crop_w, cx - crop_w / 2)))
    y_min = int(max(0, min(orig_h - crop_h, cy - crop_h / 2)))
    
    return x_min, y_min, int(x_min + crop_w), int(y_min + crop_h)

def process_image(item, output_image_dir, output_idx, max_retries = 15):
    image_path = item['images'][0]
    # 获取脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上一层到 .autodl，再进入 data
    data_root = os.path.join(base_dir, "..", "data")
    data_root = os.path.abspath(data_root)

    full_image_path = os.path.join(data_root, image_path)
    print(f"DEBUG full path: {full_image_path}")  
    if not os.path.exists(full_image_path):
        print(f'警告: 找不到图片，期望路径 = {full_image_path}')
        return None, None

    img = Image.open(full_image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)

    annotations = parse_conversations(item['conversations'][1]['value'])
    
    # --- 关键修改点 1: 分离坐标和标签 ---
    bboxes = []
    text_labels = []
    for ann in annotations:
        b = ann['bbox_2d']
        # 仅存放 4 个坐标数值
        bboxes.append([b[0]/1000, b[1]/1000, b[2]/1000, b[3]/1000])
        # 字符串存放在独立的列表中
        text_labels.append(ann['text_content'])

    # --- 关键修改点 2: 增加 label_fields 定义 ---
    bbox_params = A.BboxParams(
        format='albumentations', 
        min_visibility=0.95, # 0.95 表示bbox至少要有 95% 的区域保留在图像内
        label_fields=['text_labels'] # 标记text_labels 是对应的标签字段
    )

    final_img = None
    final_bboxes = []
    final_labels = []


    for _ in range(max_retries):     # 随机crop尝试次数
        if len(annotations) == 1:
            x_min, y_min, x_max, y_max = get_safe_crop_params(annotations, orig_w, orig_h)
            aug = A.Compose([A.Crop(x_min, y_min, x_max, y_max)], bbox_params=bbox_params)
        else:
            scale = random.uniform(0.6, 0.9) # 有些车位号被crop
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            aug = A.Compose([A.RandomCrop(height=new_h, width=new_w)], bbox_params=bbox_params)

        # --- 关键修改点 3: 同时传入坐标和标签 ---
        transformed = aug(image=img_np, bboxes=bboxes, text_labels=text_labels)
        
        if len(transformed['bboxes']) > 0 or len(annotations) == 0:
            final_img = transformed['image']
            final_bboxes = transformed['bboxes']
            final_labels = transformed['text_labels'] # 获取裁剪后剩余的标签
            break
    
    if final_img is None:
        final_img = img_np
        final_bboxes = bboxes
        final_labels = text_labels

    # 构造新标注
    new_annotations = []
    for b, label in zip(final_bboxes, final_labels):
        new_annotations.append({
            "text_content": label, # 这里的 label 就是之前的字符串 "1-39"
            "bbox_2d": [
                max(0, min(1000, int(round(b[0] * 1000)))),
                max(0, min(1000, int(round(b[1] * 1000)))),
                max(0, min(1000, int(round(b[2] * 1000)))),
                max(0, min(1000, int(round(b[3] * 1000))))
            ]
        })

    # 保存图片
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_cut_{output_idx}{ext}"
    output_path = os.path.join(output_image_dir, new_filename)
    Image.fromarray(final_img).save(output_path, quality=95)

    container_img_path = os.path.join("train_images_cut", new_filename)
    return container_img_path, new_annotations

def main():
    random.seed(42)
    input_json = '/root/autodl-tmp/.autodl/data/train_sft_correct.json'
    output_json = '/root/autodl-tmp/.autodl/data/train_sft_cut_correct.json'
    local_output_dir = '/root/autodl-tmp/.autodl/data/train_images_cut'
    
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir, exist_ok=True)

    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    new_dataset = []
    print(f"开始处理，共 {len(dataset)} 条数据...")

    for idx, item in enumerate(dataset):
        try:
            new_path, new_anns = process_image(item, local_output_dir, idx)
            if new_path is None: continue
            new_item = {
                "conversations": [
                    item['conversations'][0],
                    {"from": "gpt", "value": json.dumps(new_anns, ensure_ascii=False)}
                ],
                "images": [new_path]
            }
            new_dataset.append(new_item)
            if (idx + 1) % 50 == 0: print(f"已处理: {idx + 1}/{len(dataset)}")
        except Exception as e:
            print(f"处理第 {idx} 张图出错: {e}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    print(f"处理完成！新数据集已保存至 {output_json}")

if __name__ == '__main__':
    main()
