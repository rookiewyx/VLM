# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 混合一定比例的普通样本 
# --------------------------------------------



import json
import random

def calculate_bbox_area(bbox):
    """Calculate bbox area: (x2-x1) * (y2-y1)"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def find_largest_bbox_text(items):
    """Find the text_content with largest bbox area"""
    if not items:
        return None
    largest = max(items, key=lambda x: calculate_bbox_area(x["bbox_2d"]))
    return largest["text_content"]

def is_in_criteria(item):
    """Check if item meets the original criteria (largest pred text not in gt)"""
    gt_list = item["gt"]
    pred_list = item["pred"]

    if not pred_list:
        return False

    largest_pred_text = find_largest_bbox_text(pred_list)
    gt_texts = {gt["text_content"] for gt in gt_list}

    return largest_pred_text not in gt_texts

def build_dpo_item(image_path, gt_list, pred_list):
    """Build a DPO dataset item"""
    human_prompt = "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, {\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"

    chosen_value = json.dumps(gt_list, ensure_ascii=False)
    rejected_value = json.dumps(pred_list, ensure_ascii=False)

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_prompt
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen_value
        },
        "rejected": {
            "from": "gpt",
            "value": rejected_value
        },
        "images": ["/root/autodl-tmp/smart_car_searching/data/" + image_path]
    }

def main():
    # Load existing dataset to get already used images
    with open("/root/autodl-tmp/.autodl/data/train_dpo.json", 'r', encoding='utf-8') as f:
        existing_dataset = json.load(f)

    used_images = {item["images"][0] for item in existing_dataset}
    print(f"Existing dataset has {len(existing_dataset)} items, {len(used_images)} unique images")

    # Process all three files to find remaining candidates
    files = [
        "/root/autodl-tmp/.autodl/output/predictions/sft_train_correct_enable_vit_predictions.json",
        "/root/autodl-tmp/.autodl/output/predictions/sft_train_cut_correct_enable_vit_predictions.json",
        "/root/autodl-tmp/.autodl/output/predictions/sft_train_perspective_correct_enable_vit_predictions.json"
    ]

    remaining_candidates = []
    seen_in_remaining = set()

    for idx, file in enumerate(files):
        print(f"Processing {file}...")
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            image_path = item["image_path"]

            # Skip if already in existing dataset
            if image_path in used_images:
                continue

            # Skip if already collected
            if image_path in seen_in_remaining:
                continue

            # 防止漏检
            if idx > 0 and len(item["gt"]) >= len(item["pred"]):
                continue

            # Only take items that DON'T meet original criteria (to complement)
            if not is_in_criteria(item):
                remaining_candidates.append(item)
                seen_in_remaining.add(image_path)

    print(f"Found {len(remaining_candidates)} remaining candidates")

    # Randomly select 1000 or all available
    random.seed(42)
    # 为什么是500？这里也是调优过的 
    num_to_select = min(500, len(remaining_candidates))
    selected = random.sample(remaining_candidates, num_to_select)
    print(f"Selected {num_to_select} new items")

    # Build DPO items for selected
    new_items = []
    for item in selected:
        # 排除gt和pred是空的情况
        if len(item["gt"]) == 0 and len(item["pred"]) == 0:
            continue
        dpo_item = build_dpo_item(item["image_path"], item["gt"], item["pred"])
        new_items.append(dpo_item)

    # Combine and save
    final_dataset = existing_dataset + new_items
    random.seed(42)
    random.shuffle(final_dataset)

    with open("/root/autodl-tmp/.autodl/data/train_dpo.json", 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"Final dataset has {len(final_dataset)} items, saved to train_dpo.json")

if __name__ == "__main__":
    main()
