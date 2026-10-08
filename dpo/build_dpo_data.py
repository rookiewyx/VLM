# 文件描述: 挖掘难样本构建DPO数据集

import json

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

def process_file(file_path):
    """Process a prediction file and find matching images"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in data:
        image_path = item["image_path"]
        gt_list = item["gt"]
        pred_list = item["pred"]

        if not pred_list:
            continue

        # Get largest bbox text from pred
        largest_pred_text = find_largest_bbox_text(pred_list)

        # Get all gt texts
        gt_texts = {gt["text_content"] for gt in gt_list}

        # Check if largest pred text is NOT in gt texts
        if largest_pred_text not in gt_texts:
            results.append({
                "image_path": image_path,
                "gt": gt_list,
                "pred": pred_list
            })

    return results

def build_dpo_item(image_path, gt_list, pred_list):
    """Build a DPO dataset item"""
    human_prompt = "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, {\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"

    # chosen is GT
    chosen_value = json.dumps(gt_list, ensure_ascii=False)
    
    # rejected is prediction
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
        "images": [image_path]
    }

def main():
    # Process all three files
    files = [
        "/root/autodl-tmp/.autodl/output/predictions/sft_train_correct_enable_vit_predictions.json",
        # "/root/autodl-tmp/.autodl/output/predictions/sft_train_cut_correct_enable_vit_predictions.json",
        "/root/autodl-tmp/.autodl/output/predictions/sft_train_perspective_correct_enable_vit_predictions.json"
    ]

    all_candidates = []
    seen_images = set()

    for idx, file in enumerate(files):
        print(f"Processing {file}...")
        candidates = process_file(file)
        print(f"  Found {len(candidates)} candidates")

        # Avoid duplicate images
        for item in candidates:
            if idx > 0 and len(item["gt"]) >= len(item["pred"]):
                continue
            if item["image_path"] not in seen_images:
                seen_images.add(item["image_path"])
                all_candidates.append(item)

    print(f"Total unique candidates: {len(all_candidates)}")

    # Build DPO dataset
    dpo_dataset = []
    for item in all_candidates:
        dpo_item = build_dpo_item(item["image_path"], item["gt"], item["pred"])
        dpo_dataset.append(dpo_item)

    # Save to file
    output_file = "/root/autodl-tmp/.autodl/data/train_dpo.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dpo_dataset)} items to {output_file}")

if __name__ == "__main__":
    main()

