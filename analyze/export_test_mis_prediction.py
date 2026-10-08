# 文件描述: 导出预测错误case到文件 

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

def build_test_item(image_path, gt_list, pred_list):
    """Build a test dataset item"""
    human_prompt = "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, {\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"

    # GT
    chosen_value = json.dumps(gt_list, ensure_ascii=False)

    # prediction
    rejected_value = json.dumps(pred_list, ensure_ascii=False)

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_prompt
            }
        ],
        "gt": {
            "from": "gpt",
            "value": chosen_value
        },
        "pred": {
            "from": "gpt",
            "value": rejected_value
        },
        "images": [image_path]
    }

def main():
    # Process all three files
    files = [
        # "output/predictions/sft_test_enable_vit_predictions.json",
        #"output/predictions/sft_train_enable_vit_predictions.json",
        #"output/predictions/sft_train_cut_enable_vit_predictions.json",
        #"output/predictions/sft_train_perspective_enable_vit_predictions.json",
        "/root/autodl-tmp/.autodl/output/predictions/sft_test_enable_vit_predictions.json",
    ]

    all_candidates = [] 
    seen_images = set()

    for file in files:
        print(f"Processing {file}...")
        candidates = process_file(file)

        # Avoid duplicate images
        for item in candidates:
            if item["image_path"] not in seen_images:
                seen_images.add(item["image_path"])
                all_candidates.append(item)

    # Build dataset
    test_dataset = []
    for item in all_candidates:
        error_item = build_test_item(item["image_path"], item["gt"], item["pred"])
        test_dataset.append(error_item)

    print(f"找到 {len(test_dataset)} 张图像，其pred中bbox最大的text不在gt中:")
    for item in test_dataset:
        print(item["images"][0])

    # Save to file
    output_file = "/root/autodl-tmp/.autodl/output/predictions/error_test_predict.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(test_dataset)} items to {output_file}")

if __name__ == "__main__":
    main()

