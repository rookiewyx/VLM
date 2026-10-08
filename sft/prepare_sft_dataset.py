
# 文件描述: 准备SFT的数据+转换数据格式



import json
import re
import math


def round_coord(x):
    return int(round(x))

def convert_dataset(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for item in data:
        answer_list = item.get('answer', [])

        # 构建 gpt 的 value
        parking_slots = []
        for ans in answer_list:
            label = ans.get('label')
            if label == '无':
                continue
            bbox = ans.get('bbox')
            if not bbox:
                continue

            # 获取bbox坐标
            xmin = round_coord(bbox['xmin'])
            ymin = round_coord(bbox['ymin'])
            xmax = round_coord(bbox['xmax'])
            ymax = round_coord(bbox['ymax'])

            parking_slots.append({
                "text_content": label,
                "bbox_2d": [xmin, ymin, xmax, ymax]
            })


        # 所有样本都添加到结果中，没有车位号时 value 为 []
        result.append({
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, {\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"
                },
                {
                    "from": "gpt",
                    "value": json.dumps(parking_slots, ensure_ascii=False)
                }
            ],
            "images": [item["new_img"]]
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"转换完成！共处理 {len(result)} 个样本")


base_dir = "/root/autodl-tmp/.autodl/data/"
# 训练集转换
convert_dataset(base_dir + "train_labels_with_bbox.json", base_dir + "train_sft.json")
# 测试集转换
convert_dataset(base_dir + "test_labels_with_bbox.json", base_dir + "test.json")
