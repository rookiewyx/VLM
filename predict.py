import base64
import json
import os
import re
import time
import requests
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Any

URL = "http://localhost:8000/v1/chat/completions"
# test_path = '/root/autodl-tmp/.autodl/data/test.json'
# test_path = '/root/autodl-tmp/.autodl/data/train_sft.json'
# test_path = '/root/autodl-tmp/.autodl/data/train_sft_cut.json'
# test_path = '/root/autodl-tmp/.autodl/data/train_sft_perspective.json'
# test_path = '/root/autodl-tmp/.autodl/data/train_sft_correct.json'
# test_path = '/root/autodl-tmp/.autodl/data/train_sft_cut_correct.json'
test_path = '/root/autodl-tmp/.autodl/data/test_correct.json'

# output_path = '/root/autodl-tmp/.autodl/output/predictions/sft_train_correct_enable_vit_predictions.json'
# output_path = '/root/autodl-tmp/.autodl/output/predictions/sft_train_cut_correct_enable_vit_predictions.json'
output_path = '/root/autodl-tmp/.autodl/dpo/output/predictions/kto_test_correct_enable_vit_predictions.json'

serve_model_name = "qwen3-2b" 


def get_img_base64(image_path):  
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")

    test_image = f"data:image/{ext};base64,{encoded_image_text}"
    return test_image


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # 说明两个框没有交集
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    # 交点坐标做矩阵面积计算
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def get_bbox_area(bbox: List[int]) -> int:
    """计算边界框面积"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def extract_json_from_response(text: str) -> List[Dict]:
    """从模型响应中提取JSON数组"""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


def main():
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_results = []

    # 聚合统计
    total_images = len(data)
    correct_largest = 0
    error_count = 0

    # 文本识别指标 (全局统计)
    text_correct = 0
    text_total_pred = 0
    text_total_gt = 0

    # 检测定位指标 (全局统计)
    det_correct = 0
    det_total_pred = 0
    det_total_gt = 0
    all_ious = []

    # 端到端指标 (全局统计)
    e2e_correct = 0
    e2e_total_pred = 0
    e2e_total_gt = 0
    exact_match_images = 0

    # 性能指标
    ttft_list = []
    tps_list = []

    for idx, item in enumerate(data):
        image_path = "/root/autodl-tmp/.autodl/data/" + item['images'][0]
        label_str = item["conversations"][1]["value"]

        try:
            gt_items = json.loads(label_str)
        except json.JSONDecodeError:
            gt_items = []

        print(f"Processing {idx + 1}/{len(data)}: {image_path}")

        pred_items = []
        request_start = None
        first_token_time = None
        request_end = None
        tokens_received = 0   # 接收到的token数

        try:
            image = get_img_base64(image_path)
        except Exception as e:
            print(f"  Error loading image: {e}")
            error_count += 1
            all_results.append({
                'image_path': image_path,
                'gt': gt_items,
                'pred': [],
                'error': str(e)
            })
            continue
  
        try:
            request_start = time.time() # 开始时间
            first_token_time = None    # 第一个token返回的时间
            tokens_received = 0       
            full_response = ""

            user_info = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": "你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, {\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"},
                ]
            }

            messages = [user_info]

            params = {
                "model": serve_model_name,
                "stream": True,
                "messages": messages,
                "temperature": 0
            }

            headers = {
                "Content-Type": "application/json"
            }

            chat_response = requests.post(
                url=URL,
                json=params,
                headers=headers,
                timeout=30,
                stream=True
            )

            count = 0
            for line in chat_response.iter_lines():
                if first_token_time is None:
                    first_token_time = time.time()

                line = line.decode('utf-8') if isinstance(line, bytes) else line
                if line.startswith('data: '):
                    count += 1
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    elif count > 9999:
                        break
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_response += content
                                tokens_received += 1
                    except json.JSONDecodeError:
                        pass

            request_end = time.time()
            pred_items = extract_json_from_response(full_response)

        except Exception as e:
            print(f"  Error in inference: {e}")
            error_count += 1
            pred_items = []

        all_results.append({
            'image_path': item['images'][0],
            'gt': gt_items,
            'pred': pred_items
        })

        # 计算性能指标
        if request_start and first_token_time:
            ttft = first_token_time - request_start
            ttft_list.append(ttft)

        if request_start and request_end and tokens_received > 0:
            total_time = request_end - request_start
            tps = tokens_received / total_time if total_time > 0 else 0
            tps_list.append(tps)

        # 1. 主指标：最大框准确率
        gt_texts = {gt.get('text_content', '') for gt in gt_items}

        if len(pred_items) > 0:
            largest_pred = max(pred_items, key=lambda p: get_bbox_area(p.get('bbox_2d', [0, 0, 0, 0])))
            largest_pred_text = largest_pred.get('text_content', '')

            if largest_pred_text in gt_texts:
                correct_largest += 1
        else:
            # 这种情况有可能是负样本
            if len(gt_items) == 0:
                correct_largest += 1

        # 2. 辅助指标 - 全局统计
        gt_texts_list = [gt.get('text_content', '') for gt in gt_items]
        gt_bboxes_list = [gt.get('bbox_2d', [0, 0, 0, 0]) for gt in gt_items]

        pred_texts_list = [pred.get('text_content', '') for pred in pred_items]
        pred_bboxes_list = [pred.get('bbox_2d', [0, 0, 0, 0]) for pred in pred_items]

        # 文本识别指标
        # 对于每个预测文本，检查是否在GT文本列表中
        # 这里可能出现预测是N个框，GT是M个框的情况
        img_text_correct = 0
        used_gt_for_text = set()
        for pred_text in pred_texts_list:
            if pred_text in gt_texts_list:
                img_text_correct += 1

        text_correct += img_text_correct
        text_total_pred += len(pred_items)
        text_total_gt += len(gt_items)

        # 检测定位指标
        img_det_correct = 0
        img_det_correct_list = [False] * len(pred_items)
        for pred_idx, pred_bbox in enumerate(pred_bboxes_list):
            for gt_bbox in gt_bboxes_list:
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou >= 0.5:
                    img_det_correct += 1
                    img_det_correct_list[pred_idx] = True
                    all_ious.append(iou)
                    break

        det_correct += img_det_correct
        det_total_pred += len(pred_items)
        det_total_gt += len(gt_items)

        # 端到端指标 (IoU>=0.5 且 文本精确匹配)
        # 需要为每个预测找到一个GT满足两个条件，且每个GT只能匹配一次
        img_e2e_correct = 0
        matched_gt_indices = set()

        for pred_idx, pred in enumerate(pred_items):
            pred_text = pred.get('text_content', '')
            pred_bbox = pred.get('bbox_2d', [0, 0, 0, 0])

            for gt_idx, gt in enumerate(gt_items):
                if gt_idx in matched_gt_indices:
                    continue
                gt_text = gt.get('text_content', '')
                gt_bbox = gt.get('bbox_2d', [0, 0, 0, 0])

                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou >= 0.5 and pred_text == gt_text:
                    img_e2e_correct += 1
                    matched_gt_indices.add(gt_idx)
                    break

        e2e_correct += img_e2e_correct
        e2e_total_pred += len(pred_items)
        e2e_total_gt += len(gt_items)

        # 整图精确匹配：所有预测和GT都一一匹配，且数量相等
        if len(pred_items) == len(gt_items) and img_e2e_correct == len(gt_items):
            exact_match_images += 1
            

        

    # 保存预测结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved to {output_path}")


    # 计算汇总指标
    print("\n" + "="*60)
    print("评估指标汇总")
    print("="*60)

    # 主指标
    acc_largest = correct_largest / total_images if total_images > 0 else 0
    print(f"\n【主指标】基于最大框的图片级准确率: {acc_largest:.4f} ({correct_largest}/{total_images})")

    # 辅助指标
    print(f"\n【辅助指标】")

    # 文本识别指标
    text_precision = text_correct / text_total_pred if text_total_pred > 0 else 0
    text_recall = text_correct / text_total_gt if text_total_gt > 0 else 0
    text_f1 = 2 * text_precision * text_recall / (text_precision + text_recall) if (text_precision + text_recall) > 0 else 0
    print(f"  文本识别:")
    print(f"    Precision: {text_precision:.4f} ({text_correct}/{text_total_pred})")
    print(f"    Recall: {text_recall:.4f} ({text_correct}/{text_total_gt})")
    print(f"    F1: {text_f1:.4f}")

    # 检测定位指标
    det_precision = det_correct / det_total_pred if det_total_pred > 0 else 0
    det_recall = det_correct / det_total_gt if det_total_gt > 0 else 0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0
    print(f"  检测定位 (IoU≥0.5):")
    print(f"    Precision: {det_precision:.4f} ({det_correct}/{det_total_pred})")
    print(f"    Recall: {det_recall:.4f} ({det_correct}/{det_total_gt})")
    print(f"    F1: {det_f1:.4f}")
    print(f"    平均IoU: {avg_iou:.4f}")

    # 端到端指标
    e2e_precision = e2e_correct / e2e_total_pred if e2e_total_pred > 0 else 0
    e2e_recall = e2e_correct / e2e_total_gt if e2e_total_gt > 0 else 0
    e2e_f1 = 2 * e2e_precision * e2e_recall / (e2e_precision + e2e_recall) if (e2e_precision + e2e_recall) > 0 else 0
    exact_match_rate = exact_match_images / total_images if total_images > 0 else 0
    print(f"  端到端 (文本匹配 + IoU≥0.5):")
    print(f"    Precision: {e2e_precision:.4f} ({e2e_correct}/{e2e_total_pred})")
    print(f"    Recall: {e2e_recall:.4f} ({e2e_correct}/{e2e_total_gt})")
    print(f"    F1: {e2e_f1:.4f}")
    print(f"    整图精确匹配率: {exact_match_rate:.4f} ({exact_match_images}/{total_images})")

    # 性能指标
    print(f"\n【性能指标】")
    if ttft_list:
        print(f"  首Token时间 (TTFT):")
        print(f"    平均: {np.mean(ttft_list)*1000:.2f}ms, 中位数: {np.median(ttft_list)*1000:.2f}ms")
    if tps_list:
        print(f"  生成速度 (TPS):")
        print(f"    平均: {np.mean(tps_list):.2f} token/s, 中位数: {np.median(tps_list):.2f} token/s")

    print(f"\n总处理图片数: {total_images}, 错误数: {error_count}")
    print("="*60)


if __name__ == '__main__':
    main()
