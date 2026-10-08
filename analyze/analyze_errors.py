# 文件描述: badcase分析归类工具

import json
from collections import defaultdict, Counter
import difflib


def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area > 0:
        return inter_area / union_area
    else:
        return 0.0


def match_predictions(gt_list, pred_list, iou_threshold=0.1):
    matched = []
    used_gt_indices = set()
    used_pred_indices = set()

    iou_pairs = []
    # 统计所有匹配上的框
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            iou = calculate_iou(gt['bbox_2d'], pred['bbox_2d'])
            if iou >= iou_threshold:
                # 用负号，目的是按照iou从大到小排列
                iou_pairs.append((-iou, i, j, gt, pred))

    iou_pairs.sort()
    # 去重
    for neg_iou, i, j, gt, pred in iou_pairs:
        if i not in used_gt_indices and j not in used_pred_indices:
            matched.append((gt, pred, -neg_iou))
            used_gt_indices.add(i)
            used_pred_indices.add(j)

    unmatched_gts = [gt for i, gt in enumerate(gt_list) if i not in used_gt_indices]
    unmatched_preds = [pred for j, pred in enumerate(pred_list) if j not in used_pred_indices]

    return matched, unmatched_gts, unmatched_preds


def classify_text_error(gt_text, pred_text):
    if gt_text == pred_text:
        return "正确", ""

    diff = list(difflib.ndiff(gt_text, pred_text))

    missing_chars = []  #遗漏
    extra_chars = []    #多出来的
    replaced_chars = [] # 替换的

    i = 0
    while i < len(diff):
        if diff[i].startswith('- '):
            missing_char = diff[i][2]
            if i + 1 < len(diff) and diff[i + 1].startswith('+ '):
                replaced_chars.append((missing_char, diff[i + 1][2]))
                i += 2
            else:
                missing_chars.append(missing_char)
                i += 1
        elif diff[i].startswith('+ '):
            extra_chars.append(diff[i][2])
            i += 1
        else:
            i += 1

    error_types = []
    details = []

    if replaced_chars:
        replace_desc = ", ".join([f"{g}->{p}" for g, p in replaced_chars])
        error_types.append("字符错检")
        details.append(f"字符替换: {replace_desc}")

    if missing_chars:
        error_types.append("内容缺失")
        details.append(f"缺失字符: {''.join(missing_chars)}")

    if extra_chars:
        error_types.append("多余字符")
        details.append(f"多余字符: {''.join(extra_chars)}")

    if not error_types:
        error_types.append("其他错误")

    return "|".join(error_types), "; ".join(details)


def analyze_predictions(data):
    stats = {
        'total_images': 0,
        'total_gt': 0,
        'total_pred': 0,
        'matched': 0,
        'correct_text': 0,
        'error_types': Counter(),
        'image_errors': defaultdict(list),
        'details_by_image': defaultdict(list)
    }

    for item in data:
        image_path = item['image_path']
        image_name = image_path.split('/')[-1]
        gt_list = item['gt']
        pred_list = item['pred']

        stats['total_images'] += 1
        stats['total_gt'] += len(gt_list)
        stats['total_pred'] += len(pred_list)

        matched, unmatched_gts, unmatched_preds = match_predictions(gt_list, pred_list)
        stats['matched'] += len(matched)

        for gt, pred, iou in matched:
            gt_text = gt['text_content']
            pred_text = pred['text_content']

            if gt_text == pred_text:
                stats['correct_text'] += 1
            else:
                # 分析错误类型
                error_type, details = classify_text_error(gt_text, pred_text)
                stats['error_types'][error_type] += 1
                stats['image_errors'][image_name].append({
                    'type': 'text_error',
                    'gt': gt_text,
                    'pred': pred_text,
                    'error_type': error_type,
                    'details': details,
                    'iou': iou,
                    'gt_bbox': gt['bbox_2d'],
                    'pred_bbox': pred['bbox_2d']
                })
                stats['details_by_image'][image_name].append(
                    f"文本错误: GT='{gt_text}', PRED='{pred_text}' ({error_type}: {details})"
                )

        for gt in unmatched_gts:
            stats['error_types']['漏检'] += 1
            stats['image_errors'][image_name].append({
                'type': 'missing',
                'gt': gt['text_content'],
                'gt_bbox': gt['bbox_2d']
            })
            stats['details_by_image'][image_name].append(
                f"漏检: GT='{gt['text_content']}' at {gt['bbox_2d']}"
            )

        for pred in unmatched_preds:
            stats['error_types']['无中生有'] += 1
            stats['image_errors'][image_name].append({
                'type': 'hallucination',
                'pred': pred['text_content'],
                'pred_bbox': pred['bbox_2d']
            })
            stats['details_by_image'][image_name].append(
                f"无中生有: PRED='{pred['text_content']}' at {pred['bbox_2d']}"
            )

    return stats


def print_summary(stats):
    print("=" * 80)
    print("车位号识别错误分析报告")
    print("=" * 80)
    print(f"总图片数: {stats['total_images']}")
    print(f"总GT标注数: {stats['total_gt']}")
    print(f"总预测数: {stats['total_pred']}")
    print(f"匹配成功数: {stats['matched']}")
    print(f"文本完全正确数: {stats['correct_text']}")
    print()

    print("-" * 80)
    print("错误类型统计:")
    print("-" * 80)
    total_errors = sum(stats['error_types'].values())
    for error_type, count in sorted(stats['error_types'].items(), key=lambda x: -x[1]):
        percentage = (count / total_errors * 100) if total_errors > 0 else 0
        print(f"  {error_type:20s}: {count:5d} ({percentage:5.1f}%)")
    print(f"  {'总计':20s}: {total_errors:5d}")
    print()

    print("-" * 80)
    print("按图片统计的错误详情（前20张有错误的图片）:")
    print("-" * 80)
    error_images = sorted(stats['details_by_image'].items(), key=lambda x: -len(x[1]))[:20]
    for image_name, errors in error_images:
        print(f"\n{image_name}:")
        for error in errors:
            print(f"  - {error}")


def save_detailed_report(stats, output_file):
    report = {
        'summary': {
            'total_images': stats['total_images'],
            'total_gt': stats['total_gt'],
            'total_pred': stats['total_pred'],
            'matched': stats['matched'],
            'correct_text': stats['correct_text'],
        },
        'error_type_counts': dict(stats['error_types']),
        'errors_by_image': dict(stats['image_errors'])
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n详细报告已保存到: {output_file}")


def main():

    input_file = '/root/autodl-tmp/.autodl/output/predictions/sft_test_correct_enable_vit_predictions.json'
    output_file = '/root/autodl-tmp/.autodl/output/analysis/sft_test_predictions_errors.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = analyze_predictions(data)
    print_summary(stats)
    save_detailed_report(stats, output_file)


if __name__ == '__main__':
    main()

