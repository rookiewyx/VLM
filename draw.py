# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 预测结果可视化 
# --------------------------------------------


import json

# 导入Pillow库核心模块
from PIL import Image, ImageDraw

# ===================== 1. 配置参数 =====================
# 图片路径
prediction_path = "output/predictions/dpo_test_correct_enable_vit_predictions.json"
info = json.load(open(prediction_path))
show_index = [0, 10, 45, 139, 255]
for index in show_index:
    image_path = info[index]["image_path"]
    pred = info[index]["pred"]

    # 1000×1000归一化的两个bbox坐标
    norm_bboxes = [p["bbox_2d"] for p in pred]

    # 绘制样式：边框颜色 (RGB)、线条宽度
    default_colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255)   # 青色
    ]
    # 线宽
    line_width = 3

    # ===================== 2. 读取图片 + 获取真实尺寸 =====================
    # 打开图片
    img = Image.open("data/" + image_path)

    # 获取图片原始宽度、高度
    img_w, img_h = img.size
    print(f"原始图片尺寸：宽度={img_w}，高度={img_h}")

    # 计算缩放比例（归一化1000 → 原图真实尺寸）
    scale_w = img_w / 1000.0
    scale_h = img_h / 1000.0

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # ===================== 3. 坐标还原 + 绘制边框 + 文字 =====================
    for i, bbox in enumerate(norm_bboxes):
        # 关键：归一化坐标 → 原始图片真实像素坐标
        x1 = int(bbox[0] * scale_w)
        y1 = int(bbox[1] * scale_h)
        x2 = int(bbox[2] * scale_w)
        y2 = int(bbox[3] * scale_h)

        color = default_colors[i % len(default_colors)]
        # Pillow绘制矩形框
        draw.rectangle(
            [x1, y1, x2, y2],  # 坐标 (左上角x,y, 右下角x,y)
            outline=color,  # 边框颜色
            width=line_width  # 边框粗细
        )

        label_text = pred[i]["text_content"]


        # 计算文字位置（框的左上角外侧）
        text_x = x1
        text_y = y1 - 25  # 放在框上方25像素处
        
        # 如果上方空间不够，放在下方
        if text_y < 0:
            text_y = y2 + 5
        
        # 获取文字尺寸（用于绘制背景）
        bbox_text = draw.textbbox((text_x, text_y), label_text)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 绘制文字背景（可选，使文字更清晰）
        draw.rectangle(
            [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
            fill=color,
            outline=color
        )
        
        # 绘制文字（白色）
        draw.text((text_x, text_y), label_text,fill=(255, 255, 255))
    
    
    # ===================== 4. 显示 + 保存结果 =====================
    # 保存标注后的图片
    new_name = image_path.split("/")[-1]
    img.save(f"output/display_images/annotated_{new_name}")
    print(f"✅ 标注完成！结果已保存为 annotated_{new_name}")

