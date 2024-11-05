import os

# 文件路徑列表
image_paths = [
    '/home/leo-liao/code/ncku/ncku-cvdl-hw/hw1/Dataset_CvDl_Hw1/Q1_Image/12.bmp',
    '/home/leo-liao/code/ncku/ncku-cvdl-hw/hw1/Dataset_CvDl_Hw1/Q1_Image/11.bmp',
    '/home/leo-liao/code/ncku/ncku-cvdl-hw/hw1/Dataset_CvDl_Hw1/Q1_Image/2.bmp',
    '/home/leo-liao/code/ncku/ncku-cvdl-hw/hw1/Dataset_CvDl_Hw1/Q1_Image/13.bmp'
]

# 提取文件名中的數字並排序
sorted_image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))

# 打印排序後的結果
for path in sorted_image_paths:
    print(path)
