import cv2
import os

chessboard_size = (11, 8)

# 定義細化角點的參數
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 設定圖片資料夾
image_folder = "./Dataset_CvDl_Hw1/Q1_Image"

# 獲取圖片清單
image_files = [f for f in os.listdir(image_folder) if f.endswith(".bmp")]
image_files.sort()  # 確保圖片按照名稱順序處理

# 遍歷每張圖片，找到並繪製角點
for filename in image_files:
    # 讀取圖片並轉為灰度
    img_path = os.path.join(image_folder, filename)
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 找到棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    
    if ret:
        # 增加角點準確度
        corners = cv2.cornerSubPix(gray_image, corners, winSize, zeroZone, criteria)
        
        # 在圖片上繪製角點
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        
        # 顯示結果
        cv2.imshow(f"Corners in {filename}", image)
        cv2.waitKey(5000)  # 顯示500毫秒，之後自動關閉
        
        # 選擇性地保存結果
        output_path = os.path.join("output", f"corners_{filename}")
        cv2.imwrite(output_path, image)
    else:
        print(f"Chessboard corners not found in {filename}")

# 完成後關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
