from joblib import dump
from joblib import load
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import os
import os.path
import cv2
import glob
import imutils

filename = 'random_forest_model_100.joblib'
model = load(filename)

while True:
    n = input("file: ")

    CAPTCHA = 'unlabeled_generated_captcha_images/(' + n + ').png'
    RESOLVE_STEP = 'resolve_step'

    result = ''

    # Chuyển thành ảnh thang xám
    image = cv2.imread(CAPTCHA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thêm padding 8px
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # Threshold image (chuyển thành ảnh đen trắng background đen, chữ trắng)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Tìm đường viền của những khu vực trắng
    drawContours = thresh.copy()
    contours, hierarchy = cv2.findContours(
        drawContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []  # mảng chứ các vùng chứa chữ cái

    # Duyệt từng contour
    for contour in contours:
        # Lấy thông số hình chữ nhật chứa đường viền
        (x, y, w, h) = cv2.boundingRect(contour)

        # So sánh dài và rộng để kiểm tra xem đã tách đúng chưa
        if w / h > 1.25:
            # Contour quá dài, chứa 2 chữ cái, cắt đôi ra
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # Contour thỏa mãn, chứa 1 chữ cái
            letter_image_regions.append((x, y, w, h))

    # # Nếu số chữ cái tìm được không đúng 4 thì bỏ qua, tránh dữ liệu xấu
    # if len(letter_image_regions) != 4:
    #     continue

    # Sắp xếp lại mảnh theo x để chắc chắn là đọc từ trái sang phải
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Duyệt từng chữ cái
    count = 1
    for letter_bounding_box in letter_image_regions:
        # Lấy x, y, w, h
        x, y, w, h = letter_bounding_box

        # Lấy ảnh chữ cái với margin 2px
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Link đến folder chứa chữ cái
        save_path = os.path.join(RESOLVE_STEP)

        # Lưu vào folder
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # Tăng chỉ mục
        count += 1

    for image_name in os.listdir(RESOLVE_STEP):
        # # Trích xuất đặc trưng của ảnh
        # image = Image.open(image_path)
        # image_data = list(image.getdata())
        # data.append(image_data)
        # Trích xuất đặc trưng HOG của ảnh
        # Chuyển đổi ảnh sang định dạng grayscale
        image_path = os.path.join(
            RESOLVE_STEP, image_name)
        image = Image.open(image_path).convert('L')
        image = image.resize((32, 32))  # Đặt kích thước ảnh cố định
        image_data = hog(image, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2))
        prediction = model.predict([image_data])
        print("File:", image_name)
        print("Prediction:", prediction)
        result += prediction[0]

    print(result)
    # Sử dụng mô hình để giải mã captcha mới
    # new_captcha_path = '000185.png'
    # new_captcha = Image.open(new_captcha_path).convert('L')
    # new_captcha = new_captcha.resize((32, 32))
    # new_captcha_data = hog(new_captcha, orientations=9, pixels_per_cell=(
    #     8, 8), cells_per_block=(2, 2))
    # # new_captcha_data = list(new_captcha.getdata())
    # prediction = model.predict([new_captcha_data])
    # print("Prediction:", prediction)
