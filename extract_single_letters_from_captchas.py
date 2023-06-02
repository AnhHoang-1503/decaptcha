import os
import os.path
import cv2
import glob
import imutils

# Đặt đường dẫn đến thư mục chứa ảnh
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


# Lấy danh sách ảnh đã dán nhãn
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# duyệt từng ảnh
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Lấy tên file - tương ứng với nhãn
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    print(captcha_correct_text)

    # Chuyển thành ảnh thang xám
    image = cv2.imread(captcha_image_file) 
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
    
    letter_image_regions = [] # mảng chứ các vùng chứa chữ cái

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

    # Nếu số chữ cái tìm được không đúng 4 thì bỏ qua, tránh dữ liệu xấu
    if len(letter_image_regions) != 4:
        continue

    # Sắp xếp lại mảnh theo x để chắc chắn là đọc từ trái sang phải
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Duyệt từng chữ cái
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Lấy x, y, w, h
        x, y, w, h = letter_bounding_box

        # Lấy ảnh chữ cái với margin 2px
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Link đến folder chứa chữ cái
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # Nếu chưa tồn tại thì tạo mới
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Lưu vào folder
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # Tăng chỉ mục
        counts[letter_text] = count + 1
