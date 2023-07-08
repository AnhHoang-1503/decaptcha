from joblib import dump
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn.model_selection import train_test_split

DATASET = 'extracted_letter_images'

data = []
target = []

# Duyệt qua các folder và file ảnh
for folder_name in os.listdir(DATASET):
    if os.path.isdir(os.path.join(DATASET, folder_name)):
        for image_name in os.listdir(os.path.join(DATASET, folder_name)):
            image_path = os.path.join(
                DATASET, folder_name, image_name)

            # # Trích xuất đặc trưng của ảnh
            # image = Image.open(image_path)
            # image_data = list(image.getdata())
            # data.append(image_data)
            # Trích xuất đặc trưng HOG của ảnh
            # Chuyển đổi ảnh sang định dạng grayscale
            image = Image.open(image_path).convert('L')
            image = image.resize((32, 32))  # Đặt kích thước ảnh cố định
            image_data = hog(image, orientations=9, pixels_per_cell=(
                8, 8), cells_per_block=(2, 2))
            data.append(image_data)

            # Trích xuất nhãn của ảnh
            target.append(folder_name)
            print(folder_name)

# Chia dữ liệu thành 2 phần train và test
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# # Khởi tạo model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# Lưu mô hình
filename = 'random_forest_model.joblib'
dump(model, filename)

# # Sử dụng mô hình để giải mã captcha mới
# new_captcha_path = '000001.png'
# new_captcha = Image.open(new_captcha_path).convert('L')
# new_captcha = new_captcha.resize((32, 32))
# new_captcha_data = hog(new_captcha, orientations=9, pixels_per_cell=(
#     8, 8), cells_per_block=(2, 2))
# # new_captcha_data = list(new_captcha.getdata())
# prediction = model.predict([new_captcha_data])
# print("Prediction:", prediction)
