#  Copyright (c) 2019. HaoHoangTran
#  Last modified 6/20/19, 3:07 PM
#  Developed by Hao Hoang Tran.

import face_recognition
import cv2
import numpy as np
import glob
import os
import logging, shutil

IMAGES_TRAINING_PATH = './train'  # anh train dir
IMAGES_TEST_PATH = './test'  # ảnh test dir
IMAGES_OUTPUT_PATH = './outputs'  # ảnh kết quả dir
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6  # độ nghiêm ngặt

# tao thu muc output neu chua co
if not os.path.exists(IMAGES_OUTPUT_PATH):
    os.makedirs(IMAGES_OUTPUT_PATH)


# Xóa thư mục output


def removeOutputFolder():
    for the_file in os.listdir(IMAGES_OUTPUT_PATH):
        file_path = os.path.join(IMAGES_OUTPUT_PATH, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            return False

    return True


def get_face_embeddings_from_image(image, convert_to_rgb=False):
    # Chuyển ảnh BGR sang RGB
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # Tìm location face từ ảnh
    face_locations = face_recognition.face_locations(image)

    # chạy mô hình face_recognition để lấy thông số
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings


def paint_detected_face_on_image(img, location, name=None, distances=None):
    # lấy tọa độ từ location
    top, right, bottom, left = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # màu đỏ nếu k nhận ra
    else:
        color = (0, 255, 0)  # xanh với mặt nhận ra

    # Vẽ khung quanh mặt
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)

    # Viết tên dưới ảnh
    cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


def setup_database():
    # database key chính là tên ảnh
    database = {}
    lengthDir = 0
    err = 0
    # chỉ đọc jpg, file khác bỏ
    for filename in glob.glob(os.path.join(IMAGES_TRAINING_PATH, '*.jpg')):
        lengthDir += 1
        # load ảnh
        image_rgb = face_recognition.load_image_file(filename)

        # Dùng tên File để đặt tên nhận dạng
        identity = os.path.splitext(os.path.basename(filename))[0]

        # lấy thông số mặt rồi liên kết với danh tính bên trên
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        if locations.__len__() == 0:
            print("Không nhận ra khuôn mặt từ ảnh bỏ qua " + filename)
            err += 1
            continue
        print("Train thành công ", filename)
        database[identity] = encodings[0]
    print("Tổng số: ", lengthDir, " ảnh, số lỗi: ", err, "ảnh , thành công: ", lengthDir - err, " ảnh")
    print("Danh sách ảnh trong database: ", database.keys())
    return database


# nhận diện người từ ảnh
def run_face_recognitionFromImage(filename, database, showImg):
    img = face_recognition.load_image_file(filename)
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(filename, window_width, window_height)

    face_locations, face_encodings = get_face_embeddings_from_image(img, convert_to_rgb=True)

    # trong ảnh có nhiều mặt, thử xem có mặt nào nhận diện dc k
    for location, face_encoding in zip(face_locations, face_encodings):

        distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if np.any(distances <= MAX_DISTANCE):
            best_match_idx = np.argmin(distances)
            name = known_face_names[best_match_idx]
        else:
            name = None

        # vẽ nhận dạng vào ảnh
        paint_detected_face_on_image(img, location, name, distances)

    name = os.path.splitext(os.path.basename(filename))[0] + '.jpg'
    cv2.imwrite(os.path.join(IMAGES_OUTPUT_PATH, name), img);
    if showImg:
        cv2.imshow(filename, img)
    else:
        print("Lưu ", name)


# nhận diện realtime bằng camera
def run_face_recognition(database):
    # mở handle camera
    video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)

    # lấy thông số database
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())

    while video_capture.isOpened():
        # lấy 1 ảnh duy nhất
        ok, frame = video_capture.read()
        if not ok:
            logging.error("Không thể lấy frame từ capture camera, xem lại camera của bạn có đang bị sử dụng")
            break

        # Chạy nhận dạng
        face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)

        # Loop all face trong ảnh vừa capture
        for location, face_encoding in zip(face_locations, face_encodings):

            # lấy độ chính xác
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Chọn người giống nhất với khuôn mặt
            if np.any(distances <= MAX_DISTANCE):
                best_match_idx = np.argmin(distances)
                name = known_face_names[best_match_idx]
            else:
                name = None

            # vẽ nhận dạng vào ảnh
            paint_detected_face_on_image(frame, location, name, distances)

        # hiện kêt quả
        cv2.imshow('Video', frame)

        # ấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


# chạy thử
removeOutputFolder();
database = setup_database()
showImg = False  # False nếu quá nhiều anh test, vì nó sẽ show rất nhiều cửa sổ kết quả, 1 ảnh => 1 cửa sổ

# chạy test ở thư mục
for filename in glob.glob(os.path.join(IMAGES_TEST_PATH, '*.jpg')):
    run_face_recognitionFromImage(filename, database, showImg)

# bỏ comment dòng  dưới nếu thử với camera
# run_face_recognition(database)

# show ảnh thì đợi cho xem
if (showImg):
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Thành công. Kiểm tra kết quả ở thư mục output")
