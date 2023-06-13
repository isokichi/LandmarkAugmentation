import cv2
import numpy as np
import random

image_dir_path = 'dataset'
annotation_dir_path = 'dataset/annotation'
image_name = 'dogface20211028-2'

input_image_path = f'{image_dir_path}/{image_name}.jpg'
input_annotation_path = f'{annotation_dir_path}/{image_name}.txt'

import cv2
import numpy as np
import random

def shift_image(input_image_path, input_annotation_path, image_name, max_shift_ratio=0.5):
    # 画像の読み込み
    image = cv2.imread(input_image_path)
    
    # 座標データの読み込み
    with open(input_annotation_path, 'r') as file:
        lines = file.readlines()

    # 座標データをパースして点の座標リストを作成
    points = []
    for line in lines[1:]:  # 最初の行は無視する
        x, y = map(float, line.strip().split(','))
        points.append((x, y))
    
    # 画像サイズを取得
    image_height, image_width, _ = image.shape
    
    # シフト処理のループ
    while True:
        # 最大シフト量を計算
        max_shift_x = int(image_width * max_shift_ratio)
        max_shift_y = int(image_height * max_shift_ratio)
        
        # ランダムなシフト量を生成
        shift_x = random.randint(-max_shift_x, max_shift_x)
        shift_y = random.randint(-max_shift_y, max_shift_y)
        
        # 特徴点をシフトさせる
        shifted_points = []
        for point in points:
            shifted_points.append((point[0] + shift_x, point[1] + shift_y))
        
        # シフト後の特徴点が画像の範囲内に収まっているかチェック
        # for point in shifted_pointsの中で全ての点が画像内に収まっていることを確認。all関数で一つでもfalseがあれば条件を満たさない。
        in_range = all(0 <= point[0] < image_width and 0 <= point[1] < image_height for point in shifted_points)
        if in_range:
            break  # すべての特徴点が範囲内に収まっていればループを終了
        
        # 範囲外の特徴点がある場合、max_shift_ratioを下げて再度シフトする
        max_shift_ratio *= 0.9  # max_shift_ratioを10%減少させる
    
    # 画像のシフト
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (image_width, image_height))
    
    # シフト後の画像の境界条件を処理
    shifted_image = fix_boundary_shift(shifted_image, shift_x, shift_y)
    
    # シフト後の画像と特徴点を保存
    output_image_path = f'{image_dir_path}/shifted_{image_name}.jpg'
    output_annotation_path = f'{annotation_dir_path}/shifted_{image_name}.txt'

    cv2.imwrite(output_image_path, shifted_image)
    
    with open(output_annotation_path, 'w') as file:
       # 最初の行に画像ファイル名を書き込む
        file.write(f"shifted_{image_name}\n")
        # シフト後の特徴点の座標を書き込む
        for point in shifted_points:
            file.write(f"{point[0]},{point[1]}\n")


def fix_boundary_shift(image, shift_x, shift_y):
    # シフト後の画像の境界条件を処理する
        
    # シフト量が正の場合は左端または上端からのピクセルを埋める
    if shift_x > 0:
        image[:, :abs(shift_x)] = image[:, abs(shift_x)].reshape(-1, 1, 3)
    if shift_y > 0:
        image[:abs(shift_y), :] = image[abs(shift_y), :].reshape(1, -1, 3)
    
    # シフト量が負の場合は右端または下端からのピクセルを埋める
    if shift_x < 0:
        image[:, -abs(shift_x):] = image[:, -abs(shift_x)-1].reshape(-1, 1, 3)
    if shift_y < 0:
        image[-abs(shift_y):, :] = image[-abs(shift_y)-1, :].reshape(1, -1, 3)
    
    return image


def rotate_image(input_image_path, input_annotation_path, image_name, angle):
    # 画像の読み込み
    image = cv2.imread(input_image_path)
    
    # 座標データの読み込み
    with open(input_annotation_path, 'r') as file:
        lines = file.readlines()

    # 座標データをパースして点の座標リストを作成
    points = []
    for line in lines[1:]:
        x, y = map(float, line.strip().split(','))
        points.append((int(x), int(y)))
    
    # 画像の回転
    image_height, image_width, _ = image.shape
    center = (image_width // 2, image_height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image_width, image_height))

    # 回転後の画像を補完
    mask = np.all(rotated_image == 0, axis=2).astype(np.uint8)  # 3チャンネルの画像を2値のマスク画像に変換
    expanded_mask = expand_mask(mask, kernel_size=3, iterations=1)

    rotated_image = cv2.inpaint(rotated_image, expanded_mask, 5, cv2.INPAINT_NS)


    # 回転後の特徴点の座標を計算
    rotated_points = []
    for point in points:
        x, y = point
        new_x = ((x - center[0]) * np.cos(np.radians(-angle)) - (y - center[1]) * np.sin(np.radians(-angle)) + center[0])
        new_y = ((x - center[0]) * np.sin(np.radians(-angle)) + (y - center[1]) * np.cos(np.radians(-angle)) + center[1])
        rotated_points.append((new_x, new_y))
    
    # 回転後の画像と特徴点を保存
    output_image_path = f'{image_dir_path}/rotated_{angle}_{image_name}.jpg'
    output_annotation_path = f'{annotation_dir_path}/rotated_{angle}_{image_name}.txt'

    # はみ出る場合は保存しない
    # シフト後の特徴点が画像の範囲内に収まっているかチェック
    # for point in shifted_pointsの中で全ての点が画像内に収まっていることを確認。all関数で一つでもfalseがあれば条件を満たさない。
    in_range = all(0 <= point[0] < image_width and 0 <= point[1] < image_height for point in rotated_points)
    if in_range:

        cv2.imwrite(output_image_path, rotated_image)

        with open(output_annotation_path, 'w') as file:
        # 最初の行に画像ファイル名を書き込む
            file.write(f"rotated_{angle}_{image_name}\n")
            # シフト後の特徴点の座標を書き込む
            for point in rotated_points:
                file.write(f"{point[0]},{point[1]}\n")
    else:
        print(f"{output_image_path}は特徴点がはみ出したため保存しませんでした。")

def expand_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return expanded_mask


def shear_image(input_image_path, input_annotation_path, image_name, max_shear_range=0.25):
    # 画像の読み込み
    image = cv2.imread(input_image_path)
    
    # 座標データの読み込み
    with open(input_annotation_path, 'r') as file:
        lines = file.readlines()

    # 座標データをパースして点の座標リストを作成
    points = []
    for line in lines[1:]:
        x, y = map(float, line.strip().split(','))
        points.append((int(x), int(y)))
    
    # 画像のサイズを取得
    image_height, image_width, _ = image.shape
    
    # シフト処理のループ
    while True:
        
        # せん断パラメータのランダムな値を生成
        shear_factor_x = random.uniform(-max_shear_range, max_shear_range)
        shear_factor_y = random.uniform(-max_shear_range, max_shear_range)
        
        
        # せん断変換後の特徴点の座標を計算
        sheared_points = []
        for point in points:
            x, y = point
            new_x = x + shear_factor_x * y
            new_y = y + shear_factor_y * x
            sheared_points.append((new_x, new_y))
        
        # 特徴点が画像の範囲内に収まっているかチェック
        # for point in shifted_pointsの中で全ての点が画像内に収まっていることを確認。all関数で一つでもfalseがあれば条件を満たさない。
        in_range = all(0 <= point[0] < image_width and 0 <= point[1] < image_height for point in sheared_points)
        if in_range:
            break  # すべての特徴点が範囲内に収まっていればループを終了
        
        # 範囲外の特徴点がある場合、max_shear_rangeを下げて再度シフトする
        max_shear_range *= 0.9  # max_shear_rangeを10%減少させる
        
    # せん断行列を作成
    M = np.array([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]], dtype=np.float32)
    
    # 画像のせん断変換
    sheared_image = cv2.warpAffine(image, M, (image_width, image_height))

    # せん断の画像を補完
    mask = np.all(sheared_image == 0, axis=2).astype(np.uint8)  # 3チャンネルの画像を2値のマスク画像に変換
    expanded_mask = expand_mask(mask, kernel_size=3, iterations=1)

    sheared_image = cv2.inpaint(sheared_image, expanded_mask, 5, cv2.INPAINT_NS)
    
    # せん断変換後の画像と特徴点を保存
    output_image_path = f'{image_dir_path}/sheared_{image_name}.jpg'
    output_annotation_path = f'{annotation_dir_path}/sheared_{image_name}.txt'

    cv2.imwrite(output_image_path, sheared_image)

    with open(output_annotation_path, 'w') as file:
        # 最初の行に画像ファイル名を書き込む
        file.write(f"sheared_{image_name}\n")
        # せん断変換後の特徴点の座標を書き込む
        for point in sheared_points:
            file.write(f"{point[0]},{point[1]}\n")



shift_image(input_image_path, input_annotation_path, image_name, max_shift_ratio=0.5)
rotate_image(input_image_path, input_annotation_path, image_name, angle=30)
rotate_image(input_image_path, input_annotation_path, image_name, angle=330)
shear_image(input_image_path, input_annotation_path, image_name, max_shear_range=0.25)