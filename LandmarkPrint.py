import cv2
import numpy as np
import glob
import sys

# 画像ファイルと座標データのパスを指定
image_folder = 'dataset'
annotation_folder = 'dataset/annotation'

# 画像ファイルと座標データの一覧を取得
annotation_files = sorted(glob.glob(f"{annotation_folder}/*.txt"))


def main():

    current_index = 0  # 現在の画像のインデックス

    # 引数からファイル名を取得
    if len(sys.argv) > 1:
        specified_file = sys.argv[1]
        try:
            current_index = annotation_files.index(f"{annotation_folder}/{specified_file}")
        except ValueError:
            print(f"指定されたファイル '{specified_file}' は存在しません。デフォルトのファイルを開きます。")
    
    # 画像の表示とキーボードの入力を待機するループ
    while True:
        annotation_file = annotation_files[current_index]

        # 画像を描画して表示
        image_with_points = draw_points_on_image(annotation_file)
        display_image(image_with_points)

        # キーボードの入力を待機
        key = cv2.waitKey(0)
        # キーに応じて画像を切り替える
        if key == 27:  # ESCキーが押された場合は終了
            break
        elif key == ord('a') or key == ord('A') or key == 2:  # Aキーか左キーで左に移動
            current_index = max(0, current_index - 1)
        elif key == ord('d') or key == ord('D') or key == 3:  # Dキーか右キーで右に移動
            current_index = min(len(annotation_files) - 1, current_index + 1)

    cv2.destroyAllWindows()

def draw_points_on_image(annotation_path):
    # テキストファイルから座標データを読み込む
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # 画像の読み込み
    image_name = lines[0].strip() #改行は削除
    image = cv2.imread(f"{image_folder}/{image_name}.jpg")

    # 座標データをパースして点の座標リストを作成
    points = []
    for line in lines[1:]:  # 最初の行は無視する
        x, y = map(float, line.strip().split(','))
        points.append((int(x), int(y)))

    # 各点に対して円を描画
    for point in points:
        cv2.circle(image, point, 1, (0, 0, 255), -1)  # 円の半径1、色は赤、厚さ-1 (内部を塗りつぶす)

    # 画像の名前を表示
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # 白色
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(image_name, font, font_scale, thickness)
    text_org = (image.shape[1] - text_width - 10, text_height + 10)
    cv2.putText(image, image_name, text_org, font, font_scale, color, thickness, cv2.LINE_AA)

    return image

def display_image(image):
    cv2.imshow('Image with Points', image)


if __name__ == "__main__":
    main()
