import cv2
import torch
import time

# YOLOv5モデルの読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5sの使用

# クラス名の定義（COCOデータセットのクラス名）
CLASSES = model.names

# カメラの設定
cap = cv2.VideoCapture(0)
fps = 3
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # 指定したフレームタイム出ないときはスキップ
    if current_time - prev_time < 1./fps:
        continue

    prev_time = current_time

    # YOLOv5による推論
    results = model(frame, size=320)

    # 推論結果の取得
    detections = results.pandas().xyxy[0]  # results.pandas().xyxy[0]はpandas DataFrame

    # 検出されたオブジェクトの描画
    for index, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        if label in ["person", "cell phone"]:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # 特定のオブジェクトが検出された場合のメッセージ表示
            if label == "person":
                cv2.putText(frame, "Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif label == "cell phone":
                cv2.putText(frame, "Cell Phone Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 画像の表示
    cv2.imshow('YOLOv5 Detection', frame)

    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
