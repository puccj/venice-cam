from ultralytics import YOLO 

model = YOLO('yolov8x')

result = model.track('input/PIAZZAS.MARCO(TORREOROLOGIO)-Vcam-01_2019-11-01_14h30min00s109ms.avi', conf=0.2, save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)