from ultralytics import YOLO
m = YOLO('best.pt')
m.export(format='onnx')