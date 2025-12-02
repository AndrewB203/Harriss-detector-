import cv2
import numpy as np
import os

def simple_fast(image_path, threshold=20):
    """Простой детектор FAST"""
    
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return
    
    # Загрузка
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки!")
        return
    
    # Уменьшение
    h, w = img.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    
    # FAST детектор
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    keypoints = fast.detect(gray, None)
    
    # Результат
    result = img.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
    
    print(f"FAST: найдено {len(keypoints)} точек")
    
    # Показываем
    cv2.imshow(f'FAST (порог={threshold})', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return keypoints

# Автоматический запуск для всех изображений
for file in os.listdir('.'):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Обрабатываю {file}...")
        simple_fast(file, threshold=20)
        break