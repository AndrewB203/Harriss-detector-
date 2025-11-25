import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def simple_harris_demo(image_path):  
    # Проверяем файл
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return
    
    print(f"Файл найден: {image_path}")
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(" Ошибка загрузки изображения!")
        return
    
    print(f"Размер изображения: {image.shape}")
    
    # Уменьшаем для скорости
    h, w = image.shape[:2]
    if max(h, w) > 600:
        scale = 600 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"📏 Уменьшено до: {new_w}x{new_h}")
    
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    
   
    
    # 1. Градиент по X
    Ix = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    
    # 2. Градиент по Y  
    Iy = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    
    
    # 3. Отклик Харриса
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # Гауссово размытие
    Sx2 = cv2.GaussianBlur(Ix2, (3, 3), 1.5)
    Sy2 = cv2.GaussianBlur(Iy2, (3, 3), 1.5)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.5)
    
    # Матрица и отклик Харриса
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    R = det_M - 0.04 * (trace_M ** 2)
    
    # Нормализуем для отображения
    R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
    
    plt.figure(figsize=(15, 5))
    
    # Градиент по X
    plt.subplot(1, 3, 1)
    plt.imshow(Ix, cmap='seismic')
    plt.title('Градиент по X (Ix)', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.axis('off')
    
    # Градиент по Y
    plt.subplot(1, 3, 2)
    plt.imshow(Iy, cmap='seismic')
    plt.title('Градиент по Y (Iy)', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.axis('off')
    
    # Отклик Харриса
    plt.subplot(1, 3, 3)
    plt.imshow(R_norm, cmap='hot')
    plt.title('Отклик Харриса (R)', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "photo_2025-11-25_08-59-04.jpg"
    simple_harris_demo(image_path)