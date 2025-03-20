import cv2
import time
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Критически важный этап: захват фона без мантии
    print("Уберите мантию из кадра на 5 секунд. Фон должен быть статичным")
    time.sleep(5)
    
    bg_frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            bg_frames.append(np.flip(frame, axis=1))
    background = np.median(bg_frames, axis=0).astype(np.uint8)
    
    # Параметры обработки
    hsv_lower = np.array([35, 50, 50])
    hsv_upper = np.array([90, 255, 255])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        img = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Создание маски с учетом соседних пикселей
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Уточнение границ маски
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_refined = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  
                cv2.drawContours(mask_refined, [cnt], -1, 255, -1)
        
        # Смешивание с проверкой обновления
        result = np.where(mask_refined[..., None] == 0, img, background)
        
        cv2.imshow('Result', result)
        if cv2.waitKey(1) == 27: break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
