import cv2
from matplotlib import pyplot as plt 
camera = cv2.VideoCapture(0)
def gen_frames():  
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def recog():
    img = cv2.imread("image.jpg") #camera
    # 

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    #для роспознавания ; для изображения на екран

    stop_data = cv2.CascadeClassifier('stop_data.xml') 
    #файлы чтоб распознавать обекты

    found = stop_data.detectMultiScale(img_gray, minSize =(20, 20)) 
    #настройки распознавания 
    amount_found = len(found) 
    #

    if amount_found != 0: 
        for (x, y, width, height) in found: 

            cv2.rectangle(img_rgb, (x, y),  
                (x + height, y + width),  
                (0, 255, 0), 5) 
    # распознавание       

    plt.subplot(1, 1, 1) 
    plt.imshow(img_rgb) 
    plt.show() 
    # вывод
