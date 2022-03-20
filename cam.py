import cv2

print("Before URL")
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('rtsp://admin@admin:47.61.166.232:8080')
print("After URL")

while True:

    print('About to start the read command')
    ret, frame = cap.read()
    print('About to show frames.')
    cv2.imshow("Capturing",frame)
    print('Running...')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()