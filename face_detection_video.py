import cv2

# 동영상 파일 경로 설정
video_path = 'People.mp4'  # 분석할 동영상 파일 이름
cap = cv2.VideoCapture(video_path)  # 동영상 로드

# Haar Cascade 파일 로드 (기본 제공)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 동영상 열기 확인
if not cap.isOpened():
    print("Error: 동영상을 열 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# 출력 프레임 크기 설정
output_width = 640
output_height = 360

while cap.isOpened():
    # 동영상에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:  # 더 이상 프레임이 없으면 종료
        break

    # 프레임 크기 조정
    frame = cv2.resize(frame, (output_width, output_height))

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 탐지된 얼굴에 사각형 표시
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 결과를 화면에 출력
    cv2.imshow('Video Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
