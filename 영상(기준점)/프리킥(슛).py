import cv2
import numpy as np
import os
import sys
import json

# =========================================================
# 👇 [설정] 파일명 확인하세요
# =========================================================
VIDEO_FILE = 'directfreekick_raw.mov'
SAVE_FILE = 'directfreekick_ai.mp4'
ANALYSIS_ZONE = 'Center' 

FADE_SPEED = 0.05
# =========================================================

# 1. JSON 데이터 로드
json_path = 'freekick_stats.json'
if not os.path.exists(json_path):
    print("❌ JSON 파일이 없습니다. 1단계 코드를 먼저 실행하세요!")
    sys.exit()

with open(json_path, 'r', encoding='utf-8') as f:
    STATS_DATA = json.load(f)

CURRENT_STATS = STATS_DATA.get(ANALYSIS_ZONE, {})

if ANALYSIS_ZONE == 'Center':
    CLICK_LABELS = ['1.BALL', '2.LEFT Side', '3.CENTER', '4.RIGHT Side']
    TARGET_KEYS = ['Left Side', 'Center', 'Right Side']
else:
    CLICK_LABELS = ['1.BALL', '2.NEAR Post', '3.CENTER', '4.FAR Post']
    TARGET_KEYS = ['Near Post', 'Center', 'Far Post']

points = [] 

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            img = param.copy()
            curr_idx = len(points) - 1
            color = (0, 255, 255) if curr_idx == 0 else (100, 100, 255)
            cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, CLICK_LABELS[curr_idx], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow('Click Points', img)

def get_visual_style(vaep):
    """ VAEP 값에 따라 색상과 두께만 반환 (글로우 삭제) """
    if vaep >= 0.15: 
        # High: 진한 빨강, 두꺼움
        return (0, 0, 255), 4 
    elif vaep >= 0.08: 
        # Mid: 주황/골드, 중간
        return (0, 165, 255), 2 
    else: 
        # Low: 연한 회색, 얇음
        return (200, 200, 200), 1 

def draw_clean_card(img, pt, label, n_val, v_val, y_offset, theme_color):
    """ 깔끔한 다크 글래스 UI 카드 """
    # 텍스트 내용
    lines = [
        (f"{label}", 0.6, 2), 
        (f"Freq: {n_val}", 0.45, 1), 
        (f"VAEP: {v_val:.3f}", 0.45, 1)
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 12
    
    # 박스 크기 계산
    max_w = 0; total_h = 0
    for text, scale, thick in lines:
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        max_w = max(max_w, w)
        total_h += (h + 5)
    
    total_h += 10 # 여유 공간
    
    box_w = max_w + (padding * 3) + 5
    box_h = total_h + (padding * 2)
    
    x, y = pt
    base_y = y - 30 
    
    box_x1 = x - (box_w // 2)
    box_y1 = base_y - box_h - y_offset
    box_x2 = box_x1 + box_w
    box_y2 = base_y - y_offset
    
    # 1. 반투명 배경 (Overlay 없이 직접 그림 - 나중에 전체를 blend 할 것임)
    # 그림자
    cv2.rectangle(img, (box_x1+3, box_y1+3), (box_x2+3, box_y2+3), (0,0,0), -1)
    # 메인 박스 (Dark Gray)
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 30), -1)
    # 컬러 바 (왼쪽 포인트)
    cv2.rectangle(img, (box_x1, box_y1), (box_x1+4, box_y2), theme_color, -1)
    
    # 2. 텍스트 (흰색)
    curr_y = box_y1 + padding + 10
    for i, (text, scale, thick) in enumerate(lines):
        text_x = box_x1 + padding + 8
        cv2.putText(img, text, (text_x, curr_y), font, scale, (255,255,255), thick, cv2.LINE_AA)
        curr_y += (22 if i == 0 else 18)

    # 3. 지시선
    cv2.line(img, (x, y), (x, box_y2), theme_color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, theme_color, -1, cv2.LINE_AA)

def main():
    if not os.path.exists(VIDEO_FILE): print(f"❌ 파일 없음: {VIDEO_FILE}"); return
    cap = cv2.VideoCapture(VIDEO_FILE)
    ret, first_frame = cap.read()
    if not ret: return

    print(f"\n👉 [{ANALYSIS_ZONE}] 모드. 순서대로 클릭하세요: {CLICK_LABELS}")
    cv2.namedWindow('Click Points')
    cv2.setMouseCallback('Click Points', mouse_handler, first_frame)
    cv2.imshow('Click Points', first_frame)
    while len(points) < 4:
        if cv2.waitKey(10) == 27: sys.exit()
    cv2.destroyAllWindows()

    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = first_frame.shape[:2]
    out = cv2.VideoWriter(SAVE_FILE, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    ball_pt = points[0]
    
    # 데이터 준비
    targets = []
    for i, key in enumerate(TARGET_KEYS):
        pt = points[i+1]
        data = CURRENT_STATS.get(key, {'N':0, 'V':0.0})
        # 글로우 없이 색상과 두께만 가져옴
        color, thickness = get_visual_style(data['V'])
        
        targets.append({
            'pt': pt, 'label': key, 'data': data, 
            'color': color, 'thickness': thickness, 'offset': 0
        })

    # 겹침 방지 오프셋
    sorted_targets = sorted(targets, key=lambda t: t['pt'][0])
    if ANALYSIS_ZONE == 'Center':
        sorted_targets[0]['offset'] = 80; sorted_targets[2]['offset'] = 80
    else: sorted_targets[1]['offset'] = 60

    print("🎥 생성 중... (킥 순간 'F' 키!)")
    current_alpha = 1.0; is_fading = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        if current_alpha > 0.01:
            # 1. 오버레이용 복사본 생성 (여기다가 다 그림)
            overlay = frame.copy()
            
            # 2. 직선 그리기 (글로우 X, 그냥 깔끔한 선)
            for t in targets:
                cv2.line(overlay, ball_pt, t['pt'], t['color'], t['thickness'], cv2.LINE_AA)

            # 3. 공 그리기
            cv2.circle(overlay, ball_pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, ball_pt, 8, (0, 255, 255), 2, cv2.LINE_AA)

            # 4. UI 카드 그리기 (오버레이 위에 그림)
            for t in targets:
                draw_clean_card(overlay, t['pt'], t['label'], t['data']['N'], t['data']['V'], t['offset'], t['color'])
            
            # 5. 최종 합성 (페이드 아웃용 Blend)
            # frame(원본)과 overlay(그림 그려진 것)를 current_alpha 비율로 섞음
            # alpha가 1.0이면 overlay 100%, alpha가 0이면 frame 100%
            cv2.addWeighted(overlay, current_alpha, frame, 1 - current_alpha, 0, frame)

        out.write(frame)
        cv2.imshow('Clean Result', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord('f') or key == ord('F'): is_fading = True; print("👋 Fade Out!")
        if is_fading: current_alpha = max(0, current_alpha - FADE_SPEED)

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"✅ 저장 완료: {SAVE_FILE}")

if __name__ == "__main__":
    main()
