import cv2
import numpy as np
import os
import sys
import math

# =========================================================
# [설정]
VIDEO_FILE = 'ScreenRecording_01-03-2026 14-17-46_1.MP4'
HEATMAP_FILE = 'heatmap_overlay.png'
SAVE_FILE = 'result_fade_out.mp4'

# [마스킹 설정]
IGNORE_TOP_RATIO = 0.15
IGNORE_BOTTOM_RATIO = 0.10
IGNORE_LOGO_WIDTH = 0.20
IGNORE_LOGO_HEIGHT = 0.20

# [움직임 부드러움]
SMOOTH_FACTOR = 0.5 

# [페이드 아웃 속도]
# 1프레임당 투명도가 얼마나 줄어들지 (0.02면 약 30프레임(1초) 동안 사라짐)
FADE_SPEED = 0.03
# =========================================================

clicks_src = [] 
clicks_dst = [] 

def mouse_handler_src(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks_src) < 4:
            clicks_src.append([x, y])
            img = param.copy()
            for i, p in enumerate(clicks_src):
                cv2.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
                cv2.putText(img, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('1. Heatmap Points', img)

def mouse_handler_dst(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks_dst) < 4:
            clicks_dst.append([x, y])
            img = param.copy()
            for i, p in enumerate(clicks_dst):
                cv2.circle(img, (p[0], p[1]), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('2. Video Points', img)

def get_green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def check_homography_validity(M):
    if M is None: return False
    det = np.linalg.det(M[:2, :2])
    if det < 0.8 or det > 1.2: return False 
    tx, ty = M[0, 2], M[1, 2]
    if abs(tx) > 100 or abs(ty) > 100: return False 
    return True

def get_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def rotate_points(points, angle_diff, center):
    c, s = np.cos(angle_diff), np.sin(angle_diff)
    R = np.array([[c, -s], [s, c]])
    centered = points - center
    rotated = np.dot(centered, R.T)
    return rotated + center

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, VIDEO_FILE)
    heatmap_path = os.path.join(current_dir, HEATMAP_FILE)
    save_path = os.path.join(current_dir, SAVE_FILE)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: print("❌ 영상 로드 실패"); return

    heatmap_src = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
    if heatmap_src is None: print("❌ 히트맵 로드 실패"); return

    if heatmap_src.shape[2] == 4:
        trans_mask = heatmap_src[:, :, 3] == 0
        heatmap_src[trans_mask] = [0, 0, 0, 0]
        heatmap_img = cv2.cvtColor(heatmap_src, cv2.COLOR_BGRA2BGR)
    else:
        heatmap_img = heatmap_src
    
    heatmap_img = cv2.flip(heatmap_img, 1)

    cv2.namedWindow('1. Heatmap Points')
    cv2.setMouseCallback('1. Heatmap Points', mouse_handler_src, heatmap_img)
    cv2.imshow('1. Heatmap Points', heatmap_img)
    while len(clicks_src) < 4:
        if cv2.waitKey(10) == 27: sys.exit()
    cv2.destroyWindow('1. Heatmap Points')

    cv2.namedWindow('2. Video Points')
    cv2.setMouseCallback('2. Video Points', mouse_handler_dst, first_frame)
    cv2.imshow('2. Video Points', first_frame)
    while len(clicks_dst) < 4:
        if cv2.waitKey(10) == 27: sys.exit()
    cv2.destroyWindow('2. Video Points')

    pts_src = np.float32(clicks_src)
    current_dst_corners = np.float32(clicks_dst).reshape(-1, 2)
    initial_angle = get_angle(current_dst_corners[0], current_dst_corners[1])

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h_vid, w_vid = first_frame.shape[:2]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    roi_mask = np.ones_like(prev_gray) * 255
    top_limit = int(h_vid * IGNORE_TOP_RATIO)
    roi_mask[0:top_limit, :] = 0
    bottom_limit = int(h_vid * (1.0 - IGNORE_BOTTOM_RATIO))
    roi_mask[bottom_limit:, :] = 0
    logo_x_start = int(w_vid * (1.0 - IGNORE_LOGO_WIDTH))
    logo_y_start = int(h_vid * (1.0 - IGNORE_LOGO_HEIGHT))
    roi_mask[logo_y_start:, logo_x_start:] = 0

    green_mask = get_green_mask(first_frame)
    final_mask = cv2.bitwise_and(roi_mask, roi_mask, mask=green_mask)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=2000, qualityLevel=0.01, minDistance=10, mask=final_mask)

    print(f"\n🎥 [Fade Out Mode] 키커가 찰 때 'F' 키를 누르세요!")

    # ★ 투명도 변수 (0.6에서 시작해서 0.0까지 줄어듦)
    current_alpha = 0.6
    is_fading = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        green_mask = get_green_mask(frame)
        final_mask = cv2.bitwise_and(roi_mask, roi_mask, mask=green_mask)

        # -------------------------------------------------------------------
        # 1. 추적 로직 (투명도가 0이 되면 굳이 열심히 추적할 필요 없음 -> 성능 최적화)
        # -------------------------------------------------------------------
        if current_alpha > 0.01:
            if prev_pts is None or len(prev_pts) < 50:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=2000, qualityLevel=0.01, minDistance=10, mask=final_mask)
            
            if prev_pts is not None and len(prev_pts) > 0:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                idx = np.where(status==1)[0]
                good_prev = prev_pts[idx]
                good_curr = curr_pts[idx]

                if len(good_prev) > 20:
                    M_curr, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
                    
                    if check_homography_validity(M_curr):
                        M_curr = M_curr.astype(np.float32)

                        temp_corners = current_dst_corners.reshape(-1, 1, 2)
                        temp_corners = cv2.perspectiveTransform(temp_corners, M_curr)
                        temp_corners = temp_corners.reshape(-1, 2)

                        # 회전 보정
                        current_angle = get_angle(temp_corners[0], temp_corners[1])
                        angle_diff = initial_angle - current_angle
                        center = np.mean(temp_corners, axis=0)
                        corrected_corners = rotate_points(temp_corners, angle_diff, center)

                        # 스무딩
                        current_dst_corners = current_dst_corners * (1 - SMOOTH_FACTOR) + corrected_corners * SMOOTH_FACTOR

                prev_pts = good_curr.reshape(-1, 1, 2)
        else:
            # 히트맵이 사라졌으면 그냥 원본 프레임만 보여주면 됨 (추적 중단)
            prev_pts = None 
        
        prev_gray = curr_gray.copy()

        # -------------------------------------------------------------------
        # 2. 합성 로직 (Alpha 값 조절)
        # -------------------------------------------------------------------
        if current_alpha > 0.01:
            # 호모그래피 변환 및 합성
            M_final = cv2.getPerspectiveTransform(pts_src, current_dst_corners.astype(np.float32))
            warped_heatmap = cv2.warpPerspective(heatmap_img, M_final, (w_vid, h_vid))
            warp_gray = cv2.cvtColor(warped_heatmap, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(warp_gray, 5, 255, cv2.THRESH_BINARY)
            
            video_crop = cv2.bitwise_and(frame, frame, mask=mask)
            heatmap_crop = cv2.bitwise_and(warped_heatmap, warped_heatmap, mask=mask)
            
            # ★ 핵심: current_alpha 값을 사용하여 합성
            blended = cv2.addWeighted(video_crop, 1.0, heatmap_crop, current_alpha, 0)
            
            frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame_bg, blended)

        # -------------------------------------------------------------------
        # 3. 키보드 입력 및 페이드 아웃 처리
        # -------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): break # 스페이스바 종료
        if key == ord('f'): # 'F' 키 누르면 페이드 아웃 시작
            is_fading = True
            print("👋 Fade Out Start!")
        
        if is_fading:
            current_alpha -= FADE_SPEED
            if current_alpha < 0:
                current_alpha = 0
        
        # 상태 표시
        if current_alpha > 0:
            cv2.putText(frame, f"Heatmap ON (Alpha: {current_alpha:.2f}) - Press 'F' to Fade", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Heatmap OFF", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Result with Fade Out', frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 완료! 결과 파일: {SAVE_FILE}")

if __name__ == "__main__":
    main()
