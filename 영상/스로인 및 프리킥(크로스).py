import cv2
import numpy as np
import os
import sys
import math

# =========================================================
# 👇 [설정] 스로인 영상으로 수정하세요!
# =========================================================
VIDEO_FILE = 'ScreenRecording_01-11-2026 14-30-23_1.mov'  # 스로인 원본 영상
HEATMAP_FILE = 'heatmap_throwin.png'   # 스로인 히트맵
SAVE_FILE = 'throw_ai3.mp4'

FLIP_HEATMAP = False  # 상황에 맞춰 True/False 조절
# =========================================================

# [기타 설정]
FADE_SPEED = 0.03
SMOOTH_FACTOR = 0.5 
IGNORE_TOP_RATIO = 0.15
IGNORE_BOTTOM_RATIO = 0.10
IGNORE_LOGO_WIDTH = 0.20
IGNORE_LOGO_HEIGHT = 0.20

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
    global clicks_src, clicks_dst
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, VIDEO_FILE)
    heatmap_path = os.path.join(current_dir, HEATMAP_FILE)
    save_path = os.path.join(current_dir, SAVE_FILE)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: print(f"❌ 영상 로드 실패: {VIDEO_FILE}"); return

    heatmap_src = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
    if heatmap_src is None: print(f"❌ 히트맵 로드 실패: {HEATMAP_FILE}"); return

    if heatmap_src.shape[2] == 4:
        trans_mask = heatmap_src[:, :, 3] == 0
        heatmap_src[trans_mask] = [0, 0, 0, 0]
        heatmap_img = cv2.cvtColor(heatmap_src, cv2.COLOR_BGRA2BGR)
    else:
        heatmap_img = heatmap_src
    
    if FLIP_HEATMAP:
        heatmap_img = cv2.flip(heatmap_img, 1)

    h_vid, w_vid = first_frame.shape[:2]

    # ============================================================
    # 🔁 [반복 구간] 마음에 들 때까지 점 찍기 반복
    # ============================================================
    while True:
        clicks_src = []
        clicks_dst = []
        print("\n🔄 [1단계] 히트맵 기준점 4개를 찍으세요. (좌상->우상->우하->좌하 순서 추천)")
        
        cv2.namedWindow('1. Heatmap Points')
        cv2.setMouseCallback('1. Heatmap Points', mouse_handler_src, heatmap_img)
        cv2.imshow('1. Heatmap Points', heatmap_img)
        while len(clicks_src) < 4:
            if cv2.waitKey(10) == 27: sys.exit()
        cv2.destroyWindow('1. Heatmap Points')

        print("🔄 [2단계] 영상 기준점 4개를 '같은 순서'로 찍으세요.")
        
        temp_frame = first_frame.copy()
        cv2.namedWindow('2. Video Points')
        cv2.setMouseCallback('2. Video Points', mouse_handler_dst, temp_frame)
        cv2.imshow('2. Video Points', temp_frame)
        while len(clicks_dst) < 4:
            if cv2.waitKey(10) == 27: sys.exit()
        cv2.destroyWindow('2. Video Points')

        # --- 미리보기 생성 ---
        pts_src = np.float32(clicks_src)
        pts_dst = np.float32(clicks_dst)
        M_preview = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_preview = cv2.warpPerspective(heatmap_img, M_preview, (w_vid, h_vid))
        
        preview_frame = first_frame.copy()
        warp_gray = cv2.cvtColor(warped_preview, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(warp_gray, 5, 255, cv2.THRESH_BINARY)
        video_crop = cv2.bitwise_and(preview_frame, preview_frame, mask=mask)
        heatmap_crop = cv2.bitwise_and(warped_preview, warped_preview, mask=mask)
        blended = cv2.addWeighted(video_crop, 1.0, heatmap_crop, 0.7, 0) # 미리보기는 좀 진하게
        frame_bg = cv2.bitwise_and(preview_frame, preview_frame, mask=cv2.bitwise_not(mask))
        final_preview = cv2.add(frame_bg, blended)

        cv2.putText(final_preview, "Space: Start / R: Retry", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow('Preview (Check Match)', final_preview)
        
        print("\n👀 미리보기를 확인하세요!")
        print("   - [Spacebar]: 좋음! 추적 시작")
        print("   - [R 키]: 꼬였음.. 다시 찍기")
        
        key = cv2.waitKey(0)
        cv2.destroyWindow('Preview (Check Match)')

        if key == ord('r') or key == ord('R'):
            print("🔄 다시 찍습니다...")
            continue # 루프 처음으로
        elif key == ord(' ') or key == 27: # 스페이스바
            break # 루프 탈출 -> 추적 시작
    # ============================================================

    # 여기서부터 추적 시작 (기존 코드와 동일)
    current_dst_corners = np.float32(clicks_dst).reshape(-1, 2)
    initial_angle = get_angle(current_dst_corners[0], current_dst_corners[1])

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
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

    print(f"\n🎥 [시작] 추적 중... 키커가 던질 때 'F' 키를 누르세요!")

    current_alpha = 0.6
    is_fading = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        green_mask = get_green_mask(frame)
        final_mask = cv2.bitwise_and(roi_mask, roi_mask, mask=green_mask)

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
                        current_angle = get_angle(temp_corners[0], temp_corners[1])
                        angle_diff = initial_angle - current_angle
                        center = np.mean(temp_corners, axis=0)
                        corrected_corners = rotate_points(temp_corners, angle_diff, center)
                        current_dst_corners = current_dst_corners * (1 - SMOOTH_FACTOR) + corrected_corners * SMOOTH_FACTOR

                prev_pts = good_curr.reshape(-1, 1, 2)
        else:
            prev_pts = None 
        
        prev_gray = curr_gray.copy()

        if current_alpha > 0.01:
            M_final = cv2.getPerspectiveTransform(pts_src, current_dst_corners.astype(np.float32))
            warped_heatmap = cv2.warpPerspective(heatmap_img, M_final, (w_vid, h_vid))
            warp_gray = cv2.cvtColor(warped_heatmap, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(warp_gray, 5, 255, cv2.THRESH_BINARY)
            
            video_crop = cv2.bitwise_and(frame, frame, mask=mask)
            heatmap_crop = cv2.bitwise_and(warped_heatmap, warped_heatmap, mask=mask)
            blended = cv2.addWeighted(video_crop, 1.0, heatmap_crop, current_alpha, 0)
            frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame_bg, blended)

        out.write(frame)
        cv2.imshow('Result', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): break 
        if key == 27: break
        if key == ord('f') or key == ord('F'): 
            is_fading = True
            print("👋 Fade Out Start!")
        
        if is_fading:
            current_alpha -= FADE_SPEED
            if current_alpha < 0: current_alpha = 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 완료! 파일 저장됨: {SAVE_FILE}")

if __name__ == "__main__":
    main()
