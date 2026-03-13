import cv2
import numpy as np
import random
import time
import math

class HologramEffect:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Sci-Fi Hologram Palette (BGR format) ---
        self.holo_blue  = (255, 200, 50)
        self.ui_color   = (255, 200, 50)
        self.laser_blue = (255, 220, 100)
        self.gun_yellow = (50, 255, 255)

        # Kernels
        self.kernel_3 = np.ones((3, 3), np.uint8)
        self.kernel_5 = np.ones((5, 5), np.uint8)

        # State trackers
        self.frame_count = 0
        self.bg_cache    = None
        self.particles   = None

        self.ui_items = [
            "Box", "Sphere", "Pyramid",
            "Cone", "Cylinder", "Laser Beam", "Finger guns"
        ]

        # Per-hand independent state
        self.hand_state = {
            'left':  self._new_hand_state(),
            'right': self._new_hand_state(),
        }

        self.gun_projectiles = []

    @staticmethod
    def _new_hand_state():
        return dict(
            ui_active=False,       # whole system on/off
            panel_open=False,      # selection panel visible (separate from ui_active)
            selected_object=None,
            palm_open_start=0,
            fist_closed_start=0,
            pinch_start=0,
            hovered_item_index=-1,
        )

    # ================================================================
    #  BACKGROUND
    # ================================================================

    def _draw_cyber_grid_base(self, h, w):
        bg = np.zeros((h, w, 3), dtype=np.float32)
        cx, cy = w // 2, int(h * 0.75)
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        radial_gradient = 1 - (dist_from_center / max_dist)
        bg[:, :, 0] = 45 * radial_gradient
        bg[:, :, 1] = 15 * radial_gradient
        bg[:, :, 2] =  5 * radial_gradient
        bg = bg.astype(np.uint8)
        horizon_y  = int(h * 0.4)
        grid_color = (70, 40, 15)
        for x in range(-w, w * 2, int(w * 0.08)):
            cv2.line(bg, (cx, horizon_y), (x, h), grid_color, 1, cv2.LINE_AA)
        for i in range(1, 16):
            depth_ratio = (i / 15) ** 2.5
            y = horizon_y + int((h - horizon_y) * depth_ratio)
            cv2.line(bg, (0, y), (w, y), grid_color, max(1, int(2 * depth_ratio)), cv2.LINE_AA)
        glow_layer = np.zeros_like(bg)
        cv2.ellipse(glow_layer, (cx, cy), (int(w*0.3), int(h*0.05)), 0, 0, 360, (255, 120, 0), -1)
        glow_layer = cv2.GaussianBlur(glow_layer, (151, 151), 0)
        intense_glow = np.zeros_like(bg)
        cv2.ellipse(intense_glow, (cx, cy), (int(w*0.15), int(h*0.02)), 0, 0, 360, (255, 230, 120), -1)
        intense_glow = cv2.GaussianBlur(intense_glow, (41, 41), 0)
        bg = cv2.add(bg, glow_layer)
        bg = cv2.add(bg, intense_glow)
        cv2.ellipse(bg, (cx, cy), (int(w*0.25), int(h*0.04)), 0, 0, 360, (255, 200, 50), 2, cv2.LINE_AA)
        axes = (int(w*0.35), int(h*0.06))
        for angle in range(0, 360, 15):
            cv2.ellipse(bg, (cx, cy), axes, 0, angle, angle + 8, (255, 150, 20), 2, cv2.LINE_AA)
        return bg

    def _update_particles(self, h, w):
        if self.particles is None:
            self.particles = np.zeros((40, 5))
            self.particles[:, 0] = np.random.randint(0, w, 40)
            self.particles[:, 1] = np.random.randint(0, h, 40)
            self.particles[:, 2] = np.random.uniform(0.5, 2.0, 40)
            self.particles[:, 3] = np.random.uniform(1, 3, 40)
        self.particles[:, 1] -= self.particles[:, 2]
        off_screen = self.particles[:, 1] < 0
        self.particles[off_screen, 1] = h
        self.particles[off_screen, 0] = np.random.randint(0, w, np.sum(off_screen))
        pulse = (np.sin(self.frame_count * 0.1) + 1) / 2
        self.particles[:, 4] = 120 + 135 * pulse

    def _get_dynamic_background(self, h, w):
        if self.bg_cache is None or self.bg_cache.shape[:2] != (h, w):
            self.bg_cache = self._draw_cyber_grid_base(h, w)
        final_bg   = self.bg_cache.copy()
        scanline_y = int((self.frame_count * 3) % h)
        cv2.line(final_bg, (0, scanline_y), (w, scanline_y), (80, 40, 10), 1)
        breathe  = 0.96 + 0.04 * np.sin(self.frame_count * 0.05)
        
        # Accelerated floating point math
        final_bg = cv2.convertScaleAbs(final_bg, alpha=breathe)
        
        self._update_particles(h, w)
        for p in self.particles:
            x, y, _, size, bright = int(p[0]), int(p[1]), p[2], int(p[3]), int(p[4])
            color = (bright, int(bright*0.7), int(bright*0.2))
            cv2.circle(final_bg, (x, y), size, color, -1, cv2.LINE_AA)
        return final_bg

    # ================================================================
    #  PERSON HOLOGRAM 
    # ================================================================

    def _create_hologram_person(self, frame, mask, h, w):
        person_rgb = cv2.bitwise_and(frame, frame, mask=mask)
        gray       = cv2.cvtColor(person_rgb, cv2.COLOR_BGR2GRAY)
        holo_color = np.zeros_like(frame, dtype=np.float32)
        holo_color[:, :, 0] = gray * 1.6
        holo_color[:, :, 1] = gray * 1.2
        holo_color[:, :, 2] = gray * 0.3
        holo_color = np.clip(holo_color, 0, 255).astype(np.uint8)
        edges     = cv2.Canny(gray, 50, 150)
        edges     = cv2.dilate(edges, self.kernel_3, iterations=1)
        edge_glow = cv2.GaussianBlur(edges, (9, 9), 0)
        edge_layer = np.zeros_like(frame)
        edge_layer[:, :, 0] = edge_glow
        edge_layer[:, :, 1] = (edge_glow * 0.6).astype(np.uint8)
        holo_color = cv2.add(holo_color, edge_layer)
        y_indices     = np.arange(h).reshape(h, 1)
        scanline_wave = 0.85 + 0.15 * np.sin(y_indices * 0.4 - self.frame_count * 0.5)
        holo_color    = (holo_color * scanline_wave[:, :, np.newaxis]).astype(np.uint8)
        if random.random() < 0.02:
            shift = random.randint(-12, 12)
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            holo_color = cv2.warpAffine(holo_color, M, (w, h))
        return holo_color

    def _filter_segmentation(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_5)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        return mask

    # ================================================================
    #  GESTURE HELPERS 
    # ================================================================

    def _dist(self, p1, p2, w, h):
        return math.hypot((p1[0]-p2[0])*w, (p1[1]-p2[1])*h)

    def _is_palm_open(self, lms):
        if lms is None: return False
        tips = [8, 12, 16, 20]; mcps = [5, 9, 13, 17]
        for tip, mcp in zip(tips, mcps):
            if self._dist(lms[tip], lms[0], 1, 1) < self._dist(lms[mcp], lms[0], 1, 1):
                return False
        return True

    def _is_fist(self, lms):
        if lms is None: return False
        tips = [8, 12, 16, 20]; mcps = [5, 9, 13, 17]
        for tip, mcp in zip(tips, mcps):
            if self._dist(lms[tip], lms[0], 1, 1) > self._dist(lms[mcp], lms[0], 1, 1):
                return False
        return True

    def _is_pinching(self, lms, w, h):
        if lms is None: return False
        return self._dist(lms[4], lms[8], w, h) < (w * 0.05)

    def _is_palm_up(self, lms):
        if lms is None or not self._is_palm_open(lms): return False
        return lms[0][1] > lms[12][1]

    def _is_pointing(self, lms):
        if lms is None: return False
        index_extended = self._dist(lms[8],  lms[0], 1,1) > self._dist(lms[5],  lms[0], 1,1)
        middle_curled  = self._dist(lms[12], lms[0], 1,1) < self._dist(lms[9],  lms[0], 1,1)
        ring_curled    = self._dist(lms[16], lms[0], 1,1) < self._dist(lms[13], lms[0], 1,1)
        pinky_curled   = self._dist(lms[20], lms[0], 1,1) < self._dist(lms[17], lms[0], 1,1)
        return index_extended and middle_curled and ring_curled and pinky_curled

    # ================================================================
    #  UI THUMBNAILS 
    # ================================================================

    def _draw_thumbnail(self, img, item, cx, cy, size, color):
        s = size // 2
        if item == "Box":
            pts = np.array([[cx, cy-s],[cx+s, cy-s//2],[cx, cy],[cx-s, cy-s//2]], np.int32)
            cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
            cv2.line(img, (cx-s, cy-s//2), (cx-s, cy+s//2), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx+s, cy-s//2), (cx+s, cy+s//2), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy+s), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx-s, cy+s//2), (cx, cy+s), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx+s, cy+s//2), (cx, cy+s), color, 1, cv2.LINE_AA)
        elif item == "Sphere":
            cv2.circle(img, (cx, cy), s, color, 1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (s, s//3), 0, 0, 360, color, 1, cv2.LINE_AA)
        elif item == "Pyramid":
            pts = np.array([[cx, cy-s],[cx+s, cy+s//2],[cx-s, cy+s//2]], np.int32)
            cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy-s), (cx, cy+s//2), color, 1, cv2.LINE_AA)
        elif item == "Cone":
            cv2.line(img, (cx, cy-s), (cx-s, cy+s//2), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy-s), (cx+s, cy+s//2), color, 1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy+s//2), (s, s//4), 0, 0, 360, color, 1, cv2.LINE_AA)
        elif item == "Cylinder":
            cv2.ellipse(img, (cx, cy-s//2), (s, s//4), 0, 0, 360, color, 1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy+s//2), (s, s//4), 0, 0, 360, color, 1, cv2.LINE_AA)
            cv2.line(img, (cx-s, cy-s//2), (cx-s, cy+s//2), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx+s, cy-s//2), (cx+s, cy+s//2), color, 1, cv2.LINE_AA)
        elif item == "Laser Beam":
            cv2.line(img, (cx-s, cy+s), (cx+s, cy-s), color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
        elif item == "Finger guns":
            cv2.line(img, (cx-s, cy-1), (cx+s, cy-1), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx-s//3, cy+1), (cx-s//3, cy+s//2), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx-s,    cy+1), (cx-s//3, cy+s//2), color, 1, cv2.LINE_AA)
            flash_c = self.gun_yellow
            cv2.circle(img, (cx+s+3, cy-1), 2, flash_c, -1, cv2.LINE_AA)
            cv2.circle(img, (cx+s+6, cy-3), 1, flash_c, -1, cv2.LINE_AA)
            cv2.circle(img, (cx+s+6, cy+1), 1, flash_c, -1, cv2.LINE_AA)
            cv2.line(img, (cx+s, cy-1), (cx+s+5, cy-1), flash_c, 1, cv2.LINE_AA)

    # ================================================================
    #  UI PANEL 
    # ================================================================

    def _draw_main_ui(self, target_img, hand_lms, h, w, side='right'):
        hs = self.hand_state[side]
        if not hs['panel_open']:
            return target_img

        panel_w = int(w * 0.17) 
        item_h  = 50            
        panel_h = len(self.ui_items) * (item_h + 3) + 44

        # Affix UI panels to the edges based on mirrored selfie view mapping:
        # Physical Right Hand ('right') appears on the left side of the screen
        # Physical Left Hand ('left') appears on the right side of the screen
        if side == 'right':
            px = 14
        else:
            px = w - panel_w - 14

        py = (h - panel_h) // 2

        tl = (px, py)
        br = (px + panel_w, py + panel_h)

        try:
            sub   = target_img[tl[1]:br[1], tl[0]:br[0]]
            glass = np.full_like(sub, (25, 12, 0), dtype=np.uint8)
            target_img[tl[1]:br[1], tl[0]:br[0]] = cv2.add(sub, glass)
        except Exception:
            pass

        glow_b = np.zeros_like(target_img)
        cv2.rectangle(glow_b, tl, br, self.ui_color, 3)
        glow_b = cv2.GaussianBlur(glow_b, (11, 11), 0)
        target_img = cv2.add(target_img, glow_b)
        cv2.rectangle(target_img, tl, br, self.ui_color, 1, cv2.LINE_AA)

        CL = 12
        for (cpx, cpy), (dx, dy) in zip(
                [tl, (br[0], tl[1]), (tl[0], br[1]), br],
                [(1,1),(-1,1),(1,-1),(-1,-1)]):
            cv2.line(target_img,(cpx,cpy),(cpx+dx*CL,cpy),(255,255,160),2,cv2.LINE_AA)
            cv2.line(target_img,(cpx,cpy),(cpx,cpy+dy*CL),(255,255,160),2,cv2.LINE_AA)

        label = f"{'L' if side=='left' else 'R'} HAND  [PINCH TO SELECT]"
        cv2.putText(target_img, label,
                    (tl[0]+6, tl[1]+16), self.font, 0.30, (200,220,180), 1, cv2.LINE_AA)
        cv2.line(target_img, (tl[0]+3, tl[1]+22), (br[0]-3, tl[1]+22), self.ui_color, 1)

        pointer_x, pointer_y = -1, -1
        is_pinching = False
        if hand_lms is not None:
            pointer_x   = int(hand_lms[8][0] * w)
            pointer_y   = int(hand_lms[8][1] * h)
            is_pinching = self._is_pinching(hand_lms, w, h)
            cv2.circle(target_img, (pointer_x, pointer_y), 5, (255,255,255), -1)
            cv2.circle(target_img, (pointer_x, pointer_y), 9, self.ui_color, 1, cv2.LINE_AA)

        start_y = tl[1] + 28
        currently_hovered = -1

        for i, item in enumerate(self.ui_items):
            i_tl = (tl[0]+4,   start_y + i*(item_h+3))
            i_br = (br[0]-4,   i_tl[1]+item_h)

            is_hovered  = (i_tl[0] < pointer_x < i_br[0] and
                           i_tl[1] < pointer_y < i_br[1])
            is_selected = (hs['selected_object'] == item)

            color    = (255,255,255) if (is_hovered or is_selected) else (140,140,140)
            bg_color = None
            if is_selected:
                bg_color = (90, 45, 0)
            elif is_hovered:
                currently_hovered = i
                bg_color = (70, 55, 15)

            if bg_color:
                try:
                    ovr = target_img[i_tl[1]:i_br[1], i_tl[0]:i_br[0]]
                    hl  = np.full_like(ovr, bg_color, dtype=np.uint8)
                    target_img[i_tl[1]:i_br[1], i_tl[0]:i_br[0]] = cv2.add(ovr, hl)
                except Exception:
                    pass

            cv2.rectangle(target_img, i_tl, i_br, self.ui_color, 1)

            icon_cx = i_tl[0] + 15
            icon_cy = i_tl[1] + item_h // 2
            self._draw_thumbnail(target_img, item, icon_cx, icon_cy, 16, color)

            fs = 0.30 if len(item) > 8 else 0.33
            cv2.putText(target_img, item,
                        (icon_cx+18, icon_cy+5), self.font, fs, color, 1, cv2.LINE_AA)

            if is_hovered and is_pinching and hs['pinch_start'] > 0:
                elapsed  = time.time() - hs['pinch_start']
                progress = min(1.0, elapsed / 1.5)
                arc_cx   = i_br[0] - 12
                arc_cy   = i_tl[1] + item_h // 2
                cv2.ellipse(target_img, (arc_cx, arc_cy), (9,9),
                            -90, 0, int(360*progress), (0,255,210), 2, cv2.LINE_AA)
                cv2.ellipse(target_img, (arc_cx, arc_cy), (9,9),
                            0, 0, 360, (50,40,10), 1, cv2.LINE_AA)

        if currently_hovered != -1 and is_pinching:
            if hs['hovered_item_index'] != currently_hovered:
                hs['hovered_item_index'] = currently_hovered
                hs['pinch_start']        = time.time()
            elif time.time() - hs['pinch_start'] >= 1.5:
                hs['selected_object']    = self.ui_items[currently_hovered]
                hs['panel_open']         = False 
                hs['pinch_start']        = 0
                hs['hovered_item_index'] = -1
        else:
            hs['pinch_start']        = 0
            hs['hovered_item_index'] = -1

        return target_img

    # ================================================================
    #  3-D WIREFRAME 
    # ================================================================

    def _draw_3d_wireframe(self, img, shape, palm_cx, palm_cy, radius):
        angle_y    = self.frame_count * 0.032
        tilt_x     = 0.45
        bob_offset = math.sin(self.frame_count * 0.07) * 10
        cy = int((palm_cy - 200) + bob_offset)
        cx = palm_cx

        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        cos_x, sin_x = math.cos(tilt_x),  math.sin(tilt_x)

        def project(x, y, z):
            rx = x * cos_y - z * sin_y
            rz = x * sin_y + z * cos_y
            ry = y * cos_x - rz * sin_x
            return (int(cx + rx * radius), int(cy - ry * radius))

        pts   = []
        lines = []

        if shape == "Box":
            vs = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                  (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)]
            pts   = [project(*v) for v in vs]
            lines = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

        elif shape == "Pyramid":
            vs    = [(-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1),(0,1.2,0)]
            pts   = [project(*v) for v in vs]
            lines = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]

        elif shape == "Cone":
            for i in range(16):
                a = (i / 16.0) * math.pi * 2
                pts.append(project(math.cos(a), -1, math.sin(a)))
            pts.append(project(0, 1.2, 0))
            lines  = [(i, (i+1)%16) for i in range(16)]
            lines += [(i, 16) for i in range(0, 16, 4)]

        elif shape == "Cylinder":
            for i in range(16):
                a = (i / 16.0) * math.pi * 2
                pts.append(project(math.cos(a), -1, math.sin(a)))
                pts.append(project(math.cos(a),  1, math.sin(a)))
            for i in range(16):
                lines.append((i*2,   ((i+1)%16)*2))
                lines.append((i*2+1, ((i+1)%16)*2+1))
            for i in range(0, 16, 4):
                lines.append((i*2, i*2+1))

        elif shape == "Sphere":
            for ring in range(3):
                ring_angle = ring * (math.pi / 3)
                for i in range(16):
                    a = (i / 16.0) * math.pi * 2
                    x = math.cos(a) * math.cos(ring_angle)
                    y = math.sin(a)
                    z = math.cos(a) * math.sin(ring_angle)
                    pts.append(project(x, y, z))
                offset = ring * 16
                for i in range(16):
                    lines.append((offset+i, offset+(i+1)%16))

        overlay = np.zeros_like(img)

        beam_color = (self.holo_blue[0]//4, self.holo_blue[1]//4, self.holo_blue[2]//4)
        pulse_r    = int(14 + 4 * math.sin(self.frame_count * 0.12))
        cv2.circle(overlay, (palm_cx, palm_cy), pulse_r, self.holo_blue, 2, cv2.LINE_AA)
        cv2.circle(overlay, (palm_cx, palm_cy), 5, (255,255,255), -1)
        if len(pts) > 0:
            for pt in [pts[0], pts[len(pts)//2]]:
                cv2.line(overlay, (palm_cx, palm_cy), pt, beam_color, 1, cv2.LINE_AA)

        for p1_idx, p2_idx in lines:
            pt1 = pts[p1_idx]; pt2 = pts[p2_idx]
            cv2.line(overlay, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)
            cv2.line(overlay, pt1, pt2, self.holo_blue, 3, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(overlay, pt, 3, (255,255,255), -1, cv2.LINE_AA)

        y_idx   = np.arange(img.shape[0], dtype=np.float32).reshape(-1, 1)
        scan    = 0.72 + 0.28 * np.sin(y_idx * 0.65 - self.frame_count * 0.42)
        overlay = (overlay.astype(np.float32) * scan[:,:,None]).astype(np.uint8)

        # FAST DOWNSCALE-BLUR-UPSCALE 
        h_ov, w_ov = overlay.shape[:2]
        
        glow1 = cv2.GaussianBlur(overlay, (7, 7), 0)
        
        small_ov2 = cv2.resize(overlay, (w_ov // 2, h_ov // 2), interpolation=cv2.INTER_LINEAR)
        glow2_small = cv2.GaussianBlur(small_ov2, (11, 11), 0)
        glow2 = cv2.resize(glow2_small, (w_ov, h_ov), interpolation=cv2.INTER_LINEAR)
        
        small_ov3 = cv2.resize(overlay, (w_ov // 4, h_ov // 4), interpolation=cv2.INTER_LINEAR)
        glow3_small = cv2.GaussianBlur(small_ov3, (13, 13), 0)
        glow3 = cv2.resize(glow3_small, (w_ov, h_ov), interpolation=cv2.INTER_LINEAR)

        flicker = 0.82 + 0.18 * math.sin(self.frame_count * 0.19)
        glow2   = cv2.convertScaleAbs(glow2, alpha=flicker)
        glow3   = cv2.convertScaleAbs(glow3, alpha=flicker * 0.55)

        img = cv2.add(img, glow3)
        img = cv2.add(img, glow2)
        img = cv2.add(img, glow1)
        img = cv2.add(img, overlay)
        return img

    # ================================================================
    #  WEAPONS 
    # ================================================================

    def _shoot_laser(self, img, start_pt, end_pt, h, w):
        dist = math.hypot(end_pt[0]-start_pt[0], end_pt[1]-start_pt[1])
        if dist < 1: return img
        dx = (end_pt[0]-start_pt[0]) / dist
        dy = (end_pt[1]-start_pt[1]) / dist
        overlay = np.zeros_like(img)
        cv2.line(overlay, start_pt, end_pt, (255,255,255), 2, cv2.LINE_AA)
        cv2.line(overlay, start_pt, end_pt, self.laser_blue, 8, cv2.LINE_AA)
        time_offset = self.frame_count * 0.5
        for i in range(0, int(dist), 8):
            px = start_pt[0] + dx * i
            py = start_pt[1] + dy * i
            swirl_x = math.cos(i*0.1 + time_offset) * 15
            swirl_y = math.sin(i*0.1 + time_offset) * 15
            nx, ny = -dy, dx
            fx = int(px + nx * swirl_x)
            fy = int(py + ny * swirl_y)
            if 0 <= fx < w and 0 <= fy < h:
                cv2.circle(overlay, (fx, fy), 2, (255,255,255), -1)
                cv2.circle(overlay, (fx, fy), 5, self.laser_blue, -1)
        glow = cv2.GaussianBlur(overlay, (21, 21), 0)
        img  = cv2.add(img, overlay)
        img  = cv2.add(img, glow)
        return img

    def _shoot_finger_guns(self, img, start_pt, dir_vec, h, w):
        if self.frame_count % 4 == 0:
            self.gun_projectiles.append({
                'x': start_pt[0], 'y': start_pt[1],
                'vx': dir_vec[0]*30, 'vy': dir_vec[1]*30, 'life': 20
            })
        overlay = np.zeros_like(img)
        alive   = []
        for p in self.gun_projectiles:
            p['x'] += p['vx']; p['y'] += p['vy']; p['life'] -= 1
            if p['life'] > 0:
                p1 = (int(p['x']),              int(p['y']))
                p2 = (int(p['x']-p['vx']*0.5), int(p['y']-p['vy']*0.5))
                cv2.line(overlay,   p1, p2, (255,255,255),   2, cv2.LINE_AA)
                cv2.line(overlay,   p1, p2, self.gun_yellow, 6, cv2.LINE_AA)
                cv2.circle(overlay, p1, 4, (255,255,255), -1)
                alive.append(p)
        self.gun_projectiles = alive
        glow = cv2.GaussianBlur(overlay, (15, 15), 0)
        img  = cv2.add(img, overlay)
        img  = cv2.add(img, glow)
        return img

    # ================================================================
    #  STATUS BAR + PALM LOADING BARS
    # ================================================================

    def _draw_status_and_loading(self, img, h, w):
        BAR_H = 38
        try:
            bar_bg = np.full((BAR_H, w, 3), (10, 5, 2), dtype=np.uint8)
            img[:BAR_H] = cv2.addWeighted(img[:BAR_H], 0.55, bar_bg, 0.45, 0)
        except Exception:
            pass
        cv2.line(img, (0, BAR_H), (w, BAR_H), self.ui_color, 1)

        lhs = self.hand_state['left']
        rhs = self.hand_state['right']
        either_active = lhs['ui_active'] or rhs['ui_active']

        if either_active:
            pulse = int(175 + 80 * math.sin(self.frame_count * 0.14))
            lc    = (0, pulse, int(pulse * 0.82))
            label = "UI ACTIVE"
            parts = []
            if lhs['selected_object']: parts.append(f"L:{lhs['selected_object']}")
            if rhs['selected_object']: parts.append(f"R:{rhs['selected_object']}")
            hint = "  |  ".join(parts) if parts else "Pinch to select  |  Fist 5s to close"
        else:
            lc    = (70, 75, 60)
            label = "UI INACTIVE"
            hint  = "Hold Open Palm 5s to Activate"

        (tw, _), _ = cv2.getTextSize(label, self.font, 0.52, 1)
        cv2.putText(img, label, ((w-tw)//2, 26), self.font, 0.52, lc, 1, cv2.LINE_AA)
        (hw, _), _ = cv2.getTextSize(hint,  self.font, 0.30, 1)
        cv2.putText(img, hint,  (w-hw-8,   26), self.font, 0.30, (95,115,75), 1, cv2.LINE_AA)

        BAR_Y   = BAR_H + 4
        BAR_TH  = 5
        BAR_LEN = int(w * 0.18)

        # Swapped positions to align with mirrored selfie view
        # Right hand ('right') is on the left side of the screen
        # Left hand ('left') is on the right side of the screen
        for side, hs, bar_x in [('right', rhs,    14),
                                ('left',  lhs,    w - 14 - BAR_LEN)]:
            now = time.time()
            if not hs['ui_active'] and hs['palm_open_start'] > 0:
                prog   = min(1.0, (now - hs['palm_open_start']) / 5.0)
                filled = int(BAR_LEN * prog)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+BAR_LEN, BAR_Y+BAR_TH), (40,30,10), -1)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+filled,  BAR_Y+BAR_TH), (0,200,170), -1)
                cv2.putText(img, f"{'L' if side=='left' else 'R'} OPEN {int(prog*100)}%",
                            (bar_x, BAR_Y+BAR_TH+12), self.font, 0.30, (0,200,170), 1, cv2.LINE_AA)
            elif hs['ui_active'] and hs['fist_closed_start'] > 0:
                prog   = min(1.0, (now - hs['fist_closed_start']) / 5.0)
                filled = int(BAR_LEN * prog)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+BAR_LEN, BAR_Y+BAR_TH), (20,10,30), -1)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+filled,  BAR_Y+BAR_TH), (0,60,220), -1)
                cv2.putText(img, f"{'L' if side=='left' else 'R'} OFF {int(prog*100)}%",
                            (bar_x, BAR_Y+BAR_TH+12), self.font, 0.30, (0,80,240), 1, cv2.LINE_AA)

        return img

    # ================================================================
    #  PER-HAND GESTURE STATE UPDATE
    # ================================================================

    def _update_hand_state(self, lms, side, h, w):
        hs  = self.hand_state[side]
        now = time.time()

        if lms is None:
            hs['palm_open_start']   = 0
            hs['fist_closed_start'] = 0
            return

        palm_open = self._is_palm_open(lms)
        fist      = self._is_fist(lms)

        if palm_open and not hs['ui_active'] and not fist:
            if hs['palm_open_start'] == 0:
                hs['palm_open_start'] = now
            elif now - hs['palm_open_start'] >= 5.0:
                hs['ui_active']       = True
                hs['panel_open']      = True  
                hs['palm_open_start'] = 0
        else:
            hs['palm_open_start'] = 0

        if fist and hs['ui_active']:
            if hs['fist_closed_start'] == 0:
                hs['fist_closed_start'] = now
            elif now - hs['fist_closed_start'] >= 5.0:
                hs['ui_active']         = False
                hs['panel_open']        = False
                hs['selected_object']   = None
                hs['fist_closed_start'] = 0
        else:
            hs['fist_closed_start'] = 0

    # ================================================================
    #  MAIN PROCESS FRAME
    # ================================================================

    def process_frame(self, frame: np.ndarray, tracking_data=None) -> np.ndarray:
        self.frame_count += 1
        h, w = frame.shape[:2]

        # 1. Background
        output = self._get_dynamic_background(h, w)

        # 2. Hologram person
        if tracking_data and getattr(tracking_data, 'has_person', False):
            raw_mask = getattr(tracking_data, 'segmentation_mask', None)
            if raw_mask is not None:
                if len(raw_mask.shape) > 2: raw_mask = raw_mask.squeeze()
                if raw_mask.shape != (h, w): raw_mask = cv2.resize(raw_mask, (w, h))
                binary_mask  = (raw_mask > 128).astype(np.uint8) * 255
                cleaned_mask = self._filter_segmentation(binary_mask)
                holo_person  = self._create_hologram_person(frame, cleaned_mask, h, w)
                breathe      = 0.7 + 0.1 * np.sin(self.frame_count * 0.1)
                
                # Accelerated floating point masking
                scaled_holo = cv2.convertScaleAbs(holo_person, alpha=breathe)
                holo_masked = cv2.bitwise_and(scaled_holo, scaled_holo, mask=cleaned_mask)
                output       = cv2.add(output, holo_masked)

        left_lms  = getattr(tracking_data, 'left_hand_landmarks',  None) if tracking_data else None
        right_lms = getattr(tracking_data, 'right_hand_landmarks', None) if tracking_data else None

        # 3. Update gesture state
        self._update_hand_state(left_lms,  'left',  h, w)
        self._update_hand_state(right_lms, 'right', h, w)

        # 4. Draw UI panels
        if self.hand_state['left']['panel_open']:
            output = self._draw_main_ui(output, left_lms,  h, w, side='left')
        if self.hand_state['right']['panel_open']:
            output = self._draw_main_ui(output, right_lms, h, w, side='right')

        # 5. Render selected objects / weapons per hand
        for lms, side in [(left_lms, 'left'), (right_lms, 'right')]:
            hs  = self.hand_state[side]
            obj = hs['selected_object']
            if obj is None or lms is None:
                continue

            palm_cx = int(lms[9][0] * w)
            palm_cy = int(lms[9][1] * h)

            if obj in ["Box", "Sphere", "Pyramid", "Cone", "Cylinder"]:
                if self._is_palm_up(lms):
                    output = self._draw_3d_wireframe(output, obj, palm_cx, palm_cy, radius=70)

            elif obj in ["Laser Beam", "Finger guns"]:
                if self._is_pointing(lms):
                    index_mcp = (int(lms[5][0]*w), int(lms[5][1]*h))
                    index_tip = (int(lms[8][0]*w), int(lms[8][1]*h))
                    dist = math.hypot(index_tip[0]-index_mcp[0], index_tip[1]-index_mcp[1])
                    if dist > 0:
                        dx  = (index_tip[0]-index_mcp[0]) / dist
                        dy  = (index_tip[1]-index_mcp[1]) / dist
                        end = (int(index_tip[0]+dx*2000), int(index_tip[1]+dy*2000))
                        if obj == "Laser Beam":
                            output = self._shoot_laser(output, index_tip, end, h, w)
                        else:
                            output = self._shoot_finger_guns(output, index_tip, (dx,dy), h, w)

        # 6. Status strip + loading bars
        output = self._draw_status_and_loading(output, h, w)

        return output

    def close(self):
        pass


if __name__ == "__main__":
    print("Run this via main.py to pass tracking_data into the processor.")