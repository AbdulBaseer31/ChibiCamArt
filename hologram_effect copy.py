import cv2
import numpy as np
import random
import time
import math


# ================================================================
#  SCENE PHYSICS ENGINE
# ================================================================
class ScenePhysicsEngine:
    """
    2D/3D physics and state management engine for an OpenCV/MediaPipe AR application.
    Handles entity lifecycle, movement, and collision detection.
    """
    def __init__(self):
        self.bullets: list[dict] = []
        self.lasers: list[dict] = []
        self.shapes: dict = {
            'left':  {'type': None, 'cx': 0.0, 'cy': 0.0, 'r': 70.0, 'glowing': False},
            'right': {'type': None, 'cx': 0.0, 'cy': 0.0, 'r': 70.0, 'glowing': False},
        }
        self.sphere_state: dict = {
            'water_level': 0.0,   # 0.0 – 100.0
            'last_update': time.time(),
        }
        self._last_time: float = time.time()

    def register_input(self, hands_data: dict, fired_bullets: list, fired_lasers: list):
        for side in ('left', 'right'):
            info = (hands_data or {}).get(side)
            if info:
                self.shapes[side].update(info)
                self.shapes[side]['glowing'] = False   # reset each frame
            else:
                self.shapes[side]['type'] = None

        for b in (fired_bullets or []):
            self.bullets.append(dict(b))

        for laser in (fired_lasers or []):
            entry = dict(laser)
            entry.setdefault('heat', 0)
            entry.setdefault('active', True)
            self.lasers.append(entry)

    def update(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # Sphere water leak
        ws = self.sphere_state
        ws['water_level'] = max(0.0, ws['water_level'] - 10 * dt)
        ws['last_update'] = now

        # Move bullets
        for b in self.bullets:
            b['x'] += b['vx']
            b['y'] += b['vy']
            b['life'] -= 1

        # Process lasers (Heat, Movement, and Life)
        for laser in self.lasers:
            if laser['active']:
                # Move laser if it has velocity
                if 'vx' in laser and 'vy' in laser:
                    laser['p1'] = (laser['p1'][0] + laser['vx'], laser['p1'][1] + laser['vy'])
                    laser['p2'] = (laser['p2'][0] + laser['vx'], laser['p2'][1] + laser['vy'])
                
                # Cool down the laser heat
                laser['heat'] = max(0, laser['heat'] - 5)
                
                # CRITICAL FIX: Decrease laser life so they don't get stuck forever!
                # Defaults to 2 frames for instant beams if no life is set.
                laser['life'] = laser.get('life', 2) - 1
                if laser['life'] <= 0:
                    laser['active'] = False

        self._check_collisions()

        # Purge dead entities
        self.bullets = [b for b in self.bullets if b['life'] > 0]
        self.lasers  = [l for l in self.lasers  if l['active']]

    def _check_collisions(self):
        active_lasers = [l for l in self.lasers if l['active']]
        vaporised = set()

        # Rule 7: Bullet × Laser
        for bi, b in enumerate(self.bullets):
            if b['life'] <= 0: continue
            for laser in active_lasers:
                if self._point_line_dist((b['x'], b['y']), laser['p1'], laser['p2']) < 15.0:
                    b['life'] = 0
                    vaporised.add(bi)
                    break

        # Rule 6: Laser × Laser heat
        for i, la in enumerate(active_lasers):
            for j, lb in enumerate(active_lasers):
                if j <= i: continue
                if self._segments_intersect(la['p1'], la['p2'], lb['p1'], lb['p2']):
                    # Spike heat instantly to 255 because fast-moving beams 
                    # only overlap for 1-2 frames!
                    la['heat'] = 255
                    lb['heat'] = 255

        new_bullets: list[dict] = []
        new_lasers:  list[dict] = []

        for bi, b in enumerate(self.bullets):
            if b['life'] <= 0 or bi in vaporised: continue
            for shape in self.shapes.values():
                if not shape['type']: continue
                self._apply_bullet_shape_rule(b, shape, new_bullets)

        for laser in active_lasers:
            for shape in self.shapes.values():
                if not shape['type']: continue
                self._apply_laser_shape_rule(laser, shape, new_lasers)

        self.bullets.extend(new_bullets)
        self.lasers.extend(new_lasers)

    def _apply_bullet_shape_rule(self, b: dict, shape: dict, new_bullets: list):
        cx, cy, r = shape['cx'], shape['cy'], shape['r']
        dist = math.hypot(b['x'] - cx, b['y'] - cy)
        if dist >= r: return

        stype = shape['type']
        if stype == 'Box':
            b['life'] = 0
                
        elif stype == 'Pyramid':
            # Prevent the newly spawned bullet from getting stuck in an infinite loop
            if b.get('is_refracted'):
                return

            # Destroy the incoming bullet
            b['life'] = 0
            speed = math.hypot(b['vx'], b['vy']) or 10.0

            # Spawn a single bullet at the top of the pyramid going straight up
            new_bullets.append({
                'x': cx, 
                'y': cy - r - 5,  # Subtracting 'r' and 5 puts it just above the top tip
                'vx': 0.0, 
                'vy': -speed,     # Negative Y velocity moves it straight up
                'life': 60,
                'is_refracted': True
            })

        elif stype == 'Cone':
            if dist < 1e-6: return
            nx = (b['x'] - cx) / dist
            ny = (b['y'] - cy) / dist
            dot = b['vx'] * nx + b['vy'] * ny
            b['vx'] -= 2.0 * dot * nx
            b['vy'] -= 2.0 * dot * ny
            b['x'] = cx + nx * (r + 1)
            b['y'] = cy + ny * (r + 1)
        elif stype == 'Cylinder':
            b['life'] = 0
            speed = math.hypot(b['vx'], b['vy']) or 10.0
            new_bullets.append({'x': cx, 'y': cy - r, 'vx': 0, 'vy': -speed, 'life': 60})
            new_bullets.append({'x': cx, 'y': cy + r, 'vx': 0, 'vy':  speed, 'life': 60})
        elif stype == 'Sphere':
            b['life'] = 0
            self.sphere_state['water_level'] = min(100.0, self.sphere_state['water_level'] + 10.0)

    def _apply_laser_shape_rule(self, laser: dict, shape: dict, new_lasers: list):
        # 1. Short-circuit if deactivated or already collided
        # The 'has_collided' flag prevents frozen incident lasers from spawning infinite duplicates
        if not laser.get('active', True) or laser.get('has_collided', False):
            return

        cx, cy, r = shape['cx'], shape['cy'], shape['r']
        dist = self._point_line_dist((cx, cy), laser['p1'], laser['p2'])
        
        if dist >= r:
            return

        stype = shape['type']

        # 2. Capture initial velocity and direction for all shapes
        vx = laser.get('vx', 0.0)
        vy = laser.get('vy', 0.0)
        speed = math.hypot(vx, vy)
        
        # Fallback for static continuous beams
        if speed < 1e-6:
            vx = laser['p2'][0] - laser['p1'][0]
            vy = laser['p2'][1] - laser['p1'][1]
            speed = math.hypot(vx, vy)
            if speed < 1e-6:
                return
                
        dx, dy = vx / speed, vy / speed
        
        # 3. Default approximate hit point (surface of the collider)
        hit_x = cx - dx * r
        hit_y = cy - dy * r

        # 4. Apply shape-specific logic
        if stype == 'Box':          
            laser['p2'] = (hit_x, hit_y)
            laser['vx'] = 0.0
            laser['vy'] = 0.0
            laser['has_collided'] = True

        elif stype == 'Pyramid':    
            # Stop incoming laser at the surface
            laser['p2'] = (hit_x, hit_y)
            laser['vx'] = 0.0
            laser['vy'] = 0.0
            laser['has_collided'] = True
            
            current_width = laser.get('width', 1.0)
            
            # Spawn new laser coming out of the top
            new_lasers.append({
                'p1': (cx, cy - r - 5),
                'p2': (cx, cy - r - 155),
                'color': laser.get('color', (255, 255, 255)),
                'heat': 0, 
                'active': True,
                'vx': 0.0, 
                'vy': -speed, 
                'life': laser.get('life', 25),
                'width': current_width * 1.5 
            })

        elif stype == 'Cone':       
            # 1. Get the normalized incident direction (V)
            vx = laser['p2'][0] - laser['p1'][0]
            vy = laser['p2'][1] - laser['p1'][1]
            
            # Fallback if laser is frozen
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                vx, vy = laser.get('vx', 0.0), laser.get('vy', 0.0)
                
            speed = math.hypot(vx, vy)
            if speed < 1e-6:
                return

            dx, dy = vx / speed, vy / speed

            # 2. Ray-Circle Intersection to find the EXACT hit point
            fx = laser['p1'][0] - cx
            fy = laser['p1'][1] - cy

            b = fx * dx + fy * dy
            c = (fx * fx + fy * fy) - r * r
            disc = b * b - c

            # If disc >= 0, the laser intersects the circle
            if disc >= 0:
                t = -b - math.sqrt(disc)
                
                # If t < 0, the origin is inside the shape. Clamp it.
                if t < 0: 
                    t = 0 
                
                hit_x = laser['p1'][0] + t * dx
                hit_y = laser['p1'][1] + t * dy

                # 3. Calculate the true Normal (N) at the exact surface hit point
                nx = hit_x - cx
                ny = hit_y - cy
                n_len = math.hypot(nx, ny)
                
                if n_len > 1e-6:
                    nx /= n_len
                    ny /= n_len
                else:
                    nx, ny = -dx, -dy # Fallback

                # 4. Reflection formula: R = V - 2(V * N)N
                dot = dx * nx + dy * ny
                rx = dx - 2.0 * dot * nx
                ry = dy - 2.0 * dot * ny

                # 5. Stop the incident laser exactly at the surface
                laser['p2'] = (hit_x, hit_y)
                laser['vx'] = 0.0
                laser['vy'] = 0.0
                laser['has_collided'] = True

                # 6. Spawn the reflected laser (offset slightly to prevent getting stuck)
                new_lasers.append({
                    'p1': (hit_x + rx * 2.0, hit_y + ry * 2.0),
                    'p2': (hit_x + rx * 155.0, hit_y + ry * 155.0),
                    'color': laser.get('color', (255, 255, 255)),
                    'heat': 0,
                    'active': True,
                    'vx': rx * speed,
                    'vy': ry * speed,
                    'life': laser.get('life', 25)
                })

        elif stype == 'Cylinder':   
            # Stop incoming laser at the surface
            laser['p2'] = (hit_x, hit_y)
            laser['vx'] = 0.0
            laser['vy'] = 0.0
            laser['has_collided'] = True

            new_lasers.append({
                'p1': (cx, cy - r - 5),
                'p2': (cx, cy - r - 155),
                'color': laser.get('color', (255, 255, 255)), 
                'heat': 0, 'active': True,
                'vx': 0, 'vy': -speed, 'life': laser.get('life', 25)
            })
            new_lasers.append({
                'p1': (cx, cy + r + 5),
                'p2': (cx, cy + r + 155),
                'color': laser.get('color', (255, 255, 255)), 
                'heat': 0, 'active': True,
                'vx': 0, 'vy': speed, 'life': laser.get('life', 25)
            })

        elif stype == 'Sphere':     
            shape['glowing'] = True
            
            # The laser charges the sphere, causing it to grow as a solid!
            # We allow it to exceed 100.0 up to 150.0 so it visibly expands past its outline.
            self.sphere_state['water_level'] = min(150.0, self.sphere_state['water_level'] + 4.0)
            
            # Stop incoming laser at the surface
            laser['p2'] = (hit_x, hit_y)
            laser['vx'] = 0.0
            laser['vy'] = 0.0
            laser['has_collided'] = True

    def _point_line_dist(self, pt: tuple, start: tuple, end: tuple) -> float:
        px, py = pt
        ax, ay = start
        bx, by = end
        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod

    def _segments_intersect(p1: tuple, p2: tuple, p3: tuple, p4: tuple) -> bool:
        def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        def on_segment(p, q, r): return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)): return True
        if d1 == 0 and on_segment(p3, p1, p4): return True
        if d2 == 0 and on_segment(p3, p2, p4): return True
        if d3 == 0 and on_segment(p1, p3, p2): return True
        if d4 == 0 and on_segment(p1, p4, p2): return True
        return False

# ================================================================
#  HOLOGRAM EFFECT
# ================================================================
class HologramEffect:

    def _draw_iron_man_glove(self, img, lms, w, h, side):
        if lms is None or len(lms) == 0: return img
        # --- Iron Man Palette (BGR format) ---
        armor_red = (40, 40, 200)    
        armor_gold = (50, 200, 255)  
        repulsor_cyan = (255, 255, 200)
        shadow_red = (15, 15, 120)

        # Convert normalized LMS to pixel coordinates
        pts = [(int(lm[0]*w), int(lm[1]*h)) for lm in lms]
        overlay = np.zeros_like(img)

        # 1. Draw the Palm / Back-of-hand Plate
        # Landmarks 0 (wrist), 5, 9, 13, 17 (knuckles) make a perfect hand-plate
        palm_pts = np.array([pts[0], pts[17], pts[13], pts[9], pts[5]], np.int32)
        
        # Fill plate and add golden trim
        cv2.fillPoly(overlay, [palm_pts], shadow_red)
        cv2.polylines(overlay, [palm_pts], True, armor_gold, 2, cv2.LINE_AA)

        # 2. Draw Finger Mechanical Segments
        fingers = [
            (1, 2, 3, 4),       # Thumb
            (5, 6, 7, 8),       # Index
            (9, 10, 11, 12),    # Middle
            (13, 14, 15, 16),   # Ring
            (17, 18, 19, 20)    # Pinky
        ]

        for finger in fingers:
            for i in range(len(finger) - 1):
                p1, p2 = pts[finger[i]], pts[finger[i+1]]
                
                # Outer armor shell (Thick Red)
                cv2.line(overlay, p1, p2, armor_red, 12, cv2.LINE_AA)
                # Inner mechanical skeleton (Thin Gold)
                cv2.line(overlay, p1, p2, armor_gold, 3, cv2.LINE_AA)
                # Joint nodes
                cv2.circle(overlay, p1, 5, armor_gold, -1, cv2.LINE_AA)

            # Emitter nodes at fingertips
            cv2.circle(overlay, pts[finger[-1]], 6, repulsor_cyan, -1, cv2.LINE_AA)

        # 3. Draw the Main Palm Repulsor
        # Calculate the center of the palm
        cx = sum([p[0] for p in palm_pts]) // 5
        cy = sum([p[1] for p in palm_pts]) // 5

        # If palm is open, charge up the repulsor
        if self._is_palm_open(lms):
            pulse = int(18 + 5 * math.sin(self.frame_count * 0.3))
            cv2.circle(overlay, (cx, cy), pulse, repulsor_cyan, -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), pulse + 6, armor_gold, 2, cv2.LINE_AA)
        else:
            # Back of hand glow / standby mode
            cv2.circle(overlay, (cx, cy), 12, armor_gold, -1, cv2.LINE_AA)

        # 4. Composite with Glow Magic
        glow = cv2.GaussianBlur(overlay, (15, 15), 0)
        img = cv2.addWeighted(img, 1.0, overlay, 0.85, 0)
        img = cv2.addWeighted(img, 1.0, glow, 0.6, 0)

        return img

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

        # Initialize the physics engine
        self.physics = ScenePhysicsEngine()

    @staticmethod
    def _new_hand_state():
        return dict(
            ui_active=False,
            panel_open=False,
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

    def _draw_3d_wireframe(self, img, shape, cx, cy, radius, palm_cx, palm_cy, glowing=False, water_level=0.0):
        angle_y    = self.frame_count * 0.032
        tilt_x     = 0.45

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

        # Handle glowing state and water levels from physics engine
        current_holo_blue = self.holo_blue
        pulse_r = int(14 + 4 * math.sin(self.frame_count * 0.12))

        if glowing:
            current_holo_blue = (min(255, self.holo_blue[0]+100), min(255, self.holo_blue[1]+100), min(255, self.holo_blue[2]+100))
            pulse_r += 10

        beam_color = (current_holo_blue[0]//4, current_holo_blue[1]//4, current_holo_blue[2]//4)
        cv2.circle(overlay, (palm_cx, palm_cy), pulse_r, current_holo_blue, 2, cv2.LINE_AA)
        cv2.circle(overlay, (palm_cx, palm_cy), 5, (255,255,255), -1)
        if len(pts) > 0:
            for pt in [pts[0], pts[len(pts)//2]]:
                cv2.line(overlay, (palm_cx, palm_cy), pt, beam_color, 1, cv2.LINE_AA)

        for p1_idx, p2_idx in lines:
            pt1 = pts[p1_idx]; pt2 = pts[p2_idx]
            cv2.line(overlay, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)
            cv2.line(overlay, pt1, pt2, current_holo_blue, 3, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(overlay, pt, 3, (255,255,255), -1, cv2.LINE_AA)

        if shape == "Sphere" and water_level > 0:
            solid_r = int(radius * (water_level / 100.0))
            # Main solid glowing color
            cv2.circle(overlay, (cx, cy), solid_r, current_holo_blue, -1, cv2.LINE_AA)
            # Add an intense white energy core in the center
            if solid_r > 12:
                cv2.circle(overlay, (cx, cy), int(solid_r * 0.7), (255, 255, 255), -1, cv2.LINE_AA)

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
    #  WEAPONS RENDERERS 
    # ================================================================

    def _render_lasers(self, img, lasers, h, w):
        overlay = np.zeros_like(img)
        time_offset = self.frame_count * 0.5
        
        for l in lasers:
            start_pt = (int(l['p1'][0]), int(l['p1'][1]))
            end_pt = (int(l['p2'][0]), int(l['p2'][1]))
            
            base_color = l['color']
            heat = l.get('heat', 0)
            
            # Thickness grows with heat
            thickness = 8 + int((heat / 255.0) * 20) 
            dist = math.hypot(end_pt[0]-start_pt[0], end_pt[1]-start_pt[1])
            
            if dist <= 1:
                continue

            dx = (end_pt[0]-start_pt[0]) / dist
            dy = (end_pt[1]-start_pt[1]) / dist

            # ==========================================
            #  DRAW THE BEAM (GRADIENT IF HOT)
            # ==========================================
            if heat > 50:
                # Slicing the line into 5-pixel chunks to draw the gradient
                segment_len = 5 
                for i in range(0, int(dist), segment_len):
                    pA = (int(start_pt[0] + dx * i), int(start_pt[1] + dy * i))
                    pB = (int(start_pt[0] + dx * min(i+segment_len, dist)), int(start_pt[1] + dy * min(i+segment_len, dist)))
                    
                    # Generate a shifting hue based on position, time, and heat
                    # (OpenCV Hue goes from 0-179)
                    hue = int((i * 1.5 - self.frame_count * 10) % 180)
                    
                    # Convert HSV to BGR for OpenCV
                    hsv = np.uint8([[[hue, 255, 255]]])
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                    spectrum_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                    
                    cv2.line(overlay, pA, pB, (255,255,255), 2, cv2.LINE_AA)
                    cv2.line(overlay, pA, pB, spectrum_color, thickness, cv2.LINE_AA)
            else:
                # Normal solid color when cool
                color = (
                    min(255, base_color[0] + heat),
                    min(255, base_color[1] + heat),
                    min(255, base_color[2] + heat)
                )
                cv2.line(overlay, start_pt, end_pt, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(overlay, start_pt, end_pt, color, thickness, cv2.LINE_AA)
            
            # ==========================================
            #  DRAW THE SWIRLS
            # ==========================================
            step = max(8, int(dist/50))
            for i in range(0, int(dist), step):
                swirl_radius = 15 + int((heat / 255.0) * 15)
                swirl_x = math.cos(i*0.1 + time_offset + (heat/50.0)) * swirl_radius
                swirl_y = math.sin(i*0.1 + time_offset + (heat/50.0)) * swirl_radius
                
                nx, ny = -dy, dx
                fx = int(start_pt[0] + dx * i + nx * swirl_x)
                fy = int(start_pt[1] + dy * i + ny * swirl_y)
                
                if 0 <= fx < w and 0 <= fy < h:
                    cv2.circle(overlay, (fx, fy), 2, (255,255,255), -1)
                    
                    # Make swirls match the rainbow gradient if hot!
                    if heat > 50:
                        hue = int((i * 1.5 - self.frame_count * 10) % 180)
                        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                        pt_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                    else:
                        pt_color = color
                        
                    cv2.circle(overlay, (fx, fy), 5, pt_color, -1)
                        
        glow = cv2.GaussianBlur(overlay, (21, 21), 0)
        img  = cv2.add(img, overlay)
        img  = cv2.add(img, glow)
        return img

    def _render_bullets(self, img, bullets):
        overlay = np.zeros_like(img)
        for p in bullets:
            p1 = (int(p['x']), int(p['y']))
            p2 = (int(p['x']-p['vx']*0.5), int(p['y']-p['vy']*0.5))
            cv2.line(overlay, p1, p2, (255,255,255), 2, cv2.LINE_AA)
            cv2.line(overlay, p1, p2, self.gun_yellow, 6, cv2.LINE_AA)
            cv2.circle(overlay, p1, 4, (255,255,255), -1)
            
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

        for side, hs, bar_x in [('right', rhs,    14),
                                ('left',  lhs,    w - 14 - BAR_LEN)]:
            now = time.time()
            if not hs['ui_active'] and hs['palm_open_start'] > 0:
                # MATCH THIS TO YOUR 1.5 SECOND LOGIC TIMER
                prog   = min(1.0, (now - hs['palm_open_start']) / 1.5) 
                filled = int(BAR_LEN * prog)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+BAR_LEN, BAR_Y+BAR_TH), (40,30,10), -1)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+filled,  BAR_Y+BAR_TH), (0,200,170), -1)
                cv2.putText(img, f"{'L' if side=='left' else 'R'} OPEN {int(prog*100)}%",
                            (bar_x, BAR_Y+BAR_TH+12), self.font, 0.30, (0,200,170), 1, cv2.LINE_AA)
            elif hs['ui_active'] and hs['fist_closed_start'] > 0:
                # MATCH THIS TO YOUR 1.5 SECOND LOGIC TIMER
                prog   = min(1.0, (now - hs['fist_closed_start']) / 1.5) 
                filled = int(BAR_LEN * prog)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+BAR_LEN, BAR_Y+BAR_TH), (20,10,30), -1)
                cv2.rectangle(img, (bar_x, BAR_Y), (bar_x+filled,  BAR_Y+BAR_TH), (0,60,220), -1)
                cv2.putText(img, f"{'R' if side=='left' else 'L'} OFF {int(prog*100)}%",
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
            elif now - hs['palm_open_start'] >= 1.5:
                hs['ui_active']       = True
                hs['panel_open']      = True  
                hs['palm_open_start'] = 0
        else:
            hs['palm_open_start'] = 0

        if fist and hs['ui_active']:
            if hs['fist_closed_start'] == 0:
                hs['fist_closed_start'] = now
            elif now - hs['fist_closed_start'] >= 1.5:
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


        left_lms  = getattr(tracking_data, 'left_hand_landmarks',  None) if tracking_data else None
        right_lms = getattr(tracking_data, 'right_hand_landmarks', None) if tracking_data else None

        output = self._draw_iron_man_glove(output, left_lms, w, h, 'left')
        output = self._draw_iron_man_glove(output, right_lms, w, h, 'right')

        self._update_hand_state(left_lms,  'left',  h, w)
        self._update_hand_state(right_lms, 'right', h, w)

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
                scaled_holo = cv2.convertScaleAbs(holo_person, alpha=breathe)
                holo_masked = cv2.bitwise_and(scaled_holo, scaled_holo, mask=cleaned_mask)
                output       = cv2.add(output, holo_masked)

        # 3. Update gesture state
      

        # 4. Draw UI panels
        if self.hand_state['left']['panel_open']:
            output = self._draw_main_ui(output, left_lms,  h, w, side='left')
        if self.hand_state['right']['panel_open']:
            output = self._draw_main_ui(output, right_lms, h, w, side='right')

        # 5. Extract inputs for physics engine
        hands_data = {}
        fired_bullets = []
        fired_lasers = []

        for lms, side in [(left_lms, 'left'), (right_lms, 'right')]:
            hs  = self.hand_state[side]
            obj = hs['selected_object']
            if obj is None or lms is None:
                continue

            palm_cx = int(lms[9][0] * w)
            palm_cy = int(lms[9][1] * h)

            if obj in ["Box", "Sphere", "Pyramid", "Cone", "Cylinder"]:
                if self._is_palm_up(lms):
                    bob_offset = math.sin(self.frame_count * 0.07) * 10
                    obj_cy = int((palm_cy - 200) + bob_offset)
                    hands_data[side] = {
                        'type': obj, 
                        'cx': palm_cx, 
                        'cy': obj_cy, 
                        'r': 70.0,
                        'palm_cx': palm_cx,
                        'palm_cy': palm_cy
                    }

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
                            fired_lasers.append({'p1': index_tip, 'p2': end, 'color': self.laser_blue})
                            cv2.circle(output, index_tip, 4, (255,255,255), -1)
                        else:
                            if self.frame_count % 4 == 0:
                                fired_bullets.append({'x': index_tip[0], 'y': index_tip[1], 'vx': dx*30, 'vy': dy*30, 'life': 20})
                                cv2.circle(output, index_tip, 6, self.gun_yellow, -1)

        # 6. Step Physics
        self.physics.register_input(hands_data, fired_bullets, fired_lasers)
        self.physics.update()

        # 7. Render Physics State
        for side, shape_data in self.physics.shapes.items():
            if shape_data['type'] is not None:
                glowing = shape_data.get('glowing', False)
                water_level = self.physics.sphere_state['water_level'] if shape_data['type'] == 'Sphere' else 0.0
                
                output = self._draw_3d_wireframe(
                    output, 
                    shape_data['type'], 
                    int(shape_data['cx']), 
                    int(shape_data['cy']), 
                    int(shape_data['r']), 
                    int(shape_data.get('palm_cx', shape_data['cx'])), 
                    int(shape_data.get('palm_cy', shape_data['cy'])),
                    glowing, 
                    water_level
                )
        output = self._render_lasers(output, self.physics.lasers, h, w)
        output = self._render_bullets(output, self.physics.bullets)

        # 8. Status strip + loading bars
        output = self._draw_status_and_loading(output, h, w)

        return output

    def close(self):
        pass

if __name__ == "__main__":
    print("Run this via main.py to pass tracking_data into the processor.")