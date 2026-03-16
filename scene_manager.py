import math
import time
import numpy as np

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
            'water_level': 0.0,   
            'last_update': time.time(),
        }
        self._last_time: float = time.time()

    def register_input(self, hands_data: dict, fired_bullets: list, fired_lasers: list):
        for side in ('left', 'right'):
            info = (hands_data or {}).get(side)
            if info:
                self.shapes[side].update(info)
                self.shapes[side]['glowing'] = False 
            else:
                self.shapes[side]['type'] = None

        for b in (fired_bullets or []):
            self.bullets.append(dict(b))

        for laser in (fired_lasers or []):
            entry = dict(laser)
            
            px, py = entry['p1']
            ex, ey = entry['p2']
            dist = math.hypot(ex - px, ey - py)
            
            if dist > 0:
                dx, dy = (ex - px) / dist, (ey - py) / dist
                entry['p2'] = (px + dx * 150, py + dy * 150) 
                entry['vx'] = dx * 70  
                entry['vy'] = dy * 70
            
            entry.setdefault('heat', 0)
            entry.setdefault('active', True)
            entry.setdefault('life', 25) 
            
            self.lasers.append(entry)

    def update(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        ws = self.sphere_state
        ws['water_level'] = max(0.0, ws['water_level'] - 2.5 * dt)
        ws['last_update'] = now

        for b in self.bullets:
            b['x'] += b['vx']
            b['y'] += b['vy']
            b['life'] -= 1

        for laser in self.lasers:
            if laser['active']:
                if 'vx' in laser and 'vy' in laser:
                    laser['p1'] = (laser['p1'][0] + laser['vx'], laser['p1'][1] + laser['vy'])
                    laser['p2'] = (laser['p2'][0] + laser['vx'], laser['p2'][1] + laser['vy'])
                
                laser['heat'] = max(0, laser['heat'] - 5)
                laser['life'] = laser.get('life', 30) - 1
                if laser['life'] <= 0:
                    laser['active'] = False

        self._check_collisions()

        self.bullets = [b for b in self.bullets if b['life'] > 0][:50]  # Hard cap — prevents runaway growth
        self.lasers  = [l for l in self.lasers  if l['active']]

    def _check_collisions(self):
        active_lasers = [l for l in self.lasers if l['active']]
        vaporised = set()

        for bi, b in enumerate(self.bullets):
            if b['life'] <= 0:
                continue
            for laser in active_lasers:
                if self._point_line_dist((b['x'], b['y']), laser['p1'], laser['p2']) < 15.0:
                    b['life'] = 0
                    vaporised.add(bi)
                    break

        for i, la in enumerate(active_lasers):
            for j, lb in enumerate(active_lasers):
                if j <= i:
                    continue
                if self._segments_intersect(la['p1'], la['p2'], lb['p1'], lb['p2']):
                    la['heat'] = min(255, la['heat'] + 50)
                    lb['heat'] = min(255, lb['heat'] + 50)

        new_bullets: list[dict] = []
        new_lasers:  list[dict] = []

        for bi, b in enumerate(self.bullets):
            if b['life'] <= 0 or bi in vaporised:
                continue
            for shape in self.shapes.values():
                if not shape['type']:
                    continue
                self._apply_bullet_shape_rule(b, shape, new_bullets)

        for laser in active_lasers:
            for shape in self.shapes.values():
                if not shape['type']:
                    continue
                self._apply_laser_shape_rule(laser, shape, new_lasers)

        self.bullets.extend(new_bullets)
        self.lasers.extend(new_lasers)

    def _apply_bullet_shape_rule(self, b: dict, shape: dict, new_bullets: list):
        # 1. Short-circuit: If the bullet was already destroyed by another shape this frame, stop.
        if b.get('life', 0) <= 0:
            return

        cx, cy, r = shape['cx'], shape['cy'], shape['r']
        dist = math.hypot(b['x'] - cx, b['y'] - cy)
        
        if dist >= r:
            return

        stype = shape['type']

        if stype == 'Box':          
            b['life'] = 0

        elif stype == 'Pyramid':
            if b.get('is_refracted'):
                return

            # Destroy the incoming bullet — it does not pass through
            b['life'] = 0

            speed = math.hypot(b['vx'], b['vy'])
            if speed < 1e-6:
                return

            base_angle = math.atan2(b['vy'], b['vx'])

            # Spawn exit point on the far surface of the shape
            exit_x = cx + math.cos(base_angle) * (r + 4)
            exit_y = cy + math.sin(base_angle) * (r + 4)

            # Two diverging beams — wider spread so it visually reads as light splitting
            spread = math.pi / 4   # 45° total fan (±22.5°)
            for offset in (-spread / 2, spread / 2):
                angle = base_angle + offset
                new_bullets.append({
                    'x': exit_x,
                    'y': exit_y,
                    'vx': math.cos(angle) * speed * 0.85,
                    'vy': math.sin(angle) * speed * 0.85,
                    'life': 18,
                    'scale': 0.15,
                    'is_refracted': True,
                })

        elif stype == 'Cone':       
            if dist < 1e-6:
                return
            nx = (b['x'] - cx) / dist
            ny = (b['y'] - cy) / dist
            dot = b['vx'] * nx + b['vy'] * ny
            b['vx'] -= 2.0 * dot * nx
            b['vy'] -= 2.0 * dot * ny
            b['x'] = cx + nx * (r + 5) 
            b['y'] = cy + ny * (r + 5)

        elif stype == 'Cylinder':   
            b['life'] = 0
            speed = math.hypot(b['vx'], b['vy']) or 10.0
            new_bullets.append({'x': cx, 'y': cy - r - 5, 'vx': 0, 'vy': -speed, 'life': 40, 'scale': 0.5})
            new_bullets.append({'x': cx, 'y': cy + r + 5, 'vx': 0, 'vy':  speed, 'life': 40, 'scale': 0.5})

        elif stype == 'Sphere':     
            b['life'] = 0
            self.sphere_state['water_level'] = min(
                100.0, self.sphere_state['water_level'] + 10.0
            )

    def _apply_laser_shape_rule(self, laser: dict, shape: dict, new_lasers: list):
        # Short-circuit if another shape already killed this laser
        if not laser.get('active', True):
            return

        cx, cy, r = shape['cx'], shape['cy'], shape['r']
        dist = self._point_line_dist((cx, cy), laser['p1'], laser['p2'])
        if dist >= r:
            return

        stype = shape['type']

        if stype == 'Box':          
            laser['active'] = False

        elif stype == 'Pyramid':    
            if laser.get('is_refracted'):
                return
                
            # DESTROY original laser
            laser['active'] = False
            
            rainbow_colors = [
                (148, 0, 211), (75, 0, 130), (255, 0, 0), (0, 255, 0),
                (0, 255, 255), (0, 165, 255), (0, 0, 255),
            ]
            
            lx = laser['p2'][0] - laser['p1'][0]
            ly = laser['p2'][1] - laser['p1'][1]
            speed = math.hypot(lx, ly)
            
            if speed < 1e-6:
                return

            # Spawn from the exact impact point (p1)
            start_x = laser['p1'][0]
            start_y = laser['p1'][1]

            base_angle = math.atan2(ly, lx)
            spread = math.pi / 2          
            angle_step = spread / (len(rainbow_colors) - 1) if len(rainbow_colors) > 1 else 0
            start_angle = base_angle - spread / 2
            
            for k, color in enumerate(rainbow_colors):
                angle = start_angle + k * angle_step
                dx = math.cos(angle)
                dy = math.sin(angle)
                
                new_lasers.append({
                    'p1': (start_x, start_y),
                    'p2': (start_x + dx * 100, start_y + dy * 100),
                    'color': color,
                    'heat': 0,
                    'active': True,
                    'vx': dx * 50, # Slower visual speed to ease rendering
                    'vy': dy * 50,
                    'life': 10,    # Fast fade-out
                    'is_refracted': True 
                })

        elif stype == 'Cone':       
            nx = cx - (laser['p1'][0] + laser['p2'][0]) / 2
            ny = cy - (laser['p1'][1] + laser['p2'][1]) / 2
            n_len = math.hypot(nx, ny)
            if n_len < 1e-6:
                return
            nx /= n_len; ny /= n_len
            nx, ny = -nx, -ny
            
            lx = laser['p2'][0] - laser['p1'][0]
            ly = laser['p2'][1] - laser['p1'][1]
            dot = lx * nx + ly * ny
            rlx = lx - 2 * dot * nx
            rly = ly - 2 * dot * ny
            laser['active'] = False
            
            dist_new = math.hypot(rlx, rly)
            if dist_new > 0:
                dx, dy = rlx/dist_new, rly/dist_new
                start_x = cx + dx * (r + 5)
                start_y = cy + dy * (r + 5)
                
                new_lasers.append({
                    'p1': (start_x, start_y),
                    'p2': (start_x + dx * 150, start_y + dy * 150),
                    'color': laser['color'],
                    'heat': 0,
                    'active': True,
                    'vx': dx * 70,
                    'vy': dy * 70,
                    'life': 25
                })

        elif stype == 'Cylinder':   
            laser['active'] = False
            new_lasers.append({
                'p1': (cx, cy - r - 5),
                'p2': (cx, cy - r - 155),
                'color': laser['color'], 'heat': 0, 'active': True,
                'vx': 0, 'vy': -70, 'life': 25
            })
            new_lasers.append({
                'p1': (cx, cy + r + 5),
                'p2': (cx, cy + r + 155),
                'color': laser['color'], 'heat': 0, 'active': True,
                'vx': 0, 'vy': 70, 'life': 25
            })

        elif stype == 'Sphere':     
            shape['glowing'] = True

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
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        if d1 == 0 and on_segment(p3, p1, p4): return True
        if d2 == 0 and on_segment(p3, p2, p4): return True
        if d3 == 0 and on_segment(p1, p3, p2): return True
        if d4 == 0 and on_segment(p1, p4, p2): return True

        return False