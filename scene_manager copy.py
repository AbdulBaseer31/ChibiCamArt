import math
import time
import numpy as np


class ScenePhysicsEngine:
    """
    2D/3D physics and state management engine for an OpenCV/MediaPipe AR application.
    Handles entity lifecycle, movement, and collision detection.
    No rendering logic (cv2.imshow) is included here.
    """

    def __init__(self):
        # --- Entity Stores ---
        self.bullets: list[dict] = []
        # Each bullet: {'x': float, 'y': float, 'vx': float, 'vy': float, 'life': int}

        self.lasers: list[dict] = []
        # Each laser: {'p1': (x,y), 'p2': (x,y), 'color': (B,G,R), 'heat': int, 'active': bool}

        self.shapes: dict = {
            'left':  {'type': None, 'cx': 0.0, 'cy': 0.0, 'r': 70.0, 'glowing': False},
            'right': {'type': None, 'cx': 0.0, 'cy': 0.0, 'r': 70.0, 'glowing': False},
        }

        self.sphere_state: dict = {
            'water_level': 0.0,   # 0.0 – 100.0
            'last_update': time.time(),
        }

        self._last_time: float = time.time()

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def register_input(self, hands_data: dict, fired_bullets: list, fired_lasers: list):
        """
        Called every frame by the main CV script.

        hands_data  – dict with keys 'left' / 'right', each containing:
                      {'type': str, 'cx': float, 'cy': float, 'r': float}
                      Pass None (or omit a key) if that hand is not present.

        fired_bullets – list of new bullet dicts to add this frame.
        fired_lasers  – list of new laser dicts to add this frame.
        """
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
        """
        Main per-frame update:
          1. Compute delta time.
          2. Decay sphere water level.
          3. Move bullets, decrement life.
          4. Cool laser heat.
          5. Check all collisions.
        """
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # --- Sphere water leak ---
        ws = self.sphere_state
        ws['water_level'] = max(0.0, ws['water_level'] - 2.5 * dt)
        ws['last_update'] = now

        # --- Move bullets ---
        for b in self.bullets:
            b['x'] += b['vx']
            b['y'] += b['vy']
            b['life'] -= 1

        # --- Cool lasers ---
        for laser in self.lasers:
            if laser['active']:
                laser['heat'] = max(0, laser['heat'] - 5)

        # --- Collision detection ---
        self._check_collisions()

        # --- Purge dead entities ---
        self.bullets = [b for b in self.bullets if b['life'] > 0]
        self.lasers  = [l for l in self.lasers  if l['active']]

    # ================================================================
    #  COLLISION DETECTION
    # ================================================================

    def _check_collisions(self):
        """
        Applies Rules 1-7 in order.
        Rule 7 (laser vaporises bullet) is checked first so vaporised
        bullets are not re-tested against shapes.
        """
        active_lasers = [l for l in self.lasers if l['active']]

        # Rule 7: Bullet × Laser — bullet is vaporised, checked FIRST
        vaporised = set()
        for bi, b in enumerate(self.bullets):
            if b['life'] <= 0:
                continue
            for laser in active_lasers:
                if self._point_line_dist((b['x'], b['y']), laser['p1'], laser['p2']) < 15.0:
                    b['life'] = 0
                    vaporised.add(bi)
                    break

        # Rule 6: Laser × Laser heat
        for i, la in enumerate(active_lasers):
            for j, lb in enumerate(active_lasers):
                if j <= i:
                    continue
                if self._segments_intersect(la['p1'], la['p2'], lb['p1'], lb['p2']):
                    la['heat'] = min(255, la['heat'] + 50)
                    lb['heat'] = min(255, lb['heat'] + 50)

        # Rules 1–5: bullets and lasers vs shapes
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
        cx, cy, r = shape['cx'], shape['cy'], shape['r']
        dist = math.hypot(b['x'] - cx, b['y'] - cy)
        if dist >= r:
            return

        stype = shape['type']

        if stype == 'Box':          # Rule 1
            b['life'] = 0

        elif stype == 'Pyramid':    # Rule 2
            b['life'] = 0
            angle_step = (2 * math.pi) / 7
            for k in range(7):
                angle = k * angle_step
                speed = math.hypot(b['vx'], b['vy']) or 10.0
                new_bullets.append({
                    'x': cx, 'y': cy,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': 60,
                })

        elif stype == 'Cone':       # Rule 3 – reflection
            if dist < 1e-6:
                return
            nx = (b['x'] - cx) / dist
            ny = (b['y'] - cy) / dist
            dot = b['vx'] * nx + b['vy'] * ny
            b['vx'] -= 2.0 * dot * nx
            b['vy'] -= 2.0 * dot * ny
            # Push bullet outside the shape to avoid re-collision
            b['x'] = cx + nx * (r + 1)
            b['y'] = cy + ny * (r + 1)

        elif stype == 'Cylinder':   # Rule 4
            b['life'] = 0
            speed = math.hypot(b['vx'], b['vy']) or 10.0
            new_bullets.append({'x': cx, 'y': cy - r, 'vx': 0, 'vy': -speed, 'life': 60})
            new_bullets.append({'x': cx, 'y': cy + r, 'vx': 0, 'vy':  speed, 'life': 60})

        elif stype == 'Sphere':     # Rule 5
            b['life'] = 0
            self.sphere_state['water_level'] = min(
                100.0, self.sphere_state['water_level'] + 10.0
            )

    def _apply_laser_shape_rule(self, laser: dict, shape: dict, new_lasers: list):
        cx, cy, r = shape['cx'], shape['cy'], shape['r']
        dist = self._point_line_dist((cx, cy), laser['p1'], laser['p2'])
        if dist >= r:
            return

        stype = shape['type']

        if stype == 'Box':          # Rule 1
            laser['active'] = False

        elif stype == 'Pyramid':    # Rule 2 – rainbow split
            laser['active'] = False
            rainbow_colors = [
                (148, 0, 211),   # Violet  (B,G,R converted)
                ( 75, 0, 130),   # Indigo
                (255, 0,   0),   # Blue
                (  0,255,  0),   # Green
                (  0,255,255),   # Yellow
                (  0,165,255),   # Orange
                (  0,  0,255),   # Red
            ]
            # Compute incoming direction
            lx = laser['p2'][0] - laser['p1'][0]
            ly = laser['p2'][1] - laser['p1'][1]
            base_angle = math.atan2(ly, lx)
            arc = math.pi / 2          # 90-degree arc
            angle_step = arc / 6 if len(rainbow_colors) > 1 else 0
            start_angle = base_angle - arc / 2
            length = 800.0
            for k, color in enumerate(rainbow_colors):
                angle = start_angle + k * angle_step
                ex = cx + math.cos(angle) * length
                ey = cy + math.sin(angle) * length
                new_lasers.append({
                    'p1': (cx, cy),
                    'p2': (ex, ey),
                    'color': color,
                    'heat': 0,
                    'active': True,
                })

        elif stype == 'Cone':       # Rule 3 – reflection
            # Reflect the laser endpoint direction around the surface normal
            nx = cx - (laser['p1'][0] + laser['p2'][0]) / 2
            ny = cy - (laser['p1'][1] + laser['p2'][1]) / 2
            n_len = math.hypot(nx, ny)
            if n_len < 1e-6:
                return
            nx /= n_len; ny /= n_len
            # Outward normal from shape centre
            nx, ny = -nx, -ny
            lx = laser['p2'][0] - laser['p1'][0]
            ly = laser['p2'][1] - laser['p1'][1]
            dot = lx * nx + ly * ny
            rlx = lx - 2 * dot * nx
            rly = ly - 2 * dot * ny
            laser['active'] = False
            new_lasers.append({
                'p1': (cx, cy),
                'p2': (cx + rlx, cy + rly),
                'color': laser['color'],
                'heat': 0,
                'active': True,
            })

        elif stype == 'Cylinder':   # Rule 4
            laser['active'] = False
            new_lasers.append({
                'p1': (cx, cy - r),
                'p2': (cx, cy - r - 800),
                'color': laser['color'],
                'heat': 0,
                'active': True,
            })
            new_lasers.append({
                'p1': (cx, cy + r),
                'p2': (cx, cy + r + 800),
                'color': laser['color'],
                'heat': 0,
                'active': True,
            })

        elif stype == 'Sphere':     # Rule 5
            shape['glowing'] = True

    # ================================================================
    #  GEOMETRY HELPERS
    # ================================================================

    def _point_line_dist(self, pt: tuple, start: tuple, end: tuple) -> float:
        """
        Shortest distance from point `pt` to the line *segment* [start, end].
        Uses vector projection; clamps to segment endpoints.
        """
        px, py = pt
        ax, ay = start
        bx, by = end

        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby

        if ab_len_sq < 1e-12:
            # Degenerate segment – treat as a point
            return math.hypot(px - ax, py - ay)

        # Project pt onto the line, clamp t ∈ [0, 1]
        t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))

        closest_x = ax + t * abx
        closest_y = ay + t * aby
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def _segments_intersect(p1: tuple, p2: tuple, p3: tuple, p4: tuple) -> bool:
        """
        Returns True if line segment p1–p2 intersects segment p3–p4.
        Uses the cross-product orientation test.
        """
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