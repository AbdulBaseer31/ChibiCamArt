import cv2
import numpy as np
import random
from old_tracker import TrackingData

class TerminalEffect:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.char_width = 8 
        self.char_height = 18 
        
        self.canvas = None
        self.rows = 0
        
        # Typing state
        self.current_text = ""
        self.target_text = ""
        self.cursor_x = 0
        
        # Arch Linux / Linux Kernel source snippets
        self.source_code = [
            "void __init setup_arch(char **cmdline_p) {",
            "    memblock_reserve(__pa(_stext), _end - _stext);",
            "    arch_mem_init(cmdline_p);",
            "    paging_init();",
            "}",
            "static int __ref kernel_init(void *unused) {",
            "    system_state = SYSTEM_SCHEDULING;",
            "    numa_default_policy();",
            "    if (ramdisk_execute_command) {",
            "        run_init_process(ramdisk_execute_command);",
            "    }",
            "    panic(\"No working init found. Try passing init= bootarg.\");",
            "}",
            "int handle_mm_fault(struct vm_area_struct *vma, unsigned long address) {",
            "    __set_current_state(TASK_RUNNING);",
            "    count_vm_event(PGFAULT);",
            "    return __handle_mm_fault(vma, address, flags);",
            "}",
            "pkgname=linux && pkgver=6.1.arch1-1",
            "arch=('x86_64') && license=('GPL2')",
            "source=(\"https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-$pkgver.tar.xz\")",
            "make bzImage modules && make INSTALL_MOD_PATH=\"$pkgdir\" modules_install",
            "pacman -Syu --noconfirm",
            "mkinitcpio -p linux",
            "grub-mkconfig -o /boot/grub/grub.cfg"
        ]

    def _get_new_line(self, w):
        target_chars = int(w / self.char_width)
        line = random.choice(self.source_code)
        # Ensure full line width with kernel-style padding
        while len(line) < target_chars:
            line += " " + random.choice(["0x%08x" % random.randint(0, 0xFFFFFFFF), "[OK]", "ptr", ">>"])
        return line[:target_chars]

    def _draw_terminal_rain(self, h, w):
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.rows = int(h / self.char_height)
            self.target_text = self._get_new_line(w)

        # REMOVED cv2.addWeighted (No more fading/disappearing text)

        typing_y = (self.rows) * self.char_height - 5
        
        # High-speed typing: 3-7 chars per frame
        chars_this_frame = random.randint(3, 7)
        for _ in range(chars_this_frame):
            if len(self.current_text) < len(self.target_text):
                char = self.target_text[len(self.current_text)]
                self.current_text += char
                
                # Render in permanent Gray (128, 128, 128)
                cv2.putText(self.canvas, char, (self.cursor_x, typing_y), 
                            self.font, 0.3, (128, 128, 128), 1, cv2.LINE_AA)
                self.cursor_x += self.char_width
            else:
                # Scroll logic: Move the entire buffer UP
                shift = self.char_height
                # Shift current canvas pixels up
                self.canvas[:h-shift, :] = self.canvas[shift:h, :]
                # Black out only the newly exposed bottom area
                self.canvas[h-shift:h, :] = 0 
                
                # Reset for the next line
                self.current_text = ""
                self.target_text = self._get_new_line(w)
                self.cursor_x = 0
                break

        # Blinking Cursor in Gray
        if random.random() > 0.5:
            cv2.rectangle(self.canvas, (self.cursor_x, typing_y - 10), 
                          (self.cursor_x + 6, typing_y), (128, 128, 128), -1)
        else:
            # Overwrite cursor with black when "off" to prevent cursor trailing
            cv2.rectangle(self.canvas, (self.cursor_x, typing_y - 10), 
                          (self.cursor_x + 6, typing_y), (0, 0, 0), -1)

    def process_frame(self, frame: np.ndarray, tracking_data: TrackingData = None) -> np.ndarray:
        h, w = frame.shape[:2]
        output = frame.copy()
        
        self._draw_terminal_rain(h, w)

        if tracking_data is not None and tracking_data.has_person:
            if tracking_data.segmentation_mask is not None:
                mask = tracking_data.segmentation_mask
                if len(mask.shape) > 2: mask = mask.squeeze()
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                is_person = mask > 128
                output[is_person] = self.canvas[is_person]
                
                # Boundary in (200, 200, 200)
                edges = cv2.Canny(mask, 100, 200)
                output[edges > 0] = (200, 200, 200)

        # Simplified UI buttons
        self._draw_interactive_buttons(output, tracking_data, h, w)
        return output

    def _draw_interactive_buttons(self, frame, tracking_data, h, w):
        if tracking_data and tracking_data.has_person:
            for hand_lms, label in [(tracking_data.left_hand_landmarks, "RUN"), 
                                    (tracking_data.right_hand_landmarks, "STOP")]:
                if hand_lms is not None:
                    x = int(hand_lms[9][0] * w)
                    y = int(hand_lms[9][1] * h)
                    cv2.circle(frame, (x, y), 35, (128, 128, 128), 1, cv2.LINE_AA)
                    cv2.putText(frame, label, (x - 15, y + 5), self.font, 0.4, (128, 128, 128), 1)

    def close(self):
        pass