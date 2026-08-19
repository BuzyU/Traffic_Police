"""
calibration_ui.py — Interactive Click-to-Draw Calibration Tool (OpenCV GUI).

Allows user to interactively click and draw:
  - Stop line: Click 2 points to define the line
  - Direction vector: Click start and end to define allowed flow direction

Usage:
    py -3.11 part2_detection/calibration_ui.py --video path/to/video.mp4
"""

import argparse
import cv2
import numpy as np


class CalibrationGUI:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.points = []
        self.mode = "stop_line"  # "stop_line" or "direction"
        self.stop_line = None
        self.allowed_direction = None

        self.cap = cv2.VideoCapture(video_path)
        ret, self.first_frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame from {video_path}")
        self.cap.release()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"[{self.mode.upper()}] Point captured: ({x}, {y})")

            if len(self.points) == 2:
                if self.mode == "stop_line":
                    self.stop_line = (self.points[0], self.points[1])
                    print(f"✔ Stop line set: {self.stop_line}")
                    self.mode = "direction"
                    self.points = []
                    print("\nNow click 2 points to set ALLOWED DIRECTION (start -> end)")
                elif self.mode == "direction":
                    self.allowed_direction = (self.points[0], self.points[1])
                    print(f"✔ Allowed direction set: {self.allowed_direction}")
                    self.points = []
                    print("\nCalibration finished! Press 'q' or 'ESC' to save and exit.")

    def run(self):
        window_name = "Calibration: Click 2 points for Stop Line, then 2 for Direction"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("=" * 60)
        print("  INTERACTIVE TRAFFIC CALIBRATION TOOL")
        print("=" * 60)
        print("1. Click 2 points on the video frame for STOP LINE.")
        print("2. Click 2 points for ALLOWED DIRECTION (start -> end).")
        print("3. Press 'r' to reset, 'q' to finish.")
        print("=" * 60)

        while True:
            display = self.first_frame.copy()

            # Draw current points being clicked
            for pt in self.points:
                cv2.circle(display, pt, 5, (0, 255, 255), -1)

            # Draw stop line if set
            if self.stop_line:
                cv2.line(display, self.stop_line[0], self.stop_line[1], (0, 0, 255), 3)
                cv2.putText(display, "STOP LINE", self.stop_line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw direction vector if set
            if self.allowed_direction:
                cv2.arrowedLine(display, self.allowed_direction[0], self.allowed_direction[1], (0, 255, 0), 3, tipLength=0.2)
                cv2.putText(display, "ALLOWED FLOW", self.allowed_direction[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Instruction header banner
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (20, 20, 20), -1)
            inst = f"Current task: Set {self.mode.replace('_', ' ').upper()} (click 2 points). Press 'r' to reset, 'q' to save."
            cv2.putText(display, inst, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.points = []
                self.stop_line = None
                self.allowed_direction = None
                self.mode = "stop_line"
                print("Reset all calibration.")

        cv2.destroyAllWindows()
        return {
            "stop_line": self.stop_line,
            "allowed_direction": self.allowed_direction
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration UI")
    parser.add_argument("--video", required=True, help="Video path to calibrate on")
    args = parser.parse_args()

    gui = CalibrationGUI(args.video)
    res = gui.run()
    print("\nFinal calibration coordinates:")
    print(res)
