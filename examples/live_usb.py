from collections import deque
from time import perf_counter, sleep

import cv2
import numpy as np
import psutil

from pupil_labs.ir_plane_tracker.tracker import (
    Tracker,
    TrackerParams,
)


# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_FPS = 30  # Set your desired FPS cap here (None for unlimited)
# ============================================================================


class PerformanceMonitor:
    """Simple real-time performance monitor."""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        
        # Timing metrics
        self.frame_times = deque(maxlen=window_size)
        
        # System metrics
        self.process = psutil.Process()
        self.cpu_percent = 0
        self.memory_mb = 0
        
        # Frame counter
        self.frame_count = 0
        
    def update_metrics(self):
        """Update CPU and memory usage."""
        try:
            # Process-specific CPU usage (not system-wide)
            self.cpu_percent = self.process.cpu_percent(interval=0)
            mem_info = self.process.memory_info()
            self.memory_mb = mem_info.rss / 1024 / 1024
        except:
            pass
    
    def record_frame_time(self, duration):
        """Record frame processing time."""
        self.frame_times.append(duration)
        self.frame_count += 1
        
        # Update system metrics every 10 frames (to reduce overhead)
        if self.frame_count % 10 == 0:
            self.update_metrics()
    
    def get_fps(self):
        """Calculate current FPS."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = np.mean(list(self.frame_times))
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def draw_metrics(self, frame, target_fps=None):
        """Draw performance metrics on the frame (top-left corner)."""
        # Semi-transparent background
        overlay = frame.copy()
        height = 120 if target_fps else 95
        cv2.rectangle(overlay, (5, 5), (300, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        fps = self.get_fps()
        y_offset = 30
        line_height = 25
        
        # FPS - Green if >20, Yellow if >10, Red if <10
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        fps_text = f"FPS: {fps:.1f}"
        if target_fps:
            fps_text += f" / {target_fps}"
        cv2.putText(frame, fps_text, (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        y_offset += line_height
        
        # CPU - Green if <50%, Yellow if <80%, Red if >80%
        cpu_color = (0, 255, 0) if self.cpu_percent < 50 else (0, 255, 255) if self.cpu_percent < 80 else (0, 0, 255)
        cv2.putText(frame, f"CPU: {self.cpu_percent:.1f}%", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cpu_color, 2)
        y_offset += line_height
        
        # Memory - always white for now
        cv2.putText(frame, f"Mem: {self.memory_mb:.0f} MB", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def print_realtime_stats(self, target_fps=None):
        """Print simple one-line real-time stats to CLI."""
        fps = self.get_fps()
        fps_text = f"FPS: {fps:6.1f}"
        if target_fps:
            fps_text += f" / {target_fps:<3}"
        print(f"{fps_text} | CPU: {self.cpu_percent:5.1f}% | Memory: {self.memory_mb:7.1f} MB | Frames: {self.frame_count}", end="\r", flush=True)


def main():
    from common.camera import HDDigitalCam

    cam = HDDigitalCam()
    camera_matrix = np.load("resources/camera_matrix.npy")
    dist_coeffs = np.load("resources/dist_coeffs.npy")
    params_json_path = "resources/params_hddigital.json"
    
    params = TrackerParams.from_json(params_json_path)
    tracker = Tracker(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, params=params
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(window_size=30)
    
    # Calculate target frame duration if FPS cap is set
    target_frame_duration = 1.0 / TARGET_FPS if TARGET_FPS else None
    
    print("\n" + "="*70)
    print("IR PLANE TRACKER - Performance Monitoring")
    print("="*70)
    if TARGET_FPS:
        print(f"FPS Cap: {TARGET_FPS} FPS (target frame time: {target_frame_duration*1000:.1f}ms)")
    else:
        print("FPS Cap: Unlimited")
    print("Controls: q=Quit | s=Save frame")
    print("-"*70 + "\n")
    
    frame_counter = 1006
    
    try:
        while True:
            frame_start = perf_counter()
            
            # Capture and undistort
            frame = cam.get_frame()
            img = frame.bgr
            img = cv2.undistort(img, camera_matrix, dist_coeffs)
            
            # Track the plane
            screen_corners = tracker(img)
            
            # Add metrics to the tracker's debug image (shown in all debug windows)
            if tracker.debug.img_raw is not None:
                tracker.debug.img_raw = monitor.draw_metrics(tracker.debug.img_raw, TARGET_FPS)
            
            # Show debug visualization (includes "Tracked Plane" window with our metrics)
            tracker.debug.visualize()
            
            # Record frame time
            frame_end = perf_counter()
            frame_duration = frame_end - frame_start
            monitor.record_frame_time(frame_duration)
            
            # Print real-time stats to CLI
            monitor.print_realtime_stats(TARGET_FPS)
            
            # FPS capping: sleep if we finished processing faster than target
            if target_frame_duration:
                remaining_time = target_frame_duration - frame_duration
                if remaining_time > 0:
                    sleep(remaining_time)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"frame_{frame_counter:04d}.png", img)
                frame_counter += 1
                print(f"\n✅ Saved frame_{frame_counter:04d}.png")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print(f"\n\nTotal frames processed: {monitor.frame_count}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()