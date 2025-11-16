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
TARGET_FPS = 50  # Set your desired FPS cap here (None for unlimited)
# ============================================================================


class PerformanceMonitor:
    """Simple real-time performance monitor."""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        
        # Timing metrics
        self.frame_times = deque(maxlen=window_size)
        self.tracker_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        
        # System metrics
        self.process = psutil.Process()
        self.memory_mb = 0
        self.last_cpu_times = None
        self.cpu_time_ms = 0  # CPU time for current frame
        
        # Frame counter
        self.frame_count = 0
        self.last_frame_start = None
        
    def record_frame_start(self):
        """Record the start of a new frame and measure CPU time."""
        current_time = perf_counter()
        
        # Calculate time since last frame
        if self.last_frame_start is not None:
            frame_duration = current_time - self.last_frame_start
            self.frame_times.append(frame_duration)
        
        self.last_frame_start = current_time
        self.frame_count += 1
        
        # Measure CPU time for this frame
        try:
            cpu_times = self.process.cpu_times()
            if self.last_cpu_times is not None:
                # CPU time used for this frame
                cpu_delta = (cpu_times.user - self.last_cpu_times.user + 
                           cpu_times.system - self.last_cpu_times.system)
                self.cpu_time_ms = cpu_delta * 1000  # Convert to ms
            self.last_cpu_times = cpu_times
        except:
            pass
        
        # Update memory less frequently (it's expensive)
        if self.frame_count % 10 == 0:
            try:
                mem_info = self.process.memory_info()
                self.memory_mb = mem_info.rss / 1024 / 1024
            except:
                pass
    
    def record_tracker_time(self, duration):
        """Record tracker algorithm time only."""
        self.tracker_times.append(duration)
    
    def record_processing_time(self, duration):
        """Record total processing time."""
        self.processing_times.append(duration)
    
    def get_fps(self):
        """Calculate current FPS."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = np.mean(list(self.frame_times))
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_tracker_time_ms(self):
        """Get average tracker time in milliseconds."""
        if not self.tracker_times:
            return 0.0
        return np.mean(list(self.tracker_times)) * 1000
    
    def get_processing_time_ms(self):
        """Get average processing time in milliseconds."""
        if not self.processing_times:
            return 0.0
        return np.mean(list(self.processing_times)) * 1000
    
    def get_cpu_time_ms(self):
        """Get CPU time for current frame in milliseconds."""
        return self.cpu_time_ms
    
    def draw_metrics(self, frame, target_fps=None):
        """Draw performance metrics on the frame (top-left corner)."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (340, 145), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        fps = self.get_fps()
        tracker_time = self.get_tracker_time_ms()
        proc_time = self.get_processing_time_ms()
        cpu_time = self.get_cpu_time_ms()
        y_offset = 30
        line_height = 25
        
        # FPS
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        fps_text = f"FPS: {fps:.1f}"
        if target_fps:
            fps_text += f" / {target_fps}"
        cv2.putText(frame, fps_text, (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        y_offset += line_height
        
        # Tracker time (just the algorithm)
        tracker_color = (0, 255, 0) if tracker_time < 15 else (0, 255, 255) if tracker_time < 30 else (0, 0, 255)
        cv2.putText(frame, f"Algorithm: {tracker_time:.1f}ms", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracker_color, 2)
        y_offset += line_height
        
        # Total processing time
        proc_color = (0, 255, 0) if proc_time < 20 else (0, 255, 255) if proc_time < 40 else (0, 0, 255)
        cv2.putText(frame, f"Pipeline: {proc_time:.1f}ms", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, proc_color, 2)
        y_offset += line_height
        
        # CPU time
        cpu_color = (255, 255, 255)
        cv2.putText(frame, f"CPU Work: {cpu_time:.1f}ms", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cpu_color, 2)
        y_offset += line_height
        
        # Memory
        cv2.putText(frame, f"Mem: {self.memory_mb:.0f} MB", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def print_realtime_stats(self, target_fps=None):
        """Print simple one-line real-time stats to CLI."""
        fps = self.get_fps()
        tracker_time = self.get_tracker_time_ms()
        proc_time = self.get_processing_time_ms()
        cpu_time = self.get_cpu_time_ms()
        
        fps_text = f"FPS: {fps:6.1f}"
        if target_fps:
            fps_text += f"/{target_fps:<2}"
        else:
            fps_text += "   "
            
        print(f"{fps_text} | Algorithm: {tracker_time:5.1f}ms | "
              f"Pipeline: {proc_time:5.1f}ms | CPU Work: {cpu_time:5.1f}ms | "
              f"Mem: {self.memory_mb:6.0f}MB", 
              end="\r", flush=True)


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
    
    print("\n" + "="*80)
    print("IR PLANE TRACKER - Performance Monitoring")
    print("="*80)
    if TARGET_FPS:
        print(f"FPS Cap: {TARGET_FPS} FPS (target frame time: {target_frame_duration*1000:.1f}ms)")
    else:
        print("FPS Cap: Unlimited")
    print("\nMetrics:")
    print("  - Algorithm: Time spent in tracker algorithm only")
    print("  - Pipeline:  Time for entire pipeline (capture+undistort+track+visualize)")
    print("  - CPU Work:  Total CPU computation time used across all cores")
    print("\nControls: q=Quit | s=Save frame")
    print("-"*80 + "\n")
    
    frame_counter = 1006
    
    try:
        while True:
            loop_start = perf_counter()
            
            # Capture and undistort
            frame = cam.get_frame()
            img = frame.bgr
            img = cv2.undistort(img, camera_matrix, dist_coeffs)
            
            # Track the plane - measure this specifically
            tracker_start = perf_counter()
            screen_corners = tracker(img)
            tracker_end = perf_counter()
            monitor.record_tracker_time(tracker_end - tracker_start)

            # Visualization is now controlled by the 'debug' flag in the params JSON
            if tracker.params.debug:
                # Add metrics to the tracker's debug image
                if tracker.debug and tracker.debug.img_raw is not None:
                    tracker.debug.img_raw = monitor.draw_metrics(
                        tracker.debug.img_raw, TARGET_FPS
                    )

                # Show debug visualization
                if tracker.debug:
                    tracker.debug.visualize()

            # Calculate total processing time (before sleep)
            loop_end = perf_counter()
            total_processing = loop_end - loop_start
            monitor.record_processing_time(total_processing)
            
            # Print real-time stats
            monitor.print_realtime_stats(TARGET_FPS)
            
            # Record frame timing (for FPS calculation)
            monitor.record_frame_start()
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            # FPS capping: sleep to reach target frame time
            if target_frame_duration:
                remaining_time = target_frame_duration - total_processing
                if remaining_time > 0:
                    sleep(remaining_time)
            
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
        print(f"Average tracker time: {monitor.get_tracker_time_ms():.1f}ms")
        print(f"Average processing time: {monitor.get_processing_time_ms():.1f}ms")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()