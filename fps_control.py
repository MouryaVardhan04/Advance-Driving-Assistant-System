"""Simple frame timing helpers to synchronize playback to source FPS.

Provides a FrameTimer that maps frame indices to presentation timestamps
based on a source_fps and optionally enforces a target playback rate.

This is lightweight and avoids busy-waiting; it will sleep to present frames
at the correct times and can drop frames if the processing is behind.
"""
import time


class FrameTimer:
    """Keep track of expected presentation time for frames from a source.

    Usage:
        timer = FrameTimer(source_fps=video_fps)
        timer.start()
        while True:
            # read frame
            timer.wait_for_frame()

    The timer uses the source_fps to compute when frame N should be shown:
        t_expected = N / source_fps

    If processing is behind, by default it will advance the internal index
    to catch up (frame drop). If you set allow_frame_drop=False it will not
    drop frames (but playback will stutter).
    """

    def __init__(self, source_fps: float = 30.0, allow_frame_drop: bool = True):
        self.source_fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
        self.allow_frame_drop = allow_frame_drop
        self.start_time = None
        self.frame_index = 0

    def start(self):
        self.start_time = time.time()
        self.frame_index = 0

    def reset(self):
        self.start_time = None
        self.frame_index = 0

    def wait_for_frame(self):
        """Block (sleep) until the next frame should be presented.

        Returns True if we slept to present the next frame (on-time),
        False if we were behind and advanced the index to catch up.
        """
        if self.start_time is None:
            self.start()

        expected_t = self.frame_index / self.source_fps
        now = time.time() - self.start_time
        wait = expected_t - now

        if wait > 0:
            # We're ahead of schedule: sleep until presentation time
            time.sleep(wait)
            self.frame_index += 1
            return True
        else:
            # We're behind schedule
            if self.allow_frame_drop:
                # Jump frame_index forward to current time to catch up
                self.frame_index = int(now * self.source_fps) + 1
                return False
            else:
                # Do not drop frames; present immediately and advance
                self.frame_index += 1
                return False
