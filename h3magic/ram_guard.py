# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "psutil"]
# ///
"""
RAM guard for the H3 Magic project (m00912): "kill anything immediately, even one millisecond,
if it ever goes above 75% RAM, so that you don't risk crashing."

Every POLL_SECONDS it reads /proc/meminfo. When used RAM (MemTotal - MemAvailable) passes
LIMIT_FRACTION of MemTotal, it SIGKILLs the user process with the largest resident memory
(never a PROTECTED process) and logs it, then checks again at once.
Run: uv run --script h3magic/ram_guard.py   (in tmux session ram_guard)
"""
import os
import time

import fire
import psutil

LIMIT_FRACTION = 0.75  # The user's line: 75% of system RAM
POLL_SECONDS = 0.001   # "even one millisecond"
PROTECTED = ("mount.objectivefs", "/claude/versions/", "tmux", "sshd", "systemd", "proxyd", "jupyter-lab", "webterm", "nflx-otel")  # The OS, the /root mount and the agent itself


def used_fraction(meminfo_text):
    """
    Pure function. Fraction of RAM in use, from the text of /proc/meminfo: 1 - MemAvailable / MemTotal.

    Args:
        meminfo_text (str): Contents of /proc/meminfo

    Returns:
        float

    Examples:
        >>> used_fraction("MemTotal: 1000 kB\\nMemFree: 100 kB\\nMemAvailable: 200 kB\\n")
        0.8
    """
    fields = dict(line.split(":", 1) for line in meminfo_text.splitlines() if ":" in line)
    total, available = (int(fields[k].split()[0]) for k in ("MemTotal", "MemAvailable"))
    return 1 - available / total


def is_protected(program, pid):
    """
    Pure function. Whether a process must never be killed: pid 1, the guard itself, or a program path holding a PROTECTED part.

    Args:
        program (str): The process's executable path (its first command-line word)
        pid (int)

    Examples:
        >>> is_protected("/sbin/mount.objectivefs", 1891), is_protected("/root/.local/share/claude/versions/2.1.329", 8763)
        (True, True)
        >>> is_protected("/root/miniconda3/bin/python3", 4242)  # training, encoding, servers: killable
        False
    """
    return pid in (1, os.getpid()) or any(p in program for p in PROTECTED)


def biggest_process():
    """Query. The non-protected process with the largest resident memory, or None."""
    candidates = [p for p in psutil.process_iter(["name", "memory_info", "cmdline"])
                  if p.info["memory_info"] and not is_protected((p.info["cmdline"] or [p.info["name"] or ""])[0], p.pid)]
    return max(candidates, key=lambda p: p.info["memory_info"].rss, default=None)


def main(log="agents/.scratchpad/logs/ram_guard_h3magic.log"):
    """Command. Loops forever; SIGKILLs the biggest process whenever used RAM passes LIMIT_FRACTION, and logs each kill."""
    print(f"ram_guard: limit {LIMIT_FRACTION:.0%} of RAM, poll {POLL_SECONDS * 1000:.0f} ms, log {log}", flush=True)
    with open("/proc/meminfo") as meminfo:
        while True:
            meminfo.seek(0)
            fraction = used_fraction(meminfo.read())
            if fraction > LIMIT_FRACTION:
                victim = biggest_process()
                line = f"{time.strftime('%I:%M:%S %p')} used {fraction:.1%} > {LIMIT_FRACTION:.0%}: " + (
                    f"SIGKILL pid {victim.pid} rss {victim.info['memory_info'].rss / 2**30:.1f} GiB {' '.join(victim.info['cmdline'] or [])[:200]}"
                    if victim else "no process to kill")
                print(line, flush=True)
                with open(log, "a") as f:
                    f.write(line + "\n")
                if victim:
                    victim.kill()
                continue
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    fire.Fire(main)
