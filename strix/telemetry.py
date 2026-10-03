#!/usr/bin/env python3
"""Sample Strix Halo resource use while a model runs; no root needed.

Decodes the amdgpu gpu_metrics v3.0 blob (APU: GFX activity, DRAM traffic,
socket/GFX/core power, temperatures, clocks, throttle residency) and adds
CPU utilization, NVMe read throughput and GTT use from sysfs/procfs.

  strix/telemetry.py [--interval S] [--duration S] [--csv FILE]

Prints one line per sample and a summary (mean / max) at the end.
"""
import argparse, ctypes, glob, os, sys, time

U16, U32, U64 = ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64


class GpuMetricsV30(ctypes.Structure):
    # struct gpu_metrics_v3_0, drivers/gpu/drm/amd/include/kgd_pp_interface.h
    _fields_ = [
        ("structure_size", U16), ("format_revision", ctypes.c_uint8), ("content_revision", ctypes.c_uint8),
        ("temperature_gfx", U16), ("temperature_soc", U16), ("temperature_core", U16 * 16),
        ("temperature_skin", U16),
        ("average_gfx_activity", U16), ("average_vcn_activity", U16), ("average_ipu_activity", U16 * 8),
        ("average_core_c0_activity", U16 * 16), ("average_dram_reads", U16), ("average_dram_writes", U16),
        ("average_ipu_reads", U16), ("average_ipu_writes", U16),
        ("system_clock_counter", U64),
        ("average_socket_power", U32), ("average_ipu_power", U16), ("average_apu_power", U32),
        ("average_gfx_power", U32), ("average_dgpu_power", U32), ("average_all_core_power", U32),
        ("average_core_power", U16 * 16), ("average_sys_power", U16), ("stapm_power_limit", U16),
        ("current_stapm_power_limit", U16),
        ("average_gfxclk_frequency", U16), ("average_socclk_frequency", U16),
        ("average_vpeclk_frequency", U16), ("average_ipuclk_frequency", U16),
        ("average_fclk_frequency", U16), ("average_vclk_frequency", U16),
        ("average_uclk_frequency", U16), ("average_mpipu_frequency", U16),
        ("current_coreclk", U16 * 16), ("current_core_maxfreq", U16), ("current_gfx_maxfreq", U16),
        ("throttle_residency_prochot", U32), ("throttle_residency_spl", U32),
        ("throttle_residency_fppt", U32), ("throttle_residency_sppt", U32),
        ("throttle_residency_thm_core", U32), ("throttle_residency_thm_gfx", U32),
        ("throttle_residency_thm_soc", U32), ("time_filter_alphavalue", U32),
    ]


def find_card():
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if os.path.exists(d + "/gpu_metrics") and os.path.basename(os.path.dirname(d)).count("-") == 0:
            return d
    sys.exit("no amdgpu gpu_metrics found")


def read_metrics(card):
    raw = open(card + "/gpu_metrics", "rb").read()
    m = GpuMetricsV30.from_buffer_copy(raw.ljust(ctypes.sizeof(GpuMetricsV30), b"\0"))
    if (m.format_revision, m.content_revision) != (3, 0) or m.structure_size != ctypes.sizeof(GpuMetricsV30):
        sys.exit(f"unsupported gpu_metrics {m.format_revision}.{m.content_revision} "
                 f"size {m.structure_size} (decoder expects 3.0 / {ctypes.sizeof(GpuMetricsV30)})")
    return m


def hwmon(name, field):
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(h + "/name").read().strip() == name:
                return int(open(f"{h}/{field}").read())
        except OSError:
            pass
    return None


def cpu_times():
    f = open("/proc/stat").readline().split()[1:]
    v = list(map(int, f))
    idle = v[3] + v[4]
    return sum(v), idle


def disk_read_sectors(dev="nvme0n1"):
    for line in open("/proc/diskstats"):
        p = line.split()
        if p[2] == dev:
            return int(p[5])
    return 0


def valid(x, bad=0xFFFF):
    return None if x == bad else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--csv")
    a = ap.parse_args()
    card = find_card()
    gtt = card + "/mem_info_gtt_used"
    cols = ["t", "gfx_busy", "dram_rd_mbs", "dram_wr_mbs", "socket_w", "gfx_w", "cores_w",
            "gfx_mhz", "fclk_mhz", "uclk_mhz", "gfx_maxmhz", "core_maxmhz", "cpu_busy",
            "core_c0_avg", "tctl_c", "gfx_c", "nvme_c", "nvme_rd_mbs", "gtt_gib",
            "thr_thm_core", "thr_thm_gfx", "thr_spl", "thr_fppt", "thr_sppt"]
    out = open(a.csv, "w") if a.csv else None
    if out:
        out.write(",".join(cols) + "\n")
    rows = []
    t0 = time.time()
    c_prev, d_prev = cpu_times(), disk_read_sectors()
    m_prev = read_metrics(card)
    while time.time() - t0 < a.duration:
        time.sleep(a.interval)
        m = read_metrics(card)
        c, d = cpu_times(), disk_read_sectors()
        dt_cpu = c[0] - c_prev[0]
        cpu_busy = 100.0 * (1 - (c[1] - c_prev[1]) / dt_cpu) if dt_cpu else 0.0
        nvme_mbs = (d - d_prev) * 512 / 1e6 / a.interval
        thr = lambda f: getattr(m, f) - getattr(m_prev, f)
        c0 = [x for x in m.average_core_c0_activity if x != 0xFFFF]
        row = {
            "t": round(time.time() - t0, 1),
            "gfx_busy": valid(m.average_gfx_activity),
            "dram_rd_mbs": valid(m.average_dram_reads), "dram_wr_mbs": valid(m.average_dram_writes),
            "socket_w": m.average_socket_power / 1000, "gfx_w": m.average_gfx_power / 1000,
            "cores_w": m.average_all_core_power / 1000,
            "gfx_mhz": valid(m.average_gfxclk_frequency), "fclk_mhz": valid(m.average_fclk_frequency),
            "uclk_mhz": valid(m.average_uclk_frequency), "gfx_maxmhz": valid(m.current_gfx_maxfreq),
            "core_maxmhz": valid(m.current_core_maxfreq),
            "cpu_busy": round(cpu_busy, 1), "core_c0_avg": round(sum(c0) / len(c0), 1) if c0 else None,
            "tctl_c": (hwmon("k10temp", "temp1_input") or 0) / 1000,
            "gfx_c": valid(m.temperature_gfx) / 100 if valid(m.temperature_gfx) else None,
            "nvme_c": (hwmon("nvme", "temp1_input") or 0) / 1000,
            "nvme_rd_mbs": round(nvme_mbs, 1),
            "gtt_gib": round(int(open(gtt).read()) / 2**30, 2),
            "thr_thm_core": thr("throttle_residency_thm_core"), "thr_thm_gfx": thr("throttle_residency_thm_gfx"),
            "thr_spl": thr("throttle_residency_spl"), "thr_fppt": thr("throttle_residency_fppt"),
            "thr_sppt": thr("throttle_residency_sppt"),
        }
        rows.append(row)
        line = ",".join("" if row[k] is None else str(row[k]) for k in cols)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
        c_prev, d_prev, m_prev = c, d, m
    print("\nsummary (mean / max):")
    for k in cols[1:]:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        if vals:
            print(f"  {k:14} {sum(vals) / len(vals):10.1f} {max(vals):10.1f}")


if __name__ == "__main__":
    main()
