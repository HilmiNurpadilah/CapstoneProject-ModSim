# app.py
# Streamlit App — Simulasi Antrean Kasir Minimarket (Single vs Multi Queue)
# Jalankan:
#   streamlit run app.py

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple

# -----------------------------
# Model parameters
# -----------------------------
@dataclass
class ModelParams:
    sim_minutes: int = 480
    warmup_minutes: int = 30

    # Arrival rates (customers/minute)
    lambda_normal: float = 0.6
    lambda_peak1: float = 1.2

    # One-peak window
    peak_start: int = 120
    peak_end: int = 240

    # Two-peak window (optional)
    peak2_start: int = 360
    peak2_end: int = 450
    lambda_peak2: float = 1.1

    # Service time mixture
    p_small: float = 0.70
    median_small_sec: float = 50.0
    median_large_sec: float = 110.0
    sigma_small: float = 0.45
    sigma_large: float = 0.55

    # Disruption
    p_disruption: float = 0.03
    disruption_low_sec: float = 40.0
    disruption_high_sec: float = 90.0


def arrival_rate_one_peak(t_min: float, p: ModelParams) -> float:
    return p.lambda_peak1 if (p.peak_start <= t_min < p.peak_end) else p.lambda_normal


def arrival_rate_two_peak(t_min: float, p: ModelParams) -> float:
    in_peak1 = p.peak_start <= t_min < p.peak_end
    in_peak2 = p.peak2_start <= t_min < p.peak2_end
    if in_peak1:
        return p.lambda_peak1
    if in_peak2:
        return p.lambda_peak2
    return p.lambda_normal


def lognormal_from_median(median_sec: float, sigma: float, rng: np.random.Generator) -> float:
    # median = exp(mu) => mu = ln(median)
    mu = np.log(median_sec)
    return float(rng.lognormal(mean=mu, sigma=sigma))


def sample_service_time_min(p: ModelParams, rng: np.random.Generator) -> float:
    if rng.random() < p.p_small:
        sec = lognormal_from_median(p.median_small_sec, p.sigma_small, rng)
    else:
        sec = lognormal_from_median(p.median_large_sec, p.sigma_large, rng)

    if rng.random() < p.p_disruption:
        sec += rng.uniform(p.disruption_low_sec, p.disruption_high_sec)

    return sec / 60.0


# -----------------------------
# Queue monitor (time-weighted)
# -----------------------------
class QueueMonitor:
    def __init__(self, warmup: int):
        self.warmup = warmup
        self.started = False
        self.last_t = 0.0
        self.last_q = 0
        self.area = 0.0
        self.max_q = 0

    def update(self, t: float, q_len: int):
        if t < self.warmup:
            self.last_t = t
            self.last_q = q_len
            return

        if not self.started:
            self.started = True
            self.last_t = t
            self.last_q = q_len
            self.max_q = q_len
            return

        dt = t - self.last_t
        if dt > 0:
            self.area += self.last_q * dt

        self.last_t = t
        self.last_q = q_len
        self.max_q = max(self.max_q, q_len)

    def average(self, t_end: float) -> float:
        if not self.started:
            return 0.0
        duration = t_end - self.warmup
        return self.area / duration if duration > 0 else 0.0


# -----------------------------
# Simulation (detail)
# -----------------------------
@dataclass
class RunDetail:
    waits: np.ndarray
    systems: np.ndarray
    avg_queue_len: float
    max_queue_len: int
    utilization: float


def simulate_single_queue_detail(
    cashiers_n: int,
    p: ModelParams,
    seed: int,
    arrival_fn: Callable[[float, ModelParams], float],
) -> RunDetail:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    cashiers = simpy.Resource(env, capacity=cashiers_n)

    waits: List[float] = []
    systems: List[float] = []
    monitor = QueueMonitor(p.warmup_minutes)
    busy_time = 0.0

    def customer(_cid: int):
        nonlocal busy_time
        t_arr = env.now
        monitor.update(env.now, len(cashiers.queue))

        with cashiers.request() as req:
            yield req
            t_start = env.now
            monitor.update(env.now, len(cashiers.queue))

            st = sample_service_time_min(p, rng)

            if t_arr >= p.warmup_minutes:
                waits.append(t_start - t_arr)
                if t_start >= p.warmup_minutes:
                    busy_time += st

            yield env.timeout(st)

            t_dep = env.now
            if t_arr >= p.warmup_minutes:
                systems.append(t_dep - t_arr)

            monitor.update(env.now, len(cashiers.queue))

    def arrivals():
        cid = 0
        while env.now < p.sim_minutes:
            lam = arrival_fn(env.now, p)
            inter = rng.exponential(1.0 / lam) if lam > 0 else 1e9
            yield env.timeout(inter)
            cid += 1
            env.process(customer(cid))
            monitor.update(env.now, len(cashiers.queue))

    env.process(arrivals())
    env.run(until=p.sim_minutes)

    obs = max(p.sim_minutes - p.warmup_minutes, 1e-9)
    util = busy_time / (cashiers_n * obs)

    return RunDetail(
        waits=np.array(waits, dtype=float),
        systems=np.array(systems, dtype=float),
        avg_queue_len=float(monitor.average(p.sim_minutes)),
        max_queue_len=int(monitor.max_q),
        utilization=float(util),
    )


def simulate_multi_queue_detail(
    cashiers_n: int,
    p: ModelParams,
    seed: int,
    arrival_fn: Callable[[float, ModelParams], float],
    speed_factors: List[float] | None = None,  # optional: hetero cashiers
) -> RunDetail:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    cashiers = [simpy.Resource(env, capacity=1) for _ in range(cashiers_n)]

    if speed_factors is None:
        speed_factors = [1.0] * cashiers_n
    if len(speed_factors) != cashiers_n:
        raise ValueError("speed_factors length must equal number of cashiers")

    waits: List[float] = []
    systems: List[float] = []
    monitor = QueueMonitor(p.warmup_minutes)
    busy_time = 0.0

    def approx_line_length(r: simpy.Resource) -> int:
        return len(r.queue) + (1 if r.count > 0 else 0)

    def total_waiting() -> int:
        return sum(len(r.queue) for r in cashiers)

    def customer(_cid: int):
        nonlocal busy_time
        t_arr = env.now
        monitor.update(env.now, total_waiting())

        lengths = [approx_line_length(r) for r in cashiers]
        min_len = min(lengths)
        candidates = [i for i, L in enumerate(lengths) if L == min_len]
        idx = int(rng.choice(candidates))
        chosen = cashiers[idx]

        with chosen.request() as req:
            yield req
            t_start = env.now
            monitor.update(env.now, total_waiting())

            st = sample_service_time_min(p, rng) * speed_factors[idx]

            if t_arr >= p.warmup_minutes:
                waits.append(t_start - t_arr)
            if t_start >= p.warmup_minutes:
                busy_time += st

            yield env.timeout(st)

            t_dep = env.now
            if t_arr >= p.warmup_minutes:
                systems.append(t_dep - t_arr)

            monitor.update(env.now, total_waiting())

    def arrivals():
        cid = 0
        while env.now < p.sim_minutes:
            lam = arrival_fn(env.now, p)
            inter = rng.exponential(1.0 / lam) if lam > 0 else 1e9
            yield env.timeout(inter)
            cid += 1
            env.process(customer(cid))
            monitor.update(env.now, total_waiting())

    env.process(arrivals())
    env.run(until=p.sim_minutes)

    obs = max(p.sim_minutes - p.warmup_minutes, 1e-9)
    util = busy_time / (cashiers_n * obs)

    return RunDetail(
        waits=np.array(waits, dtype=float),
        systems=np.array(systems, dtype=float),
        avg_queue_len=float(monitor.average(p.sim_minutes)),
        max_queue_len=int(monitor.max_q),
        utilization=float(util),
    )


# -----------------------------
# Metrics summary
# -----------------------------
def percentiles(x: np.ndarray, ps=(50, 90, 95)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": np.nan for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}


def summarize_runs(runs: List[RunDetail]) -> Dict[str, float]:
    waits = np.concatenate([r.waits for r in runs]) if runs else np.array([])
    systems = np.concatenate([r.systems for r in runs]) if runs else np.array([])

    d = {
        "mean_wait": float(np.mean(waits)) if waits.size else np.nan,
        "mean_system": float(np.mean(systems)) if systems.size else np.nan,
        "avg_queue_len": float(np.mean([r.avg_queue_len for r in runs])) if runs else np.nan,
        "max_queue_len": float(np.mean([r.max_queue_len for r in runs])) if runs else np.nan,
        "utilization": float(np.mean([r.utilization for r in runs])) if runs else np.nan,
        "n_customers": int(waits.size),
    }
    d.update({f"wait_{k}": v for k, v in percentiles(waits).items()})
    d.update({f"system_{k}": v for k, v in percentiles(systems).items()})
    return d


def run_replications(
    model: str,
    cashiers_n: int,
    p: ModelParams,
    arrival_fn: Callable[[float, ModelParams], float],
    n_rep: int,
    seed0: int,
    speed_factors: List[float] | None = None,
) -> List[RunDetail]:
    runs: List[RunDetail] = []
    for i in range(n_rep):
        seed = seed0 + i
        if model == "Single Queue":
            runs.append(simulate_single_queue_detail(cashiers_n, p, seed, arrival_fn))
        else:
            runs.append(simulate_multi_queue_detail(cashiers_n, p, seed, arrival_fn, speed_factors=speed_factors))
    return runs


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Simulasi Antrean Kasir Minimarket", layout="wide")
st.title("Simulasi Antrean Kasir Minimarket (SimPy)")

with st.sidebar:
    st.header("Pengaturan Model")

    model = st.radio("Model antrean", ["Single Queue", "Multi-Queue (Shortest)"], index=0)

    peak_mode = st.radio("Pola jam sibuk", ["One-Peak", "Two-Peak"], index=1)
    arrival_fn = arrival_rate_two_peak if peak_mode == "Two-Peak" else arrival_rate_one_peak

    cashiers_n = st.slider("Jumlah kasir", min_value=1, max_value=8, value=3, step=1)

    st.subheader("Durasi simulasi")
    sim_hours = st.slider("Durasi (jam)", 2, 12, 8, 1)
    warmup_min = st.slider("Warm-up (menit)", 0, 120, 30, 5)

    st.subheader("Kedatangan (customers/min)")
    lam_normal = st.slider("λ normal", 0.1, 3.0, 0.6, 0.1)
    lam_peak1 = st.slider("λ peak 1", 0.1, 5.0, 1.2, 0.1)

    peak1_start = st.slider("Mulai peak 1 (menit)", 0, int(sim_hours * 60), 120, 5)
    peak1_end = st.slider("Selesai peak 1 (menit)", 0, int(sim_hours * 60), 240, 5)

    lam_peak2 = 1.1
    peak2_start = 360
    peak2_end = 450
    if peak_mode == "Two-Peak":
        lam_peak2 = st.slider("λ peak 2", 0.1, 5.0, 1.1, 0.1)
        peak2_start = st.slider("Mulai peak 2 (menit)", 0, int(sim_hours * 60), min(360, int(sim_hours * 60)), 5)
        peak2_end = st.slider("Selesai peak 2 (menit)", 0, int(sim_hours * 60), min(450, int(sim_hours * 60)), 5)

    st.subheader("Waktu layanan (detik)")
    p_small = st.slider("P(belanja sedikit)", 0.1, 0.95, 0.70, 0.05)
    med_small = st.slider("Median 'sedikit' (detik)", 15, 180, 50, 5)
    med_large = st.slider("Median 'banyak' (detik)", 30, 300, 110, 5)
    sigma_small = st.slider("Sigma lognormal 'sedikit'", 0.10, 1.20, 0.45, 0.05)
    sigma_large = st.slider("Sigma lognormal 'banyak'", 0.10, 1.50, 0.55, 0.05)

    st.subheader("Gangguan kecil")
    p_disrupt = st.slider("P(gangguan)", 0.0, 0.20, 0.03, 0.01)
    disrupt_low = st.slider("Tambahan min (detik)", 0, 180, 40, 5)
    disrupt_high = st.slider("Tambahan max (detik)", 0, 300, 90, 5)

    st.subheader("Replikasi")
    n_rep = st.slider("Jumlah replikasi", 5, 60, 20, 5)
    seed0 = st.number_input("Seed awal", min_value=1, value=12345, step=1)

    st.subheader("SLA (opsional)")
    sla_mean = st.slider("Target mean wait ≤ (menit)", 0.5, 10.0, 3.0, 0.5)
    sla_p90 = st.slider("Target P90 wait ≤ (menit)", 1.0, 20.0, 5.0, 0.5)

    st.subheader("Kasir heterogen (opsional)")
    enable_hetero = st.checkbox("Aktifkan 1 kasir cepat (hanya untuk Multi-Queue)", value=False)
    fast_factor = st.slider("Faktor kasir cepat (lebih kecil = lebih cepat)", 0.50, 1.00, 0.85, 0.05)

    run_btn = st.button("Run Simulation", type="primary")

# Build params from UI
p = ModelParams(
    sim_minutes=int(sim_hours * 60),
    warmup_minutes=int(warmup_min),
    lambda_normal=float(lam_normal),
    lambda_peak1=float(lam_peak1),
    peak_start=int(peak1_start),
    peak_end=int(peak1_end),
    peak2_start=int(peak2_start),
    peak2_end=int(peak2_end),
    lambda_peak2=float(lam_peak2),
    p_small=float(p_small),
    median_small_sec=float(med_small),
    median_large_sec=float(med_large),
    sigma_small=float(sigma_small),
    sigma_large=float(sigma_large),
    p_disruption=float(p_disrupt),
    disruption_low_sec=float(disrupt_low),
    disruption_high_sec=float(disrupt_high),
)

# Speed factors (hetero)
speed_factors = None
if model != "Single Queue" and enable_hetero:
    speed_factors = [float(fast_factor)] + [1.0] * (cashiers_n - 1)

# Validate peak windows
if p.peak_end < p.peak_start:
    st.error("Peak 1: 'Selesai' harus ≥ 'Mulai'.")
if peak_mode == "Two-Peak" and p.peak2_end < p.peak2_start:
    st.error("Peak 2: 'Selesai' harus ≥ 'Mulai'.")

# -----------------------------
# Run + Display
# -----------------------------
if run_btn:
    with st.spinner("Menjalankan simulasi..."):
        runs = run_replications(
            model=model,
            cashiers_n=cashiers_n,
            p=p,
            arrival_fn=arrival_fn,
            n_rep=int(n_rep),
            seed0=int(seed0),
            speed_factors=speed_factors,
        )
        s = summarize_runs(runs)

    # SLA checks
    ok_mean = (s["mean_wait"] <= sla_mean)
    ok_p90 = (s["wait_p90"] <= sla_p90)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean wait (min)", f"{s['mean_wait']:.2f}", delta=None)
    col2.metric("P90 wait (min)", f"{s['wait_p90']:.2f}", delta=None)
    col3.metric("P95 wait (min)", f"{s['wait_p95']:.2f}", delta=None)
    col4.metric("Utilization", f"{s['utilization']:.2f}", delta=None)

    st.write(
        f"**SLA mean ≤ {sla_mean:.1f} min:** {'✅' if ok_mean else '❌'}  |  "
        f"**SLA P90 ≤ {sla_p90:.1f} min:** {'✅' if ok_p90 else '❌'}"
    )

    st.caption(
        f"Pelanggan terukur (setelah warm-up): {s['n_customers']} | "
        f"Avg queue length: {s['avg_queue_len']:.2f} | "
        f"Avg max queue length: {s['max_queue_len']:.1f}"
    )

    # Distribution plot (waits)
    waits_all = np.concatenate([r.waits for r in runs]) if runs else np.array([])
    if waits_all.size > 0:
        fig = plt.figure()
        plt.hist(waits_all, bins=40)
        plt.xlabel("Waiting time (menit)")
        plt.ylabel("Frekuensi")
        plt.title("Distribusi Waiting Time (gabungan semua replikasi)")
        st.pyplot(fig)

    # Small table
    table = pd.DataFrame([{
        "Model": model,
        "Peak": peak_mode,
        "Kasir": cashiers_n,
        "MeanWait": s["mean_wait"],
        "P50": s["wait_p50"],
        "P90": s["wait_p90"],
        "P95": s["wait_p95"],
        "Util": s["utilization"],
        "AvgQ": s["avg_queue_len"],
        "N": s["n_customers"],
    }])
    st.dataframe(table, use_container_width=True)

    st.subheader("Cari jumlah kasir minimum (SLA)")
    st.caption("Klik tombol di bawah untuk mencari kasir minimum pada rentang 1..8 (butuh waktu sedikit lebih lama).")
    if st.button("Cari kasir minimum", type="secondary"):
        with st.spinner("Mencari kasir minimum..."):
            rows = []
            min_mean = None
            min_p90 = None
            min_both = None
            for c in range(1, 9):
                sf = None
                if model != "Single Queue" and enable_hetero:
                    sf = [float(fast_factor)] + [1.0] * (c - 1)

                rr = run_replications(
                    model=model,
                    cashiers_n=c,
                    p=p,
                    arrival_fn=arrival_fn,
                    n_rep=int(n_rep),
                    seed0=int(seed0) + 1000 + 10 * c,
                    speed_factors=sf,
                )
                ss = summarize_runs(rr)
                ss["cashiers"] = c
                rows.append(ss)

                ok_m = ss["mean_wait"] <= sla_mean
                ok_p = ss["wait_p90"] <= sla_p90
                ok_b = ok_m and ok_p
                if min_mean is None and ok_m:
                    min_mean = c
                if min_p90 is None and ok_p:
                    min_p90 = c
                if min_both is None and ok_b:
                    min_both = c

            df = pd.DataFrame(rows).sort_values("cashiers")
            st.dataframe(
                df[["cashiers", "mean_wait", "wait_p90", "wait_p95", "utilization", "avg_queue_len", "n_customers"]],
                use_container_width=True
            )
            st.write(
                f"**Min kasir (mean):** {min_mean if min_mean is not None else 'tidak ditemukan'}  |  "
                f"**Min kasir (P90):** {min_p90 if min_p90 is not None else 'tidak ditemukan'}  |  "
                f"**Min kasir (keduanya):** {min_both if min_both is not None else 'tidak ditemukan'}"
            )

else:
    st.info("Atur parameter di sidebar, lalu klik **Run Simulation**.")