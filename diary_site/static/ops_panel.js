(function () {
  const cpuValueEl = document.getElementById("cpu-value");
  const cpuBarEl = document.getElementById("cpu-bar");
  const cpuLoadLineEl = document.getElementById("cpu-load-line");

  const memValueEl = document.getElementById("mem-value");
  const memBarEl = document.getElementById("mem-bar");
  const memLineEl = document.getElementById("mem-line");

  const diskValueEl = document.getElementById("disk-value");
  const diskBarEl = document.getElementById("disk-bar");
  const diskLineEl = document.getElementById("disk-line");

  const timeEl = document.getElementById("ops-time");
  const errorEl = document.getElementById("ops-error");
  const historyEl = document.getElementById("ops-history");

  let timer = null;

  function clampPercent(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return 0;
    return Math.max(0, Math.min(100, num));
  }

  function setMeter(barEl, percent) {
    if (!barEl) return;
    const p = clampPercent(percent);
    barEl.style.width = `${p}%`;
    if (p >= 85) {
      barEl.className = "is-danger";
    } else if (p >= 60) {
      barEl.className = "is-warning";
    } else {
      barEl.className = "is-ok";
    }
  }

  function pushHistory(text) {
    if (!historyEl) return;
    const li = document.createElement("li");
    li.textContent = text;
    historyEl.prepend(li);

    while (historyEl.children.length > 20) {
      historyEl.removeChild(historyEl.lastElementChild);
    }
  }

  function renderSnapshot(data) {
    const cpu = data.cpu || {};
    const memory = data.memory || {};
    const disk = data.disk || {};

    const cpuPercent = clampPercent(cpu.usage_percent);
    const memPercent = clampPercent(memory.usage_percent);
    const diskPercent = clampPercent(disk.usage_percent);

    cpuValueEl.textContent = `${cpuPercent.toFixed(1)}%`;
    cpuLoadLineEl.textContent = `load(1/5/15m): ${cpu.load_1m ?? "--"} / ${cpu.load_5m ?? "--"} / ${cpu.load_15m ?? "--"} · ${cpu.cpu_count ?? "--"} cores`;
    setMeter(cpuBarEl, cpuPercent);

    memValueEl.textContent = `${memPercent.toFixed(1)}%`;
    memLineEl.textContent = `${memory.used_human || "--"} / ${memory.total_human || "--"}`;
    setMeter(memBarEl, memPercent);

    diskValueEl.textContent = `${diskPercent.toFixed(1)}%`;
    diskLineEl.textContent = `${disk.used_human || "--"} / ${disk.total_human || "--"} · ${disk.path || "--"}`;
    setMeter(diskBarEl, diskPercent);

    const stamp = data.server_time || new Date().toLocaleString();
    timeEl.textContent = `更新时间：${stamp}`;

    pushHistory(`[${stamp}] CPU ${cpuPercent.toFixed(1)}% · MEM ${memPercent.toFixed(1)}% · DISK ${diskPercent.toFixed(1)}%`);
  }

  async function pull() {
    try {
      const response = await fetch("/api/admin/server-status", { cache: "no-store" });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "读取状态失败");
      }
      errorEl.textContent = "";
      renderSnapshot(data);
    } catch (err) {
      errorEl.textContent = `拉取失败：${err}`;
    }
  }

  async function bootstrap() {
    await pull();
    timer = window.setInterval(pull, 3000);
    window.addEventListener("beforeunload", () => {
      if (timer) window.clearInterval(timer);
    });
  }

  bootstrap();
})();
