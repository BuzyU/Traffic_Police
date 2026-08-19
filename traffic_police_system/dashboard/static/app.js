/* ==========================================================================
   TRAFFIC POLICE AI — DASHBOARD APP LOGIC
   ========================================================================== */

let countsChart = null;
let currentCalibMode = null; // 'stop_line' | 'direction' | null
let calibPoints = [];
let savedStopLine = null;
let savedDirection = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  initCalibrationCanvas();
  initUploadHandler();
  startPolling();
});

/* ── Live Chart Initialization ───────────────────────────────────────────── */
function initChart() {
  const ctx = document.getElementById("countsChart").getContext("2d");
  
  countsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Car (Yellow)",
          borderColor: "#eab308",
          backgroundColor: "rgba(234, 179, 8, 0.1)",
          data: [],
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: "Motorcycle (Blue)",
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          data: [],
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: "Bus (Black/Grey)",
          borderColor: "#71717a",
          backgroundColor: "rgba(113, 113, 122, 0.1)",
          data: [],
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: "Truck (Red)",
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          data: [],
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          labels: {
            color: "#94a3b8",
            font: { size: 10 }
          }
        }
      },
      scales: {
        x: {
          display: false
        },
        y: {
          beginAtZero: true,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: {
            color: "#64748b",
            font: { size: 10 },
            stepSize: 1
          }
        }
      }
    }
  });
}

/* ── Polling Server Endpoints ────────────────────────────────────────────── */
function startPolling() {
  setInterval(fetchCounts, 600);
  setInterval(fetchViolations, 1000);
  setInterval(fetchAccidents, 1000);
}

async function fetchCounts() {
  try {
    const res = await fetch("/api/counts");
    const data = await res.json();
    if (data.status === "success") {
      const c = data.counts;
      document.getElementById("count-car").textContent = c.car || 0;
      document.getElementById("count-motorcycle").textContent = c.motorcycle || 0;
      document.getElementById("count-bus").textContent = c.bus || 0;
      document.getElementById("count-truck").textContent = c.truck || 0;

      // Update timeline chart with real logged points
      if (data.timeline && data.timeline.length > 0) {
        const labels = data.timeline.map((_, i) => i);
        const carData = data.timeline.map(t => t.counts.car || 0);
        const motoData = data.timeline.map(t => t.counts.motorcycle || 0);
        const busData = data.timeline.map(t => t.counts.bus || 0);
        const truckData = data.timeline.map(t => t.counts.truck || 0);

        countsChart.data.labels = labels;
        countsChart.data.datasets[0].data = carData;
        countsChart.data.datasets[1].data = motoData;
        countsChart.data.datasets[2].data = busData;
        countsChart.data.datasets[3].data = truckData;
        countsChart.update();
      }
    }
  } catch (err) {
    console.warn("Counts fetch failed", err);
  }
}

async function fetchViolations() {
  try {
    const res = await fetch("/api/violations");
    const data = await res.json();
    if (data.status === "success" && data.violations) {
      const tbody = document.getElementById("violations-tbody");
      const countBadge = document.getElementById("violations-count-badge");
      
      countBadge.textContent = `${data.violations.length} Logged`;

      if (data.violations.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No geometry violations detected yet. Set stop-line or direction above.</td></tr>`;
        return;
      }

      let html = "";
      // Show newest first
      const rev = [...data.violations].reverse();
      for (const v of rev) {
        const sevClass = v.severity === "Critical" ? "badge-severity-critical" :
                         v.severity === "High" ? "badge-severity-high" : "badge-severity-medium";
        
        html += `
          <tr>
            <td style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #94a3b8;">${v.timestamp.split(' ')[1] || v.timestamp}</td>
            <td><strong>#${v.vehicle_id}</strong></td>
            <td><span style="text-transform: capitalize;">${v.vehicle_type}</span></td>
            <td>${v.violation_type}</td>
            <td><span class="${sevClass}">${v.severity}</span></td>
          </tr>
        `;
      }
      tbody.innerHTML = html;
    }
  } catch (err) {
    console.warn("Violations fetch failed", err);
  }
}

async function fetchAccidents() {
  try {
    const res = await fetch("/api/accidents");
    const data = await res.json();
    if (data.status === "success" && data.accidents) {
      const tbody = document.getElementById("accidents-tbody");

      if (data.accidents.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No collision events detected.</td></tr>`;
        return;
      }

      let html = "";
      const rev = [...data.accidents].reverse();
      for (const a of rev) {
        html += `
          <tr>
            <td style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #94a3b8;">${a.timestamp.split(' ')[1] || a.timestamp}</td>
            <td>#${a.vehicle_id_1} (${a.vehicle_type_1})</td>
            <td>#${a.vehicle_id_2} (${a.vehicle_type_2})</td>
            <td style="color: #f87171; font-weight: 700;">${(a.iou_overlap * 100).toFixed(0)}% IoU</td>
            <td><span class="badge-severity-high">Overlap+Decel</span></td>
          </tr>
        `;
      }
      tbody.innerHTML = html;
    }
  } catch (err) {
    console.warn("Accidents fetch failed", err);
  }
}

/* ── Interactive Calibration Canvas ──────────────────────────────────────── */
function initCalibrationCanvas() {
  const canvas = document.getElementById("calibration-canvas");
  const ctx = canvas.getContext("2d");
  const wrapper = document.getElementById("video-wrapper");

  function resizeCanvas() {
    canvas.width = wrapper.clientWidth;
    canvas.height = wrapper.clientHeight;
    redrawCalibrationLines();
  }

  window.addEventListener("resize", resizeCanvas);
  setTimeout(resizeCanvas, 300);

  const btnStopLine = document.getElementById("btn-draw-stopline");
  const btnDirection = document.getElementById("btn-draw-direction");
  const btnClear = document.getElementById("btn-clear-calib");
  const statusText = document.getElementById("calib-status-text");

  btnStopLine.addEventListener("click", () => {
    currentCalibMode = "stop_line";
    calibPoints = [];
    btnStopLine.classList.add("active-tool");
    btnDirection.classList.remove("active-tool");
    statusText.textContent = "Click 2 points on video to define STOP LINE (point 1 of 2)";
  });

  btnDirection.addEventListener("click", () => {
    currentCalibMode = "direction";
    calibPoints = [];
    btnDirection.classList.add("active-tool");
    btnStopLine.classList.remove("active-tool");
    statusText.textContent = "Click 2 points to define ALLOWED DIRECTION (start -> end)";
  });

  btnClear.addEventListener("click", async () => {
    savedStopLine = null;
    savedDirection = null;
    calibPoints = [];
    currentCalibMode = null;
    btnStopLine.classList.remove("active-tool");
    btnDirection.classList.remove("active-tool");
    statusText.textContent = "Calibration cleared.";
    redrawCalibrationLines();

    await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stop_line: null, allowed_direction: null })
    });
  });

  canvas.addEventListener("click", async (e) => {
    if (!currentCalibMode) return;

    const rect = canvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    calibPoints.push([x, y]);

    if (calibPoints.length === 1) {
      statusText.textContent = `Captured point 1 (${x}, ${y}). Click point 2.`;
      redrawCalibrationLines();
    } else if (calibPoints.length === 2) {
      if (currentCalibMode === "stop_line") {
        savedStopLine = [calibPoints[0], calibPoints[1]];
        statusText.textContent = `✔ Stop line saved.`;
      } else if (currentCalibMode === "direction") {
        savedDirection = [calibPoints[0], calibPoints[1]];
        statusText.textContent = `✔ Allowed direction saved.`;
      }

      // Send to backend
      await fetch("/api/calibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stop_line: savedStopLine,
          allowed_direction: savedDirection
        })
      });

      calibPoints = [];
      currentCalibMode = null;
      btnStopLine.classList.remove("active-tool");
      btnDirection.classList.remove("active-tool");
      redrawCalibrationLines();
    }
  });

  function redrawCalibrationLines() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw active clicked point
    if (calibPoints.length === 1) {
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(calibPoints[0][0], calibPoints[0][1], 5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw saved Stop Line
    if (savedStopLine) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(savedStopLine[0][0], savedStopLine[0][1]);
      ctx.lineTo(savedStopLine[1][0], savedStopLine[1][1]);
      ctx.stroke();

      ctx.fillStyle = "#ef4444";
      ctx.font = "bold 12px Inter";
      ctx.fillText("STOP LINE", savedStopLine[0][0] + 5, savedStopLine[0][1] - 5);
    }

    // Draw saved Direction Arrow
    if (savedDirection) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(savedDirection[0][0], savedDirection[0][1]);
      ctx.lineTo(savedDirection[1][0], savedDirection[1][1]);
      ctx.stroke();

      ctx.fillStyle = "#10b981";
      ctx.font = "bold 12px Inter";
      ctx.fillText("ALLOWED FLOW", savedDirection[0][0] + 5, savedDirection[0][1] - 5);
    }
  }
}

/* ── Video Upload Handler ────────────────────────────────────────────────── */
function initUploadHandler() {
  const input = document.getElementById("video-upload-input");
  input.addEventListener("change", async () => {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const statusPill = document.getElementById("pill-engine");
    statusPill.innerHTML = `<span class="pulse-dot cyan"></span><span>Uploading ${file.name}...</span>`;

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.status === "success") {
        statusPill.innerHTML = `<span class="pulse-dot green"></span><span>Processing: ${file.name}</span>`;
        // Reload stream image
        const streamImg = document.getElementById("stream-img");
        streamImg.src = "/api/stream?t=" + new Date().getTime();
      }
    } catch (err) {
      alert("Failed to upload video: " + err.message);
      statusPill.innerHTML = `<span class="pulse-dot red"></span><span>Upload Failed</span>`;
    }
  });
}

/* ── Modal Dialog for Blocked Features ───────────────────────────────────── */
async function showStubModal(featureKey) {
  let endpoint = "/api/emergency-vehicle";
  if (featureKey === "model") endpoint = "/api/vehicle-model/1";
  else if (featureKey === "helmet") endpoint = "/api/helmet-violation/1";

  try {
    const res = await fetch(endpoint);
    const data = await res.json();

    document.getElementById("modal-title").textContent = data.feature + " (Status: " + data.status.toUpperCase() + ")";
    document.getElementById("modal-reason").textContent = data.reason;
    document.getElementById("modal-req-format").textContent = data.required_data_format;

    document.getElementById("stub-modal").style.display = "flex";
  } catch (err) {
    console.error("Modal fetch error", err);
  }
}

function closeStubModal() {
  document.getElementById("stub-modal").style.display = "none";
}
