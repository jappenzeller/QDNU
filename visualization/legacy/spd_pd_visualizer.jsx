import { useState, useEffect, useRef, useMemo } from "react";

// Fixed standard-normal samples (seeded) — z1, z2 pairs
const Z_SAMPLES = [[-0.956,0.322],[-0.273,-0.495],[-1.842,-0.311],[-1.141,-1.137],[-0.529,0.093],[1.229,-1.124],[-0.269,0.717],[-1.802,-0.008],[-0.666,-0.555],[-3.279,0.609],[0.565,0.188],[1.003,0.201],[1.504,0.615],[0.384,-1.794],[1.110,0.191],[1.030,-1.564],[0.434,-1.117],[-1.436,0.468],[2.431,0.813],[-0.889,-0.618],[-1.022,-1.329],[-0.585,1.669],[0.491,-0.604],[0.330,1.116],[-0.423,1.529],[-1.222,-1.922],[-0.307,-0.821],[0.329,0.203],[-0.309,0.143],[0.224,0.262],[2.087,0.208],[-1.419,0.164],[0.800,-0.585],[-0.429,-2.220],[-0.065,0.361],[0.357,0.753],[0.499,-1.028],[2.056,-0.056],[0.927,-0.539],[-1.138,-0.611],[-1.213,-0.839],[-0.194,-1.491],[-1.485,0.485],[-0.198,-0.550],[0.085,-0.070],[-1.244,-0.363],[1.257,0.924],[-0.329,0.784],[0.574,-0.391],[-2.150,0.335],[0.429,0.456],[0.102,0.212],[-0.258,-0.788],[-1.479,0.902],[-1.073,-1.572],[0.395,1.423],[0.730,-0.628],[0.394,-1.444],[0.927,1.005],[-0.370,0.395],[-0.241,-3.041],[0.677,-0.670],[0.112,0.424],[-1.002,0.068],[1.759,0.541],[-0.216,-0.807],[-0.250,1.550],[1.451,0.268],[1.736,-0.071],[1.054,-0.076],[0.908,1.671],[-0.284,1.037],[-1.249,0.546],[-0.516,-1.215],[-0.699,1.163],[-0.219,0.415],[-0.909,-0.761],[-0.931,-0.585],[-0.776,1.514],[-0.667,1.404],[0.163,0.881],[-0.639,-1.773],[1.077,1.075],[1.469,0.401],[0.275,1.011],[-0.881,0.828],[0.703,-0.430],[-0.600,0.240],[1.303,-1.429],[0.610,-0.956],[-0.449,0.146],[-1.439,-1.330],[0.169,0.632],[-1.317,-0.389],[-0.177,0.009],[0.304,-0.459],[0.727,0.254],[1.122,2.020],[-0.519,-1.045],[0.738,-0.040],[-0.432,-0.469],[0.360,-0.476],[-0.600,0.814],[0.916,-0.878],[-0.512,0.712],[-0.252,-0.255],[0.835,-1.234],[-1.291,1.451],[2.404,1.432],[-0.138,-0.922],[0.218,-0.206],[-0.326,-1.192],[-0.284,1.944],[1.898,0.586],[-1.465,0.569],[-0.223,-1.039],[0.147,-0.664],[-0.906,-1.562],[-1.076,0.398],[-1.190,0.313],[-0.644,1.139],[0.921,1.229],[-0.128,1.163],[-1.075,-0.334],[-0.861,0.359],[2.001,0.047],[1.831,-0.682],[1.589,-1.678],[-1.298,-0.020],[0.937,-1.937],[-1.647,1.740],[-0.502,-1.640],[0.887,0.599],[0.000,-0.261],[0.261,-0.871],[0.370,-1.031],[0.502,1.014],[1.861,1.280],[-0.190,2.367],[-0.437,-0.777],[1.064,-0.157],[-2.200,1.668],[-0.314,1.439],[-0.588,-0.138],[1.672,-0.610],[-0.032,0.312],[0.731,-0.883],[-0.145,-0.336],[0.391,-0.988],[-1.525,0.057],[-0.721,-0.233],[-0.816,-0.290],[1.076,-0.227],[0.018,-0.177],[-1.039,-1.128],[0.593,-0.675],[0.992,-0.956],[1.104,-1.039],[-0.710,1.124],[0.582,-0.959],[0.662,-0.725],[0.501,0.698],[1.534,-0.379],[-0.186,1.186],[0.128,-0.952],[-0.171,-0.760],[1.048,-1.506],[-0.435,-0.126],[1.101,0.540],[0.922,-0.732],[-0.780,0.421],[-0.953,-0.849],[-0.810,-0.091],[1.240,2.177],[1.562,1.488],[0.660,-1.234],[-0.242,-0.543],[2.175,0.455],[-0.500,-1.539],[1.001,-1.273],[0.814,-0.524],[-0.274,0.403],[-1.497,-1.539],[-1.677,0.210],[-0.102,1.595],[-0.804,-0.097],[-1.003,-0.654],[0.420,-2.063],[0.753,-0.422],[1.933,0.749],[-1.406,2.199],[0.212,1.676],[-0.121,-0.068],[-0.321,0.495],[0.911,0.186],[0.236,-0.749],[0.395,1.342],[-0.695,-0.022],[1.728,0.986],[-0.653,0.299]];

function drawArrow(ctx, cx, cy, dx, dy, len, color, scale) {
  const ex = cx + dx * len * scale;
  const ey = cy - dy * len * scale;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  const angle = Math.atan2(-(ey - cy), ex - cx);
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - 10 * Math.cos(angle - 0.4), ey - 10 * Math.sin(angle - 0.4));
  ctx.lineTo(ex - 10 * Math.cos(angle + 0.4), ey - 10 * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawEllipseTrace(ctx, cx, cy, l1, l2, scale, sigmaK, color, lineWidth, dashed) {
  if (l1 <= 0 || l2 <= 0) return;
  const r1 = sigmaK * Math.sqrt(l1);
  const r2 = sigmaK * Math.sqrt(l2);
  const steps = 300;
  ctx.beginPath();
  if (dashed) ctx.setLineDash([5, 5]);
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * 2 * Math.PI;
    // x = r1*cos(t)*v1 + r2*sin(t)*v2, v1=[1,1]/√2, v2=[1,-1]/√2
    const ch1 = (r1 * Math.cos(t) + r2 * Math.sin(t)) / Math.sqrt(2);
    const ch2 = (r1 * Math.cos(t) - r2 * Math.sin(t)) / Math.sqrt(2);
    const px = cx + ch1 * scale;
    const py = cy - ch2 * scale;
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.setLineDash([]);
}

export default function SPDVisualizer() {
  const [rho, setRho] = useState(0.0);
  const scatterRef = useRef(null);
  const heatRef = useRef(null);

  const rhoC = Math.max(-1.2, Math.min(1.2, rho));
  const l1 = 1 + rhoC;   // eigenvalue 1 — axis along [1,1]/√2
  const l2 = 1 - rhoC;   // eigenvalue 2 — axis along [1,-1]/√2
  const det = l1 * l2;
  const isPD = l1 > 1e-6 && l2 > 1e-6;
  const isNearBoundary = Math.min(Math.abs(l1), Math.abs(l2)) < 0.12;
  const isBeyond = !isPD && !isNearBoundary;

  // Transform Z samples using Cholesky of C=[[1,rho],[rho,1]]
  // L = [[1,0],[rho, sqrt(1-rho²)]] — only valid for |rho|<1
  const rhoSafe = Math.max(-0.999, Math.min(0.999, rhoC));
  const sqrtTerm = Math.sqrt(Math.max(0, 1 - rhoSafe * rhoSafe));
  const dataPoints = useMemo(() => {
    const rs = Math.max(-0.999, Math.min(0.999, rhoSafe));
    const sq = Math.sqrt(Math.max(0, 1 - rs * rs));
    return Z_SAMPLES.map(([z1, z2]) => [z1, rs * z1 + sq * z2]);
  }, [rhoSafe, sqrtTerm]);

  // Draw scatter + ellipse
  useEffect(() => {
    const canvas = scatterRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const scale = 52;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#06060f";
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = "#10102a";
    ctx.lineWidth = 1;
    for (let i = -6; i <= 6; i++) {
      ctx.beginPath(); ctx.moveTo(cx + i*scale, 0); ctx.lineTo(cx + i*scale, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, cy + i*scale); ctx.lineTo(W, cy + i*scale); ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = "#1e1e40";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();

    // Data points
    dataPoints.forEach(([x, y]) => {
      const px = cx + x * scale, py = cy - y * scale;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.fill();
    });

    // Sigma ellipses
    if (isPD) {
      drawEllipseTrace(ctx, cx, cy, l1, l2, scale, 3, "rgba(0,212,170,0.08)", 1, false);
      drawEllipseTrace(ctx, cx, cy, l1, l2, scale, 2, "rgba(0,212,170,0.20)", 1.5, false);
      drawEllipseTrace(ctx, cx, cy, l1, l2, scale, 1, "#00d4aa", 2.5, false);
    } else if (isNearBoundary) {
      // Degenerate — draw the collapse line
      const dir = l1 < l2 ? [1/Math.sqrt(2), -1/Math.sqrt(2)] : [1/Math.sqrt(2), 1/Math.sqrt(2)];
      ctx.beginPath();
      ctx.moveTo(cx - dir[0] * 4 * scale, cy + dir[1] * 4 * scale);
      ctx.lineTo(cx + dir[0] * 4 * scale, cy - dir[1] * 4 * scale);
      ctx.strokeStyle = "#ff9900";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.font = "11px monospace";
      ctx.fillStyle = "#ff9900";
      ctx.fillText("COLLAPSED TO LINE — no area", cx - 95, cy - 3.5 * scale);
    } else {
      ctx.font = "12px monospace";
      ctx.fillStyle = "#ff4444";
      ctx.fillText("NO VALID ELLIPSE — impossible covariance", cx - 140, 24);
    }

    // Eigenvector axes (when PD)
    if (isPD) {
      const sqL1 = Math.sqrt(l1), sqL2 = Math.sqrt(l2);
      drawArrow(ctx, cx, cy,  1/Math.sqrt(2),  1/Math.sqrt(2), sqL1, "#ff6b35", scale);
      drawArrow(ctx, cx, cy,  1/Math.sqrt(2), -1/Math.sqrt(2), sqL2, "#4fc3f7", scale);
      ctx.font = "11px monospace";
      ctx.fillStyle = "#ff6b35";
      ctx.fillText(`√λ₁=${sqL1.toFixed(2)}`, cx + sqL1*scale/Math.sqrt(2) + 8, cy - sqL1*scale/Math.sqrt(2) - 6);
      ctx.fillStyle = "#4fc3f7";
      ctx.fillText(`√λ₂=${sqL2.toFixed(2)}`, cx + sqL2*scale/Math.sqrt(2) + 8, cy + sqL2*scale/Math.sqrt(2) + 14);
    }

    // Axis labels
    ctx.font = "11px monospace";
    ctx.fillStyle = "#303060";
    ctx.fillText("ch1 →", W - 50, cy - 8);
    ctx.fillText("ch2 ↑", cx + 6, 14);

  }, [rhoC, l1, l2, isPD, isNearBoundary, dataPoints]);

  // Draw quadratic form heatmap  x^T C^{-1} x
  useEffect(() => {
    const canvas = heatRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const range = 3.5;
    const scale = W / (2 * range);

    const imgData = ctx.createImageData(W, H);
    const d = imgData.data;

    for (let px = 0; px < W; px++) {
      for (let py = 0; py < H; py++) {
        const x = (px / W - 0.5) * 2 * range;
        const y = -(py / H - 0.5) * 2 * range;
        const idx = (py * W + px) * 4;

        // quadratic form: x^T C^{-1} x = (x² - 2ρxy + y²) / (1-ρ²)
        // when det <= 0, the form isn't a valid bowl
        let val;
        if (Math.abs(det) < 0.005) {
          val = 999;
        } else if (det < 0) {
          // saddle — indefinite
          val = (x*x - 2*rhoC*x*y + y*y) / det; // this goes negative in some directions
        } else {
          val = (x*x - 2*rhoC*x*y + y*y) / det;
        }

        if (det < 0) {
          // indefinite — show saddle in red/blue
          const sv = (x*x - 2*rhoC*x*y + y*y) / Math.abs(det);
          const brightness = Math.min(Math.abs(sv) / 6, 1);
          if (sv < 0) {
            d[idx]   = 150 * brightness;
            d[idx+1] = 10 * brightness;
            d[idx+2] = 10 * brightness;
          } else {
            d[idx]   = 10 * brightness;
            d[idx+1] = 10 * brightness;
            d[idx+2] = 150 * brightness;
          }
          d[idx+3] = 255;
        } else {
          const norm = Math.min(val / 12, 1);
          const b = 1 - norm;
          d[idx]   = b * 5;
          d[idx+1] = b * 80;
          d[idx+2] = b * 90;
          d[idx+3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw contour ellipses on top
    if (isPD) {
      [[1, 1.0, 2.5], [4, 0.45, 1.5], [9, 0.2, 1]].forEach(([k, alpha, lw]) => {
        const steps = 300;
        ctx.beginPath();
        for (let i = 0; i <= steps; i++) {
          const t = (i / steps) * 2 * Math.PI;
          const r1 = Math.sqrt(k * l1);
          const r2 = Math.sqrt(k * l2);
          const ch1 = (r1 * Math.cos(t) + r2 * Math.sin(t)) / Math.sqrt(2);
          const ch2 = (r1 * Math.cos(t) - r2 * Math.sin(t)) / Math.sqrt(2);
          const px = cx + ch1 * scale;
          const py = cy - ch2 * scale;
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.strokeStyle = `rgba(0,212,170,${alpha})`;
        ctx.lineWidth = lw;
        ctx.stroke();
      });
    } else if (det < 0) {
      ctx.font = "11px monospace";
      ctx.fillStyle = "#ff6666";
      ctx.fillText("SADDLE — no closed contours", 10, H - 14);
    }

    // Axis labels
    ctx.font = "11px monospace";
    ctx.fillStyle = "#2a2a50";
    ctx.fillText("ch1", W - 30, cy - 6);
    ctx.fillText("ch2", cx + 4, 14);

  }, [rhoC, l1, l2, det, isPD, isNearBoundary]);

  const statusColor = isNearBoundary ? "#ff9900" : isPD ? "#00d4aa" : "#ff4444";
  const statusText  = isNearBoundary ? "SINGULAR — boundary of SPD manifold"
                    : isPD           ? "POSITIVE DEFINITE — interior of SPD manifold"
                    :                  "NOT POSITIVE DEFINITE — outside SPD manifold";

  const insight = isNearBoundary
    ? "λ₂ → 0. The ellipse collapses to a line. Zero area in one direction means one channel is a perfect linear function of the other — no independent information. This matrix lives on the boundary of the SPD manifold."
    : !isPD
    ? "λ₂ < 0. No real ellipse can satisfy x^T C x = const. A covariance matrix can never have a negative eigenvalue — this point is geometrically outside the SPD manifold. You cannot reach here with real data."
    : rhoC === 0
    ? "ρ = 0. Channels are uncorrelated. Ellipse is a circle. Equal eigenvalues, equal axes. This is the 'center' of the SPD manifold for unit-variance signals."
    : `ρ = ${rhoC.toFixed(2)}. λ₁ = ${l1.toFixed(3)}, λ₂ = ${l2.toFixed(3)}. The ellipse stretches along the [1,1] axis (where channels move together) and shrinks in the [1,−1] axis. The covariance matrix IS this ellipse shape.`;

  return (
    <div style={{
      background: "#06060f",
      minHeight: "100vh",
      padding: "28px 24px",
      fontFamily: "'Courier New', monospace",
      color: "#8080aa",
      boxSizing: "border-box"
    }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, letterSpacing: 4, color: "#404060", marginBottom: 6 }}>
            SPD MANIFOLD · VISUALIZATION
          </div>
          <h1 style={{ margin: 0, fontSize: 20, color: "#e0e0ff", fontWeight: "normal", letterSpacing: 1 }}>
            What is Positive Definiteness?
          </h1>
          <p style={{ margin: "8px 0 0", fontSize: 12, color: "#404060", lineHeight: 1.6 }}>
            2-channel EEG · correlation matrix C = [[1, ρ], [ρ, 1]] · eigenvalues λ₁ = 1+ρ, λ₂ = 1−ρ
          </p>
        </div>

        {/* Status bar */}
        <div style={{
          background: "#0d0d1f",
          border: `1px solid ${statusColor}33`,
          borderLeft: `3px solid ${statusColor}`,
          borderRadius: 4,
          padding: "10px 16px",
          marginBottom: 20,
          fontSize: 12,
          color: statusColor,
          letterSpacing: 1
        }}>
          {statusText}
        </div>

        {/* Canvases */}
        <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 10, color: "#303055", marginBottom: 6, letterSpacing: 2 }}>
              COVARIANCE ELLIPSE &nbsp;·&nbsp; data points + 1σ 2σ 3σ contours
            </div>
            <canvas ref={scatterRef} width={440} height={380}
              style={{ width: "100%", border: "1px solid #12122a", borderRadius: 4, display: "block" }} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 10, color: "#303055", marginBottom: 6, letterSpacing: 2 }}>
              QUADRATIC FORM &nbsp;·&nbsp; x<sup>T</sup>C<sup>−1</sup>x &nbsp; (contours = σ-ellipses, dark = low, bright = high)
            </div>
            <canvas ref={heatRef} width={440} height={380}
              style={{ width: "100%", border: "1px solid #12122a", borderRadius: 4, display: "block" }} />
          </div>
        </div>

        {/* Eigenvalue bars */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          {[
            { label: "λ₁ = 1 + ρ", val: l1, max: 2.2, color: "#ff6b35", axis: "axis [1,1]/√2  (+45°)" },
            { label: "λ₂ = 1 − ρ", val: l2, max: 2.2, color: "#4fc3f7", axis: "axis [1,−1]/√2 (−45°)" }
          ].map(({ label, val, max, color, axis }) => (
            <div key={label} style={{ flex: 1, background: "#0a0a1a", borderRadius: 4, padding: "12px 16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ color, fontSize: 12 }}>{label} = {val.toFixed(3)}</span>
                <span style={{ fontSize: 10, color: "#303055" }}>{axis}</span>
              </div>
              <div style={{ height: 8, background: "#111128", borderRadius: 2, overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${Math.max(0, val / max * 100)}%`,
                  background: val <= 0 ? "#ff3333" : color,
                  transition: "width 0.08s, background 0.1s"
                }} />
              </div>
              <div style={{ fontSize: 10, color: "#252545", marginTop: 4 }}>
                {val <= 0 ? "⚠ NEGATIVE — not PD" : val < 0.1 ? "→ 0 approaching boundary" : `√λ = ${Math.sqrt(Math.max(0,val)).toFixed(3)} (half-axis length)`}
              </div>
            </div>
          ))}
        </div>

        {/* Slider */}
        <div style={{ background: "#0a0a1a", borderRadius: 4, padding: "16px 20px", marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
            <span style={{ fontSize: 12, color: "#6060a0" }}>
              Channel correlation &nbsp;<span style={{ color: "#e0e0ff" }}>ρ</span>&nbsp; (off-diagonal of C)
            </span>
            <span style={{ fontSize: 18, color: "#00d4aa", minWidth: 60, textAlign: "right" }}>
              {rho.toFixed(3)}
            </span>
          </div>
          <input
            type="range" min={-1.2} max={1.2} step={0.005}
            value={rho}
            onChange={e => setRho(parseFloat(e.target.value))}
            style={{ width: "100%", accentColor: "#00d4aa", cursor: "pointer" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#252545", marginTop: 6 }}>
            <span style={{ color: "#ff4444" }}>−1.2 impossible</span>
            <span style={{ color: "#ff9900" }}>−1 boundary</span>
            <span style={{ color: "#4fc3f7" }}>0 circle</span>
            <span style={{ color: "#ff9900" }}>+1 boundary</span>
            <span style={{ color: "#ff4444" }}>+1.2 impossible</span>
          </div>
        </div>

        {/* Insight box */}
        <div style={{
          background: "#08081a",
          border: "1px solid #151530",
          borderRadius: 4,
          padding: "14px 18px",
          fontSize: 12,
          color: "#6060a0",
          lineHeight: 1.7
        }}>
          <span style={{ color: "#00d4aa" }}>↳ </span>{insight}
        </div>

        {/* Key equation */}
        <div style={{
          marginTop: 16,
          display: "flex",
          gap: 12,
          fontSize: 11,
          color: "#303055"
        }}>
          <div style={{ flex: 1, background: "#08081a", borderRadius: 4, padding: "12px 14px" }}>
            <div style={{ color: "#404065", marginBottom: 6 }}>WHAT POSITIVE DEFINITE MEANS</div>
            <div style={{ lineHeight: 1.8 }}>
              For every direction <span style={{ color: "#ff6b35" }}>v ≠ 0</span>:<br/>
              <span style={{ color: "#e0e0ff", fontSize: 13 }}>v<sup>T</sup>Cv &gt; 0</span><br/>
              The matrix "pushes back" in every direction.<br/>
              No direction collapses to zero energy.<br/>
              Geometrically: <span style={{ color: "#00d4aa" }}>the ellipse has real positive area.</span>
            </div>
          </div>
          <div style={{ flex: 1, background: "#08081a", borderRadius: 4, padding: "12px 14px" }}>
            <div style={{ color: "#404065", marginBottom: 6 }}>EACH EEG WINDOW IS A POINT</div>
            <div style={{ lineHeight: 1.8 }}>
              One 1s window → one covariance matrix → one ellipse shape<br/>
              → <span style={{ color: "#00d4aa" }}>one point on the SPD manifold</span><br/><br/>
              The manifold is the space of all valid shapes.<br/>
              Interictal brain = one region. Ictal brain = another.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
