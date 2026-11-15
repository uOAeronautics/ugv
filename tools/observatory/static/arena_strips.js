// Telemetry strips under the POV: vision validity, gyro, and commanded actions. Click to scrub.

// ---- strips: 1 vision validity, 2 gyro, 3 actions. Click or drag to scrub. ----
function drawStrips() {
  const frames = AR.episodes[AR.epIdx]?.frames || [];
  const n = frames.length;
  const playFrac = n > 0 ? AR.frame / n : 0;
  const pal = palette();
  ['strip1', 'strip2', 'strip3'].forEach((id) => {
    const c = document.getElementById(id);
    if (!c) return;
    const { ctx, w: W, h: H } = fitCanvas(c);
    ctx.fillStyle = pal.plot;
    ctx.fillRect(0, 0, W, H);
    if (!n) {
      ctx.fillStyle = pal.muted;
      ctx.font = MONO_SM;
      ctx.fillText('no frames', 10, H / 2);
      return;
    }
    const padT = 12;
    const X = (i) => (i / n) * W;
    const BW = Math.max(1, W / n);
    const mid = H / 2;
    if (id === 'strip1') {
      let maxAge = 1;
      frames.forEach((f) => {
        const a = Number(f.tag_age) || 0;
        if (a > maxAge) maxAge = a;
      });
      frames.forEach((f, i) => {
        let col = pal.border;
        if (f.outlier) col = pal.warn;
        else if (f.blackout) col = pal.err;
        else if (f.valid) col = pal.succ;
        ctx.fillStyle = col;
        ctx.fillRect(X(i), 3, BW, H - 16);
      });
      ctx.strokeStyle = pal.warn;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      frames.forEach((f, i) => {
        const a = Math.min(1, (Number(f.tag_age) || 0) / maxAge);
        const y = H - 13 - a * (H - padT - 16);
        const x = X(i) + BW / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    } else if (id === 'strip2') {
      let m = 1e-6;
      frames.forEach((f) => {
        m = Math.max(m, Math.abs(Number(f.imu_w) || 0), Math.abs(Number(f.imu_yaw) || 0));
      });
      ctx.strokeStyle = pal.border;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(W, mid);
      ctx.stroke();
      const trace = (key, col) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        frames.forEach((f, i) => {
          const v = (Number(f[key]) || 0) / m;
          const y = mid - v * (mid - padT);
          const x = X(i) + BW / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };
      trace('imu_w', pal.series[0]);
      trace('imu_yaw', pal.series[3]);
    } else {
      let m = 1e-6;
      frames.forEach((f) => {
        const a = f.act || [0, 0, 0];
        m = Math.max(m, Math.abs(a[0] || 0), Math.abs(a[1] || 0), Math.abs(a[2] || 0));
      });
      frames.forEach((f, i) => {
        const w = (f.act || [0, 0, 0])[2] || 0;
        if (Math.abs(w) > 0.8) {
          ctx.fillStyle = 'rgba(217,119,6,0.16)';
          ctx.fillRect(X(i), 0, BW, H);
        }
      });
      ctx.strokeStyle = pal.border;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(W, mid);
      ctx.stroke();
      const trace = (idx, col) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        frames.forEach((f, i) => {
          const a = f.act || [0, 0, 0];
          const v = (a[idx] || 0) / m;
          const y = mid - v * (mid - padT);
          const x = X(i) + BW / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };
      trace(0, pal.succ);
      trace(1, pal.series[5]);
      trace(2, pal.warn);
      frames.forEach((f, i) => {
        if (f.shield_blocked) {
          ctx.fillStyle = pal.err;
          ctx.fillRect(X(i), 0, BW, 3);
        }
      });
    }
    const px = playFrac * W;
    ctx.strokeStyle = pal.ink;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(Math.round(px) + 0.5, 0);
    ctx.lineTo(Math.round(px) + 0.5, H);
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function scrubFromStrip(e) {
  const frames = AR.episodes[AR.epIdx]?.frames || [];
  if (!frames.length) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
  setFrame(Math.round(frac * (frames.length - 1)));
}
