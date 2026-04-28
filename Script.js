(function() {
  // ---------- STARFIELD ANIMATION ----------
  const starsCanvas = document.getElementById('stars');
  const ctxStars = starsCanvas.getContext('2d');
  let width, height, starsArray = [];
  function initStars() {
    width = window.innerWidth;
    height = window.innerHeight;
    starsCanvas.width = width;
    starsCanvas.height = height;
    starsArray = [];
    const count = Math.floor(width * height * 0.00055);
    for (let i = 0; i < count; i++) {
      starsArray.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 2.0 + 0.5,
        alpha: Math.random() * 0.7 + 0.25,
        speed: Math.random() * 0.22 + 0.07,
      });
    }
  }
  function drawStars() {
    ctxStars.clearRect(0, 0, width, height);
    for (let s of starsArray) {
      ctxStars.beginPath();
      ctxStars.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
      ctxStars.fillStyle = `rgba(220, 210, 255, ${s.alpha})`;
      ctxStars.fill();
      s.alpha += (Math.random() - 0.5) * 0.014;
      if (s.alpha > 0.9) s.alpha = 0.9;
      if (s.alpha < 0.15) s.alpha = 0.15;
      s.y -= s.speed;
      if (s.y < -5) { s.y = height + 5; s.x = Math.random() * width; }
    }
    requestAnimationFrame(drawStars);
  }
  initStars();
  drawStars();
  window.addEventListener('resize', () => {
    initStars();
  });

  // ---------- GLOBAL VARIABLE ----------
  let chartInstance = null;
  let bestModel = null;
  let dataPoints = [];

  // ---------- UTILS ----------
  function parseData(str) {
    const lines = str.trim().split(/\r?\n/);
    const points = [];
    for (let line of lines) {
      const parts = line.trim().split(/[ ,\t]+/);
      if (parts.length >= 2) {
        const x = parseFloat(parts[0]);
        const y = parseFloat(parts[1]);
        if (!isNaN(x) && !isNaN(y)) points.push({ x, y });
      }
    }
    return points;
  }

  function mean(arr) {
    return arr.reduce((a,b)=>a+b,0)/arr.length;
  }

  // ---------- REGRESSION MODELS ----------
  function linearFit(xs, ys) {
    const n = xs.length;
    const sumX = xs.reduce((a,b)=>a+b,0);
    const sumY = ys.reduce((a,b)=>a+b,0);
    const sumXY = xs.reduce((a,b,i)=>a + b*ys[i], 0);
    const sumX2 = xs.reduce((a,b)=>a + b*b, 0);
    const det = n*sumX2 - sumX*sumX;
    if (Math.abs(det) < 1e-12) return null;
    const p1 = (n*sumXY - sumX*sumY) / det;
    const p0 = (sumY - p1*sumX) / n;
    const yPred = xs.map(x => p0 + p1*x);
    const ssRes = ys.reduce((s, y, i) => s + (y - yPred[i])**2, 0);
    const yMean = mean(ys);
    const ssTot = ys.reduce((s, y) => s + (y - yMean)**2, 0);
    const r2 = ssTot === 0 ? 0 : 1 - ssRes/ssTot;
    return { p0, p1, r2, func: x => p0 + p1*x };
  }

  function polyFit(xs, ys, degree = 2) {
    const n = xs.length;
    const A = [];
    const B = [];
    for (let i = 0; i <= degree; i++) {
      B[i] = xs.reduce((sum, x, idx) => sum + Math.pow(x, i) * ys[idx], 0);
      A[i] = [];
      for (let j = 0; j <= degree; j++) {
        A[i][j] = xs.reduce((sum, x) => sum + Math.pow(x, i+j), 0);
      }
    }
    const size = degree + 1;
    const aug = A.map((row, i) => [...row, B[i]]);
    for (let col = 0; col < size; col++) {
      let maxRow = col;
      for (let row = col+1; row < size; row++) {
        if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
      }
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
      if (Math.abs(aug[col][col]) < 1e-12) return null;
      for (let row = col+1; row < size; row++) {
        const factor = aug[row][col] / aug[col][col];
        for (let k = col; k <= size; k++) aug[row][k] -= factor * aug[col][k];
      }
    }
    const coeffs = new Array(size).fill(0);
    for (let i = size-1; i >= 0; i--) {
      let sum = aug[i][size];
      for (let j = i+1; j < size; j++) sum -= aug[i][j] * coeffs[j];
      coeffs[i] = sum / aug[i][i];
    }
    const func = x => {
      let val = 0;
      for (let i = 0; i < coeffs.length; i++) val += coeffs[i] * Math.pow(x, i);
      return val;
    };
    const yPred = xs.map(func);
    const ssRes = ys.reduce((s, y, i) => s + (y - yPred[i])**2, 0);
    const yMean = mean(ys);
    const ssTot = ys.reduce((s, y) => s + (y - yMean)**2, 0);
    const r2 = ssTot === 0 ? 0 : 1 - ssRes/ssTot;
    return { coeffs, r2, func };
  }

  function exponentialFit(xs, ys) {
    if (ys.some(y => y <= 0)) return null;
    const lnYs = ys.map(y => Math.log(y));
    const lin = linearFit(xs, lnYs);
    if (!lin) return null;
    const a = Math.exp(lin.p0);
    const b = lin.p1;
    const func = x => a * Math.exp(b * x);
    const yPred = xs.map(func);
    const ssRes = ys.reduce((s, y, i) => s + (y - yPred[i])**2, 0);
    const yMean = mean(ys);
    const ssTot = ys.reduce((s, y) => s + (y - yMean)**2, 0);
    const r2 = ssTot === 0 ? 0 : 1 - ssRes/ssTot;
    return { a, b, r2, func, type: b > 0 ? 'Eksponensial Pertumbuhan' : 'Eksponensial Peluruhan' };
  }

  function saturationFit(xs, ys, maxIter = 8) {
    const n = xs.length;
    let a = Math.max(...ys) * 1.08;
    let b = 0.5;
    for (let iter = 0; iter < maxIter; iter++) {
      const valid = ys.every(y => y < a);
      if (!valid) a = Math.max(...ys) * 1.12;
      const Y = ys.map(y => Math.log(Math.max(1e-9, 1 - y/a)));
      let sumX2 = 0, sumXY = 0;
      for (let i = 0; i < n; i++) {
        sumX2 += xs[i] * xs[i];
        sumXY += xs[i] * Y[i];
      }
      if (sumX2 === 0) break;
      b = -sumXY / sumX2;
      if (b <= 0) b = 0.01;
      const z = xs.map(x => 1 - Math.exp(-b*x));
      let sumZ2 = 0, sumYZ = 0;
      for (let i = 0; i < n; i++) {
        sumZ2 += z[i]*z[i];
        sumYZ += ys[i]*z[i];
      }
      if (sumZ2 > 1e-12) a = sumYZ / sumZ2;
      if (a <= Math.max(...ys)) a = Math.max(...ys) * 1.02;
    }
    const func = x => a * (1 - Math.exp(-b*x));
    const yPred = xs.map(func);
    const ssRes = ys.reduce((s, y, i) => s + (y - yPred[i])**2, 0);
    const yMean = mean(ys);
    const ssTot = ys.reduce((s, y) => s + (y - yMean)**2, 0);
    const r2 = ssTot === 0 ? 0 : 1 - ssRes/ssTot;
    return { a, b, r2, func };
  }

  function findBestModel(points) {
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const candidates = [];
    const lin = linearFit(xs, ys);
    if (lin) candidates.push({ type: 'Linear', r2: lin.r2, func: lin.func, params: `p0=${lin.p0.toFixed(4)}, p1=${lin.p1.toFixed(4)}`, expr: `${lin.p0.toFixed(4)} + ${lin.p1.toFixed(4)}·x` });
    const poly2 = polyFit(xs, ys, 2);
    if (poly2 && poly2.coeffs) {
      const [c0,c1,c2] = poly2.coeffs;
      candidates.push({ type: 'Polynomial (derajat 2)', r2: poly2.r2, func: poly2.func, params: `c0=${c0.toFixed(4)}, c1=${c1.toFixed(4)}, c2=${c2.toFixed(4)}`, expr: `${c0.toFixed(4)} + ${c1.toFixed(4)}·x + ${c2.toFixed(4)}·x²` });
    }
    const expo = exponentialFit(xs, ys);
    if (expo) candidates.push({ type: expo.type, r2: expo.r2, func: expo.func, params: `a=${expo.a.toFixed(4)}, b=${expo.b.toFixed(4)}`, expr: `${expo.a.toFixed(4)} · exp(${expo.b.toFixed(4)}·x)` });
    const sat = saturationFit(xs, ys);
    if (sat) candidates.push({ type: 'Eksponensial Saturasi', r2: sat.r2, func: sat.func, params: `a=${sat.a.toFixed(4)}, b=${sat.b.toFixed(4)}`, expr: `${sat.a.toFixed(4)} · (1 - exp(-${sat.b.toFixed(4)}·x))` });
    if (candidates.length === 0) return null;
    candidates.sort((a,b) => b.r2 - a.r2);
    return candidates[0];
  }

  function renderChart(points, modelFunc) {
    const ctx = document.getElementById('fitChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();
    const xsData = points.map(p => p.x);
    const xMin = Math.min(...xsData);
    const xMax = Math.max(...xsData);
    const step = (xMax - xMin) / 100;
    const curveX = [], curveY = [];
    for (let x = xMin - (xMax-xMin)*0.1; x <= xMax + (xMax-xMin)*0.1; x += step) {
      curveX.push(x);
      curveY.push(modelFunc(x));
    }
    chartInstance = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Data', data: points.map(p => ({ x: p.x, y: p.y })), backgroundColor: '#f0abfc', borderColor: '#d8b4fe', pointRadius: 4, pointHoverRadius: 7, order: 1 },
          { label: 'Model Fit', data: curveX.map((x,i)=>({x, y:curveY[i]})), type: 'line', borderColor: '#a78bfa', borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.2, order: 0 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e9d5ff', font: { size: 11 } } } },
        scales: {
          x: { title: { display: true, text: 'x', color: '#d8b4fe' }, ticks: { color: '#c4b5fd' }, grid: { color: '#331d5c' } },
          y: { title: { display: true, text: 'y', color: '#d8b4fe' }, ticks: { color: '#c4b5fd' }, grid: { color: '#331d5c' } }
        }
      }
    });
  }

  function updateUI(model, points) {
    document.getElementById('modelName').textContent = model.type;
    document.getElementById('equationDisplay').textContent = 'y = ' + model.expr;
    document.getElementById('rSquared').textContent = model.r2.toFixed(6);
    document.getElementById('modelReason').textContent = `Dipilih karena R² tertinggi = ${model.r2.toFixed(6)}. Model: ${model.type}. Parameter: ${model.params}`;
    renderChart(points, model.func);
  }

  function bisection(f, target, aInit, bInit, maxIter=50, tol=1e-8) {
    let a = aInit, b = bInit;
    const g = x => f(x) - target;
    let fa = g(a), fb = g(b);
    const tableRows = [];
    if (fa * fb > 0) {
      let expand = (b-a)*0.5;
      for (let tries=0; tries<20; tries++) {
        a -= expand; b += expand;
        fa = g(a); fb = g(b);
        if (fa*fb <= 0) break;
        expand *= 1.5;
      }
      if (fa*fb > 0) return { success: false, message: 'Nilai target tidak terapit. Masukkan interval manual.' };
    }
    let c, fc;
    for (let i=0; i<maxIter; i++) {
      c = (a+b)/2;
      fc = g(c);
      tableRows.push({ iter: i+1, a, b, c, fc, errorAbs: Math.abs(b-a), errorRel: Math.abs((b-a)/(c!==0?c:1e-12)) });
      if (Math.abs(b-a) < tol || Math.abs(fc) < 1e-12) break;
      if (fa*fc < 0) { b = c; fb = fc; }
      else { a = c; fa = fc; }
    }
    return { success: true, root: (a+b)/2, tableRows, finalErrorAbs: Math.abs(b-a), finalErrorRel: Math.abs((b-a)/(((a+b)/2)||1e-12)) };
  }

  function renderBisectionTable(rows) {
    if (!rows || rows.length===0) return '';
    let html = '<div class="table-wrapper"><table><thead><tr><th>Iter</th><th>a</th><th>b</th><th>c</th><th>f(c)-target</th><th>|b-a|</th><th>Relatif</th></tr></thead><tbody>';
    rows.forEach(r => {
      html += `<tr><td>${r.iter}</td><td>${r.a.toFixed(6)}</td><td>${r.b.toFixed(6)}</td><td>${r.c.toFixed(6)}</td><td>${r.fc.toExponential(4)}</td><td>${r.errorAbs.toExponential(4)}</td><td>${(r.errorRel*100).toFixed(6)}%</td></tr>`;
    });
    html += '</tbody></table></div>';
    return html;
  }

  // event listeners
  document.getElementById('analyzeBtn').addEventListener('click', ()=>{
    const raw = document.getElementById('dataInput').value;
    const points = parseData(raw);
    document.getElementById('inputError').textContent = '';
    if (points.length < 16) {
      document.getElementById('inputError').textContent = 'Error: Masukkan minimal 16 pasangan data (x,y).';
      return;
    }
    dataPoints = points;
    const best = findBestModel(points);
    if (!best) {
      document.getElementById('inputError').textContent = 'Gagal menemukan model, cek data.';
      return;
    }
    bestModel = best;
    updateUI(best, points);
  });

  document.getElementById('loadSampleBtn').addEventListener('click', ()=>{
    document.getElementById('dataInput').value = `1 2\n2 3.5\n3 5\n4 6.8\n5 8.5\n6 10\n7 11.5\n8 13\n9 14.2\n10 15.3\n11 16.2\n12 17\n13 17.7\n14 18.3\n15 18.8\n16 19.2\n17 19.5\n18 19.7\n19 19.9\n20 20`;
    document.getElementById('inputError').textContent = '';
  });

  document.getElementById('predictBtn').addEventListener('click', ()=>{
    const xVal = parseFloat(document.getElementById('predictX').value);
    if (isNaN(xVal) || !bestModel) {
      document.getElementById('predictResult').textContent = 'Input x dan pastikan model sudah di-fit.';
      return;
    }
    document.getElementById('predictResult').textContent = bestModel.func(xVal).toFixed(6);
  });

  document.getElementById('bisectionBtn').addEventListener('click', ()=>{
    document.getElementById('bisectionError').textContent = '';
    document.getElementById('bisectionResult').textContent = '-';
    document.getElementById('bisectionErrors').textContent = '';
    document.getElementById('tableContainer').innerHTML = '';
    if (!bestModel) {
      document.getElementById('bisectionError').textContent = 'Jalankan fitting model terlebih dahulu.';
      return;
    }
    const target = parseFloat(document.getElementById('targetY').value);
    if (isNaN(target)) {
      document.getElementById('bisectionError').textContent = 'Masukkan target y yang valid.';
      return;
    }
    const xs = dataPoints.map(p=>p.x);
    let a = Math.min(...xs), b = Math.max(...xs);
    const manA = parseFloat(document.getElementById('manualA').value);
    const manB = parseFloat(document.getElementById('manualB').value);
    if (!isNaN(manA) && !isNaN(manB)) { a = manA; b = manB; }
    const result = bisection(bestModel.func, target, a, b);
    if (!result.success) {
      document.getElementById('bisectionError').textContent = result.message || 'Gagal.';
      return;
    }
    document.getElementById('bisectionResult').textContent = result.root.toFixed(8);
    document.getElementById('bisectionErrors').innerHTML = `Error absolut akhir: ${result.finalErrorAbs.toExponential(4)} &nbsp; | &nbsp; Error relatif: ${(result.finalErrorRel*100).toFixed(8)}%`;
    document.getElementById('tableContainer').innerHTML = renderBisectionTable(result.tableRows) + '<div style="margin-top:0.5rem; color:#c4b5fd; font-size:0.8rem;"><strong>Interpretasi:</strong> Error absolut ≈ setengah lebar interval akhir, menunjukkan ketidakpastian x. Error relatif menunjukkan presisi terhadap nilai x.</div>';
  });

  // auto-load sample on first load
  window.addEventListener('load', ()=>{
    document.getElementById('loadSampleBtn').click();
    document.getElementById('analyzeBtn').click();
  });
})();