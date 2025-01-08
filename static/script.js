const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let simulationInterval = null;

function drawSpins(spins, width, vortices, antivortices) {
    const cellSize = canvas.width / width;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    spins.forEach((angle, i) => {
        const x = (i % width) * cellSize + cellSize / 2;
        const y = Math.floor(i / width) * cellSize + cellSize / 2;
        const dx = Math.cos(angle) * cellSize / 2;
        const dy = Math.sin(angle) * cellSize / 2;
        drawArrow(ctx, x, y, x + dx, y + dy, 'black');
    });

    vortices.forEach(([i, j]) => {
        const x = j * cellSize + cellSize / 2;
        const y = i * cellSize + cellSize / 2;
        ctx.beginPath();
        ctx.arc(x, y, cellSize / 4, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
    });

    antivortices.forEach(([i, j]) => {
        const x = j * cellSize + cellSize / 2;
        const y = i * cellSize + cellSize / 2;
        ctx.beginPath();
        ctx.arc(x, y, cellSize / 4, 0, 2 * Math.PI);
        ctx.fillStyle = 'blue';
        ctx.fill();
    });
}

function drawArrow(ctx, fromX, fromY, toX, toY, color) {
    const headLength = 10;
    const dx = toX - fromX;
    const dy = toY - fromY;
    const angle = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(
        toX - headLength * Math.cos(angle - Math.PI / 6),
        toY - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
        toX - headLength * Math.cos(angle + Math.PI / 6),
        toY - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.lineTo(toX, toY);
    ctx.fillStyle = color;
    ctx.fill();
}


async function updateSystemParams() {
    const temperature = parseFloat(document.getElementById('temperature').value);
    const width = parseInt(document.getElementById('width').value);
    const timeStep = parseFloat(document.getElementById('time-step').value);
    const stepsPerFrame = parseInt(document.getElementById('steps-per-frame').value);

    const response = await fetch('/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature, width, time_step: timeStep, steps_per_frame: stepsPerFrame })
    });

    const data = await response.json();

    drawSpins(data.spins, data.width, data.vortices, data.antivortices);
    document.getElementById('energy-value').textContent = `Energia: ${data.energy.toFixed(4)}`;
    document.getElementById('temperature-value').textContent = data.temperature.toFixed(1);
    document.getElementById('width-value').textContent = data.width;
    document.getElementById('time-step-value').textContent = data.time_step.toFixed(2);
    document.getElementById('steps-per-frame-value').textContent = data.steps_per_frame;
}

function startSimulation() {
    fetch('/start', { method: 'POST' });
    simulationInterval = setInterval(fetchSimulationData, 100);
}

function stopSimulation() {
    fetch('/stop', { method: 'POST' });
    clearInterval(simulationInterval);
}

async function fetchSimulationData() {
    const response = await fetch('/data');
    const data = await response.json();

    drawSpins(data.spins, data.width, data.vortices, data.antivortices);
    document.getElementById('energy-value').textContent = `Energia: ${data.energy.toFixed(4)}`;
}

async function equilibrateSystem() {
    const response = await fetch('/equilibrate', { method: 'POST' });
    const data = await response.json();

    drawSpins(data.spins, data.width, data.vortices, data.antivortices);
    document.getElementById('energy-value').textContent = `Energia: ${data.energy.toFixed(4)}`;
    document.getElementById('steps-value').textContent = `Kroki do równowagi: ${data.steps_to_equilibrium}`; // Nowa linia
}

document.getElementById('temperature').addEventListener('input', updateSystemParams);
document.getElementById('width').addEventListener('input', updateSystemParams);
document.getElementById('time-step').addEventListener('input', updateSystemParams);
document.getElementById('steps-per-frame').addEventListener('input', updateSystemParams);
document.getElementById('start-button').addEventListener('click', startSimulation);
document.getElementById('stop-button').addEventListener('click', stopSimulation);
document.getElementById('equilibrate-button').addEventListener('click', equilibrateSystem);
