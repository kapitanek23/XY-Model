from flask import Flask, render_template, request, jsonify
from threading import Thread
import time
import numpy as np
from XY_model import XYSystem

app = Flask(__name__)

# Inicjalizacja systemu XY
xy_system = XYSystem(temperature=2.5, width=10)
running = False  # Flaga kontrolująca symulację
in_equilibrate = False  # Flaga kontrolująca stan równoważenia
time_step = 0.05  # Domyślna długość kroku czasowego
steps_per_frame = 1  # Liczba kroków czasowych przed aktualizacją obrazu
simulation_data = {
    'temperatures': [],
    'energies': [],
    'steps': []
}
canvas_scale = 1.0  # Skala siatki dla lepszej widoczności

def run_simulation():
    """Funkcja uruchamiana w tle do ciągłej symulacji."""
    global xy_system, running, simulation_data, time_step, steps_per_frame
    step = 0
    while running:
        if xy_system.temperature == 0:
            time.sleep(0.1)
            continue

        for _ in range(steps_per_frame):
            xy_system.sweep()
            step += 1

        time.sleep(time_step)  # Kontrola płynności symulacji

@app.route('/')
def index():
    """Strona główna."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_simulation():
    """Rozpoczyna ciągłą symulację."""
    global running, in_equilibrate
    if not running:
        running = True
        in_equilibrate = False  # Wyłącz tryb równoważenia
        thread = Thread(target=run_simulation)  # Symulacja działa w osobnym wątku
        thread.start()
    return jsonify({'status': 'Simulation started'})

@app.route('/stop', methods=['POST'])
def stop_simulation():
    """Zatrzymuje ciągłą symulację."""
    global running
    running = False
    return jsonify({'status': 'Simulation stopped'})

@app.route('/data', methods=['GET'])
def get_data():
    """Zwraca aktualny stan systemu."""
    spins = xy_system.spin_config.tolist()
    vortices, antivortices = xy_system.find_vortices()
    current_energy = xy_system.get_energy()  # Oblicz energię

    return jsonify({
        'spins': spins,
        'temperature': xy_system.temperature,
        'width': xy_system.width,
        'energy': current_energy,  # Wysyłaj energię do frontendu
        'vortices': vortices,
        'antivortices': antivortices,
        'canvas_scale': canvas_scale
    })

@app.route('/update', methods=['POST'])
def update_system():
    """Aktualizuje parametry systemu."""
    global xy_system, time_step, steps_per_frame, canvas_scale
    data = request.json
    temperature = data.get('temperature')
    width = data.get('width')
    new_time_step = data.get('time_step')
    new_steps_per_frame = data.get('steps_per_frame')
    new_canvas_scale = data.get('canvas_scale')

    if temperature is not None:
        xy_system.set_temperature(temperature)
    if width and width != xy_system.width:
        xy_system = XYSystem(temperature=temperature or 2.5, width=width)
    if new_time_step is not None:
        time_step = max(0.01, float(new_time_step))  # Minimalny krok czasowy
    if new_steps_per_frame is not None:
        steps_per_frame = max(1, int(new_steps_per_frame))  # Minimalnie 1 krok na ramkę
    if new_canvas_scale is not None:
        canvas_scale = max(0.1, float(new_canvas_scale))  # Minimalna skala 0.1

    print(f"Updated values - time_step: {time_step}, steps_per_frame: {steps_per_frame}, canvas_scale: {canvas_scale}")

    spins = xy_system.spin_config.tolist()
    vortices, antivortices = xy_system.find_vortices()
    current_energy = xy_system.get_energy()

    return jsonify({
        'status': 'System updated',
        'spins': spins,
        'temperature': xy_system.temperature,
        'width': xy_system.width,
        'energy': current_energy,
        'vortices': vortices,
        'antivortices': antivortices,
        'time_step': time_step,
        'steps_per_frame': steps_per_frame,
        'canvas_scale': canvas_scale
    })


@app.route('/equilibrate', methods=['POST'])
def equilibrate_system():
    """Doprowadza układ do stanu równowagi."""
    global xy_system, running, in_equilibrate
    if running:
        return jsonify({'status': 'Error', 'message': 'Stop the simulation before equilibrating.'}), 400

    in_equilibrate = True  # Włącz tryb równoważenia
    max_sweeps = 1000  # Maksymalna liczba iteracji
    tolerance = 1e-4  # Tolerancja zmiany energii
    prev_energy = xy_system.get_energy()

    for _ in range(max_sweeps):
        xy_system.sweep()
        current_energy = xy_system.get_energy()
        if abs(current_energy - prev_energy) < tolerance:
            break
        prev_energy = current_energy

    in_equilibrate = False  # Wyłącz tryb równoważenia
    spins = xy_system.spin_config.tolist()
    vortices, antivortices = xy_system.find_vortices()

    return jsonify({
        'status': 'Equilibrated',
        'spins': spins,
        'temperature': xy_system.temperature,
        'width': xy_system.width,
        'energy': prev_energy,
        'vortices': vortices,
        'antivortices': antivortices
    })

if __name__ == '__main__':
    app.run(debug=True)
