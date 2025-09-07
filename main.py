#!/usr/bin/env python3
"""
Fixed Simplified Command & Control (C2) Node System
UDP-based sensor simulation with AI inference and dynamic web dashboard
FIXED: Frontend connectivity, data fetching, and display issues
"""

import utils.session
import json
import time
import threading
import random
import numpy as np
import socket
import math
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import csv
import os
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
UDP_IP = "127.0.0.1"  # Local machine IP
UDP_SENSOR_PORT = 5005  # Sensor data port
UDP_AI_PORT = 5006     # AI results port
MAX_DETECTIONS = 1000   # Stop after this many detections
SEND_RATE = 10         # Sensor update rate (Hz)
BASE_COORDS = (24.4539, 54.3773)  # Abu Dhabi coordinates

class SensorSimulator:
    """Generates synthetic sensor data for 4 sensors via UDP"""
    
    def __init__(self):
        # 4 sensors: 2 radar, 2 IMU
        self.sensors = [
            {'id': 'radar_01', 'type': 'radar'},
            {'id': 'radar_02', 'type': 'radar'},
            {'id': 'imu_01', 'type': 'imu'},
            {'id': 'imu_02', 'type': 'imu'}
        ]
        self.running = False
        self.sock = None
        
    def calculate_magnitude(self, x, y, z):
        """Calculate 3D magnitude"""
        return math.sqrt(x**2 + y**2 + z**2)
    
    def generate_radar_data(self, sensor_id: str) -> Dict[str, Any]:
        """Generate realistic radar data"""
        base_range = 1000 if '01' in sensor_id else 2000
        num_targets = random.randint(1, 4)
        
        targets = []
        for _ in range(num_targets):
            targets.append({
                "distance": float(random.uniform(base_range/2, base_range + 3000)),
                "azimuth": float(random.uniform(-45, 45)),
                "elevation": float(random.uniform(-15, 15)),
                "velocity": float(random.uniform(-50, 50)),
                "rcs": float(random.uniform(-20, 10))
            })
        
        return {
            'sensor_id': sensor_id,
            'timestamp': datetime.now().isoformat(),
            'targets': targets,
            'num_targets': len(targets)
        }
    
    def generate_imu_data(self, sensor_id: str) -> Dict[str, Any]:
        """Generate realistic IMU data"""
        scale = 1.2 if '01' in sensor_id else 0.8
        
        # Generate acceleration with gravity
        accel = np.random.normal(0, 0.5, 3) * scale
        accel[2] += 9.81  # Add gravity to Z-axis
        
        # Generate orientation
        orientation = np.random.uniform(-180, 180, 3)
        
        return {
            'sensor_id': sensor_id,
            'timestamp': datetime.now().isoformat(),
            'acceleration': {
                'x': float(accel[0]),
                'y': float(accel[1]),
                'z': float(accel[2]),
                'magnitude': float(self.calculate_magnitude(*accel))
            },
            'orientation': {
                'roll': float(orientation[0]),
                'pitch': float(orientation[1]),
                'yaw': float(orientation[2])
            },
            'gyro': {
                'x': float(random.uniform(-180, 180) * scale),
                'y': float(random.uniform(-180, 180) * scale),
                'z': float(random.uniform(-180, 180) * scale)
            }
        }
    
    def generate_gps_data(self) -> Dict[str, Any]:
        """Generate GPS data around Abu Dhabi"""
        lat, lon = BASE_COORDS
        return {
            "latitude": lat + random.uniform(-0.135, 0.135),
            "longitude": lon + random.uniform(-0.135, 0.135),
            "altitude": 15.0 + random.uniform(-5, 5),
            "speed": random.uniform(0, 120),
            "satellites": random.randint(5, 12)
        }
    
    def start_simulation(self):
        """Start UDP sensor simulation"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.running = True
            
            logger.info(f"Sensor simulator sending to {UDP_IP}:{UDP_SENSOR_PORT}")
            
            while self.running:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                gps = self.generate_gps_data()
                
                # Generate data for all 4 sensors
                sensor_data = {
                    "timestamp": timestamp,
                    "gps": gps,
                    "sensors": {}
                }
                
                for sensor in self.sensors:
                    if sensor['type'] == 'radar':
                        sensor_data["sensors"][sensor['id']] = self.generate_radar_data(sensor['id'])
                    else:
                        sensor_data["sensors"][sensor['id']] = self.generate_imu_data(sensor['id'])
                
                # Send via UDP
                try:
                    self.sock.sendto(json.dumps(sensor_data).encode(), (UDP_IP, UDP_SENSOR_PORT))
                except Exception as e:
                    logger.error(f"UDP send error: {e}")
                    break
                    
                time.sleep(1 / SEND_RATE)
                
        except Exception as e:
            logger.error(f"Sensor simulator error: {e}")
        finally:
            if self.sock:
                self.sock.close()
            self.running = False

class AIInferenceService:
    """Simplified AI service for anomaly detection"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self._train_model()
    
    def _generate_training_data(self, n_samples=2000):
        """Generate training data for 85%+ accuracy"""
        data = []
        labels = []
        
        for _ in range(n_samples):
            features = [
                random.uniform(-15, 15),    # accel_magnitude
                random.uniform(0, 8),       # target_count
                random.uniform(100, 5000),  # avg_distance
                random.uniform(-50, 50),    # avg_velocity
                random.uniform(-180, 180),  # gyro_magnitude
            ]
            
            # More sophisticated anomaly detection rules
            anomaly_conditions = [
                abs(features[0]) > 12,      # High acceleration
                features[1] > 5,            # Many targets
                features[2] < 300,          # Very close targets
                abs(features[3]) > 40,      # High velocity
                abs(features[4]) > 150,     # High rotation
                (features[1] > 3 and features[2] < 500),  # Combined condition
            ]
            
            anomaly_score = sum(anomaly_conditions)
            label = 1 if anomaly_score >= 2 else 0  # Need at least 2 conditions
            
            data.append(features)
            labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def _train_model(self):
        """Train the AI model to meet 85% accuracy requirement"""
        logger.info("Training AI model for 85%+ accuracy...")
        
        X, y = self._generate_training_data()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Test accuracy
        X_test, y_test = self._generate_training_data(500)
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        self.accuracy = np.mean(predictions == y_test)
        
        self.is_trained = True
        logger.info(f"AI model trained with accuracy: {self.accuracy:.3f}")
        
        if self.accuracy < 0.85:
            logger.warning(f"Accuracy {self.accuracy:.3f} below 85% requirement!")
    
    def extract_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Extract features from combined sensor data"""
        try:
            # Extract IMU features
            imu_data = [data for sensor_id, data in sensor_data["sensors"].items() if 'imu' in sensor_id]
            if imu_data:
                accel_mag = np.mean([data["acceleration"]["magnitude"] for data in imu_data])
                gyro_mag = np.mean([
                    self._calculate_gyro_magnitude(data["gyro"]) for data in imu_data
                ])
            else:
                accel_mag = gyro_mag = 0
            
            # Extract radar features
            radar_data = [data for sensor_id, data in sensor_data["sensors"].items() if 'radar' in sensor_id]
            if radar_data:
                total_targets = sum(data["num_targets"] for data in radar_data)
                all_distances = []
                all_velocities = []
                
                for data in radar_data:
                    for target in data["targets"]:
                        all_distances.append(target["distance"])
                        all_velocities.append(target["velocity"])
                
                avg_distance = np.mean(all_distances) if all_distances else 0
                avg_velocity = np.mean(all_velocities) if all_velocities else 0
            else:
                total_targets = avg_distance = avg_velocity = 0
            
            return [accel_mag, total_targets, avg_distance, avg_velocity, gyro_mag]
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return [0, 0, 0, 0, 0]
    
    def _calculate_gyro_magnitude(self, gyro_data):
        """Calculate gyroscope magnitude"""
        return math.sqrt(gyro_data['x']**2 + gyro_data['y']**2 + gyro_data['z']**2)
    
    def predict(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make anomaly prediction"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            features = np.array([self.extract_features(sensor_data)])
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'prediction': int(prediction),
                'classification': 'anomaly' if prediction == 1 else 'normal',
                'confidence': float(max(probability)),
                'probabilities': {
                    'normal': float(probability[0]),
                    'anomaly': float(probability[1])
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

class DataLogger:
    """Simple CSV logging"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.sensor_log = os.path.join(log_dir, 'sensor_data.csv')
        self.ai_log = os.path.join(log_dir, 'ai_decisions.csv')
        
        # Initialize CSV files
        if not os.path.exists(self.sensor_log):
            with open(self.sensor_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'total_targets', 'avg_distance', 'accel_magnitude'])
        
        if not os.path.exists(self.ai_log):
            with open(self.ai_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'classification', 'confidence'])
    
    def log_sensor_data(self, data: Dict[str, Any]):
        """Log sensor summary to CSV"""
        try:
            # Calculate summary metrics
            radar_data = [d for k, d in data["sensors"].items() if 'radar' in k]
            imu_data = [d for k, d in data["sensors"].items() if 'imu' in k]
            
            total_targets = sum(d["num_targets"] for d in radar_data)
            
            all_distances = []
            for d in radar_data:
                all_distances.extend([t["distance"] for t in d["targets"]])
            avg_distance = np.mean(all_distances) if all_distances else 0
            
            avg_accel = np.mean([d["acceleration"]["magnitude"] for d in imu_data]) if imu_data else 0
            
            with open(self.sensor_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([data['timestamp'], total_targets, avg_distance, avg_accel])
                
        except Exception as e:
            logger.error(f"Error logging sensor data: {e}")
    
    def log_ai_decision(self, decision: Dict[str, Any]):
        """Log AI decision to CSV"""
        try:
            with open(self.ai_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    decision['timestamp'],
                    decision['classification'],
                    decision['confidence']
                ])
        except Exception as e:
            logger.error(f"Error logging AI decision: {e}")

class SimplifiedC2System:
    """Main C2 system with UDP communication and web dashboard"""
    
    def __init__(self):
        self.sensor_sim = SensorSimulator()
        self.ai_service = AIInferenceService()
        self.data_logger = DataLogger()
        self.running = False
        
        self.detection_count = 0
        self.anomaly_count = 0
        self.start_time = None
        
        # Flask app for dashboard
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'c2_simple_secret'
        
        # Enable CORS for all routes
        CORS(self.app)
        
        # Initialize SocketIO with CORS support
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
            async_mode='threading'
        )
        
        # Dashboard data storage with thread lock
        self.data_lock = threading.Lock()
        self.dashboard_data = {
            'sensors': {},
            'alerts': [],
            'metrics': {
                'total_detections': 0,
                'total_anomalies': 0,
                'uptime': 0,
                'detection_rate': 0
            },
            'system_status': 'stopped'
        }
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes for dashboard"""
        
        @self.app.route('/')
        def dashboard():
            return self.get_dashboard_html()
        
        @self.app.route('/api/status')
        def get_status():
            try:
                uptime = time.time() - self.start_time if self.start_time else 0
                return jsonify({
                    'status': 'running' if self.running else 'stopped',
                    'ai_accuracy': float(self.ai_service.accuracy),
                    'sensors_active': len(self.sensor_sim.sensors),
                    'uptime': float(uptime),
                    'total_detections': int(self.detection_count),
                    'total_anomalies': int(self.anomaly_count),
                    'metrics': {
                        'total_detections': int(self.detection_count),
                        'total_anomalies': int(self.anomaly_count),
                        'uptime': float(uptime),
                        'detection_rate': float(self.detection_count / (uptime / 60) if uptime > 0 else 0)
                    }
                })
            except Exception as e:
                logger.error(f"Status API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/sensors')
        def get_sensors():
            try:
                with self.data_lock:
                    return jsonify(self.dashboard_data['sensors'])
            except Exception as e:
                logger.error(f"Sensors API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            try:
                with self.data_lock:
                    return jsonify(self.dashboard_data['alerts'][-50:])  # Last 50 alerts
            except Exception as e:
                logger.error(f"Alerts API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/start', methods=['POST'])
        def start_system():
            try:
                if not self.running:
                    self.start_background()
                    return jsonify({'message': 'System started successfully', 'status': 'success'}), 200
                return jsonify({'message': 'System already running', 'status': 'info'}), 200
            except Exception as e:
                logger.error(f"Start API error: {e}")
                return jsonify({'message': f'Error starting system: {str(e)}', 'status': 'error'}), 500
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_system():
            try:
                if self.running:
                    self.stop()
                    return jsonify({'message': 'System stopped successfully', 'status': 'success'}), 200
                return jsonify({'message': 'System already stopped', 'status': 'info'}), 200
            except Exception as e:
                logger.error(f"Stop API error: {e}")
                return jsonify({'message': f'Error stopping system: {str(e)}', 'status': 'error'}), 500
    
    def get_dashboard_html(self):
        """Return the dashboard HTML directly"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simplified C2 Dashboard - Real-time Monitoring</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #fff; }
        .header { background: linear-gradient(135deg, #2c3e50, #3498db); padding: 20px; border-radius: 10px; margin: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .status-bar { display: flex; gap: 30px; flex-wrap: wrap; }
        .status-item { display: flex; align-items: center; gap: 10px; }
        .status { padding: 8px 16px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 0.9em; }
        .status.running { background: #27ae60; }
        .status.stopped { background: #e74c3c; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px; }
        .card { background: #2d3748; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #4a5568; }
        .card h3 { color: #63b3ed; margin-bottom: 15px; font-size: 1.3em; border-bottom: 2px solid #63b3ed; padding-bottom: 5px; }
        .alert { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 5px solid #fff; animation: slideIn 0.3s ease-out; }
        .alert.medium { background: linear-gradient(135deg, #f39c12, #e67e22); }
        .alert-time { font-size: 0.8em; opacity: 0.9; }
        #alerts-container { max-height: 500px; overflow-y: auto; }
        #sensor-data-container { max-height: 600px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #4a5568; padding: 12px; text-align: left; }
        th { background: #4a5568; color: #e2e8f0; font-weight: 600; }
        td { background: #2d3748; }
        .sensor-row { transition: background-color 0.3s; }
        .sensor-row:hover { background: #3d4852 !important; }
        .controls { display: flex; gap: 15px; flex-wrap: wrap; }
        .btn { padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; text-transform: uppercase; transition: all 0.3s; }
        .btn-start { background: #27ae60; color: white; }
        .btn-stop { background: #e74c3c; color: white; }
        .btn-clear { background: #3498db; color: white; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .metric { background: #4a5568; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #63b3ed; }
        .metric-label { font-size: 0.9em; color: #a0aec0; margin-top: 5px; }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
        .chart-container { position: relative; height: 300px; margin-top: 15px; }
        .sensor-type { font-weight: bold; color: #63b3ed; text-transform: uppercase; }
        .timestamp { color: #a0aec0; font-size: 0.85em; }
        .no-data { text-align: center; color: #a0aec0; padding: 20px; font-style: italic; }
        .console-output { background: #1a1a1a; border: 1px solid #4a5568; border-radius: 5px; padding: 15px; font-family: monospace; font-size: 0.9em; max-height: 400px; overflow-y: auto; }
        .console-line { margin: 2px 0; }
        .anomaly-highlight { color: #e74c3c; font-weight: bold; }
        .normal-highlight { color: #27ae60; }
        .connection-status { padding: 5px 10px; border-radius: 15px; font-size: 0.8em; }
        .connected { background: #27ae60; }
        .disconnected { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Simplified C2 Dashboard</h1>
        <div class="status-bar">
            <div class="status-item">
                <span>Status:</span>
                <span id="system-status" class="status stopped">STOPPED</span>
            </div>
            <div class="status-item">
                <span>Connection:</span>
                <span id="connection-status" class="connection-status disconnected">DISCONNECTED</span>
            </div>
            <div class="status-item">
                <span>AI Accuracy:</span>
                <span id="ai-accuracy">0%</span>
            </div>
            <div class="status-item">
                <span>Uptime:</span>
                <span id="uptime">0s</span>
            </div>
            <div class="status-item">
                <span>Detection Rate:</span>
                <span id="detection-rate">0/min</span>
            </div>
        </div>
    </div>

    <div class="grid">
        <!-- System Metrics -->
        <div class="card">
            <h3>📊 System Metrics</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="total-detections">0</div>
                    <div class="metric-label">Total Detections</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="total-anomalies">0</div>
                    <div class="metric-label">Anomalies</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="anomaly-rate">0%</div>
                    <div class="metric-label">Anomaly Rate</div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>

        <!-- Live Sensor Data -->
        <div class="card">
            <h3>📡 Live Sensor Data (4 Sensors)</h3>
            <div id="sensor-data-container">
                <table id="sensor-table">
                    <thead>
                        <tr>
                            <th>Sensor ID</th>
                            <th>Type</th>
                            <th>Timestamp</th>
                            <th>Key Metrics</th>
                        </tr>
                    </thead>
                    <tbody id="sensor-table-body">
                        <tr><td colspan="4" class="no-data">No sensor data yet</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- AI Alerts -->
        <div class="card">
            <h3>🚨 AI Anomaly Alerts</h3>
            <div id="alerts-container">
                <div class="no-data">No alerts</div>
            </div>
        </div>

        <!-- Console Output -->
        <div class="card">
            <h3>💻 Console Output</h3>
            <div id="console-output" class="console-output">
                <div class="console-line">System ready - waiting for data...</div>
            </div>
        </div>

        <!-- System Controls -->
        <div class="card">
            <h3>⚙️ System Controls</h3>
            <div class="controls">
                <button class="btn btn-start" id="start-btn" onclick="startSystem()">Start System</button>
                <button class="btn btn-stop" id="stop-btn" onclick="stopSystem()">Stop System</button>
                <button class="btn btn-clear" onclick="clearAlerts()">Clear Alerts</button>
                <button class="btn btn-clear" onclick="clearConsole()">Clear Console</button>
            </div>
            <div style="margin-top: 15px; font-size: 0.9em; color: #a0aec0;">
                <p>🔵 UDP Sensor Port: 5005</p>
                <p>🔵 AI Results Port: 5006</p>
                <p>🔵 Max Detections: 1000</p>
            </div>
        </div>
    </div>

    <script>
        let socket;
        let metricsChart;
        let detectionHistory = [];
        let anomalyHistory = [];
        let isConnected = false;

        // Initialize connection
        function initializeConnection() {
            try {
                socket = io({
                    transports: ['websocket', 'polling'],
                    timeout: 5000,
                    reconnection: true,
                    reconnectionAttempts: 5,
                    reconnectionDelay: 1000
                });

                socket.on('connect', function() {
                    isConnected = true;
                    updateConnectionStatus('connected');
                    addConsoleOutput('Connected to backend server', 'normal-highlight');
                });

                socket.on('disconnect', function() {
                    isConnected = false;
                    updateConnectionStatus('disconnected');
                    addConsoleOutput('Disconnected from backend server', 'anomaly-highlight');
                });

                socket.on('connect_error', function(error) {
                    isConnected = false;
                    updateConnectionStatus('disconnected');
                    addConsoleOutput('Connection error: ' + error, 'anomaly-highlight');
                });

                // Socket event handlers
                socket.on('sensor_data', function(data) {
                    updateSensorTable(data);
                    addConsoleOutput(`📡 ${data.sensor_id}: ${data.type.toUpperCase()} data received`);
                });

                socket.on('ai_alert', function(alert) {
                    addAlert(alert);
                    addConsoleOutput(`🚨 ANOMALY: ${alert.message}`, 'anomaly-highlight');
                });

                socket.on('system_status', function(data) {
                    updateSystemStatus(data);
                });

            } catch (error) {
                addConsoleOutput('Failed to initialize connection: ' + error, 'anomaly-highlight');
            }
        }

        // Initialize metrics chart
        function initializeChart() {
            const ctx = document.getElementById('metricsChart').getContext('2d');
            metricsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Total Detections',
                            data: [],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: 'Anomalies',
                            data: [],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            fill: true,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            beginAtZero: true,
                            ticks: { color: '#e2e8f0' }
                        },
                        x: { 
                            maxTicksLimit: 10,
                            ticks: { color: '#e2e8f0' }
                        }
                    },
                    plugins: {
                        legend: { 
                            position: 'top',
                            labels: { color: '#e2e8f0' }
                        },
                        title: { 
                            display: true, 
                            text: 'Detection & Anomaly Trends', 
                            color: '#e2e8f0' 
                        }
                    }
                }
            });
        }

        function updateConnectionStatus(status) {
            const element = document.getElementById('connection-status');
            element.textContent = status.toUpperCase();
            element.className = `connection-status ${status}`;
        }

        function updateSensorTable(data) {
            const tbody = document.getElementById('sensor-table-body');
            
            // Remove no-data row if it exists
            const noDataRow = tbody.querySelector('td[colspan="4"]');
            if (noDataRow) {
                noDataRow.parentElement.remove();
            }
            
            let row = tbody.querySelector(`tr[data-sensor-id="${data.sensor_id}"]`);
            
            if (!row) {
                row = document.createElement('tr');
                row.setAttribute('data-sensor-id', data.sensor_id);
                row.className = 'sensor-row';
                tbody.insertBefore(row, tbody.firstChild);
            }
            
            let metricValue = 'N/A';
            if (data.type === 'radar') {
                metricValue = `${data.data.targets || 0} targets`;
                if (data.data.avg_distance) {
                    metricValue += ` | Avg: ${data.data.avg_distance.toFixed(1)}m`;
                }
            } else if (data.type === 'imu') {
                metricValue = `${data.data.acceleration?.toFixed(2) || 0} m/s²`;
                if (data.data.orientation) {
                    metricValue += ` | Roll: ${data.data.orientation.roll?.toFixed(1) || 0}°`;
                }
            }
            
            row.innerHTML = `
                <td class="sensor-type">${data.sensor_id}</td>
                <td>${data.type.toUpperCase()}</td>
                <td class="timestamp">${new Date(data.timestamp).toLocaleTimeString()}</td>
                <td>${metricValue}</td>
            `;
        }

        function addAlert(alert) {
            const container = document.getElementById('alerts-container');
            if (container.querySelector('.no-data')) {
                container.innerHTML = '';
            }
            
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${alert.severity}`;
            alertDiv.innerHTML = `
                <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                <strong>${alert.message}</strong><br>
                Severity: ${alert.severity.toUpperCase()}
            `;
            container.insertBefore(alertDiv, container.firstChild);
            
            // Limit to 20 alerts
            while (container.children.length > 20) {
                container.removeChild(container.lastChild);
            }
        }

        function updateSystemStatus(data) {
            try {
                document.getElementById('system-status').textContent = data.status.toUpperCase();
                document.getElementById('system-status').className = `status ${data.status}`;
                document.getElementById('ai-accuracy').textContent = (data.ai_accuracy * 100).toFixed(1) + '%';
                
                const metrics = data.metrics;
                document.getElementById('uptime').textContent = Math.round(metrics.uptime) + 's';
                document.getElementById('total-detections').textContent = metrics.total_detections;
                document.getElementById('total-anomalies').textContent = metrics.total_anomalies;
                document.getElementById('detection-rate').textContent = metrics.detection_rate.toFixed(1) + '/min';
                
                // Calculate anomaly rate
                const anomalyRate = metrics.total_detections > 0 ? 
                    (metrics.total_anomalies / metrics.total_detections * 100) : 0;
                document.getElementById('anomaly-rate').textContent = anomalyRate.toFixed(1) + '%';
                
                // Update chart
                const currentTime = new Date().toLocaleTimeString();
                if (metricsChart.data.labels.length > 20) {
                    metricsChart.data.labels.shift();
                    metricsChart.data.datasets[0].data.shift();
                    metricsChart.data.datasets[1].data.shift();
                }
                
                metricsChart.data.labels.push(currentTime);
                metricsChart.data.datasets[0].data.push(metrics.total_detections);
                metricsChart.data.datasets[1].data.push(metrics.total_anomalies);
                metricsChart.update('none'); // No animation for better performance
                
                // Update button states
                updateButtonStates(data.status);
                
            } catch (error) {
                console.error('Error updating system status:', error);
            }
        }

        function updateButtonStates(status) {
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            if (status === 'running') {
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }

        function addConsoleOutput(message, className = '') {
            const container = document.getElementById('console-output');
            const line = document.createElement('div');
            line.className = `console-line ${className}`;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            
            container.insertBefore(line, container.firstChild);
            
            // Limit console lines
            while (container.children.length > 100) {
                container.removeChild(container.lastChild);
            }
        }

        function startSystem() {
            const startBtn = document.getElementById('start-btn');
            startBtn.disabled = true;
            startBtn.textContent = 'Starting...';
            
            fetch('/api/start', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                addConsoleOutput(data.message || 'System started', 
                    data.status === 'success' ? 'normal-highlight' : '');
                if (data.status !== 'success') {
                    startBtn.disabled = false;
                    startBtn.textContent = 'Start System';
                }
            })
            .catch(error => {
                addConsoleOutput('Error starting system: ' + error, 'anomaly-highlight');
                startBtn.disabled = false;
                startBtn.textContent = 'Start System';
            });
        }

        function stopSystem() {
            const stopBtn = document.getElementById('stop-btn');
            stopBtn.disabled = true;
            stopBtn.textContent = 'Stopping...';
            
            fetch('/api/stop', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                addConsoleOutput(data.message || 'System stopped', 
                    data.status === 'success' ? 'normal-highlight' : '');
                if (data.status !== 'success') {
                    stopBtn.disabled = false;
                    stopBtn.textContent = 'Stop System';
                }
            })
            .catch(error => {
                addConsoleOutput('Error stopping system: ' + error, 'anomaly-highlight');
                stopBtn.disabled = false;
                stopBtn.textContent = 'Stop System';
            });
        }

        function clearAlerts() {
            const container = document.getElementById('alerts-container');
            container.innerHTML = '<div class="no-data">No alerts</div>';
            addConsoleOutput('Alerts cleared', 'normal-highlight');
        }

        function clearConsole() {
            const container = document.getElementById('console-output');
            container.innerHTML = '<div class="console-line">Console cleared</div>';
        }

        // Periodic status update
        function startStatusUpdater() {
            setInterval(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => updateSystemStatus(data))
                    .catch(error => console.error('Status update error:', error));
            }, 2000);
        }

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            addConsoleOutput('Initializing dashboard...', 'normal-highlight');
            
            initializeChart();
            initializeConnection();
            startStatusUpdater();
            
            // Initial status check
            setTimeout(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => updateSystemStatus(data))
                    .catch(error => {
                        console.error('Initial status check failed:', error);
                        addConsoleOutput('Failed to connect to backend', 'anomaly-highlight');
                    });
            }, 1000);
            
            addConsoleOutput('Dashboard initialized', 'normal-highlight');
        });
    </script>
</body>
</html>'''
    
    def udp_listener(self):
        """Listen for sensor data via UDP and process with AI"""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((UDP_IP, UDP_SENSOR_PORT))
            sock.settimeout(1.0)  # Add timeout to prevent hanging
            
            logger.info(f"UDP listener started on {UDP_IP}:{UDP_SENSOR_PORT}")
            
            while self.running:
                try:
                    data, _ = sock.recvfrom(4096)  # Larger buffer for more data
                    sensor_data = json.loads(data.decode())
                    
                    # Count total targets/detections
                    total_targets = sum(
                        sensor_data["sensors"][sensor_id]["num_targets"]
                        for sensor_id in sensor_data["sensors"]
                        if 'radar' in sensor_id
                    )
                    self.detection_count += total_targets
                    
                    # Log sensor data
                    self.data_logger.log_sensor_data(sensor_data)
                    
                    # Run AI inference
                    ai_result = self.ai_service.predict(sensor_data)
                    
                    if 'error' not in ai_result:
                        self.data_logger.log_ai_decision(ai_result)
                        
                        if ai_result['classification'] == 'anomaly':
                            self.anomaly_count += 1
                            
                            # Create alert for dashboard
                            alert = {
                                'id': len(self.dashboard_data['alerts']) + 1,
                                'timestamp': ai_result['timestamp'],
                                'classification': ai_result['classification'],
                                'confidence': ai_result['confidence'],
                                'message': f"Anomaly detected - Confidence: {ai_result['confidence']:.2f}",
                                'severity': 'high' if ai_result['confidence'] > 0.8 else 'medium'
                            }
                            
                            with self.data_lock:
                                self.dashboard_data['alerts'].append(alert)
                            
                            # Emit to dashboard
                            try:
                                self.socketio.emit('ai_alert', alert)
                            except Exception as e:
                                logger.error(f"Error emitting alert: {e}")
                    
                    # Update dashboard data
                    self._update_dashboard_data(sensor_data, ai_result)
                    
                    # Print console summary
                    self._print_summary(sensor_data, ai_result, total_targets)
                    
                    if self.detection_count >= MAX_DETECTIONS:
                        logger.info(f"Stopping after {MAX_DETECTIONS} detections")
                        break
                        
                except socket.timeout:
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"UDP processing error: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"UDP listener error: {e}")
        finally:
            if sock:
                sock.close()
    
    def _update_dashboard_data(self, sensor_data, ai_result):
        """Update dashboard data and emit to web interface"""
        try:
            with self.data_lock:
                # Update sensor data for dashboard
                for sensor_id, data in sensor_data["sensors"].items():
                    sensor_type = 'radar' if 'radar' in sensor_id else 'imu'
                    
                    if sensor_type not in self.dashboard_data['sensors']:
                        self.dashboard_data['sensors'][sensor_type] = []
                    
                    # Keep only latest data per sensor
                    sensor_list = self.dashboard_data['sensors'][sensor_type]
                    sensor_list = [s for s in sensor_list if s.get('sensor_id') != sensor_id]
                    
                    # Add formatted data for dashboard
                    dashboard_sensor_data = {
                        'sensor_id': sensor_id,
                        'timestamp': data['timestamp'],
                        'type': sensor_type
                    }
                    
                    if sensor_type == 'radar':
                        dashboard_sensor_data['targets'] = data['num_targets']
                        dashboard_sensor_data['avg_distance'] = np.mean([t['distance'] for t in data['targets']]) if data['targets'] else 0
                    else:
                        dashboard_sensor_data['acceleration'] = data['acceleration']['magnitude']
                        dashboard_sensor_data['orientation'] = data['orientation']
                    
                    sensor_list.append(dashboard_sensor_data)
                    self.dashboard_data['sensors'][sensor_type] = sensor_list[-2:]  # Keep last 2
                
                # Update metrics
                uptime = time.time() - self.start_time if self.start_time else 0
                self.dashboard_data['metrics'] = {
                    'total_detections': self.detection_count,
                    'total_anomalies': self.anomaly_count,
                    'uptime': uptime,
                    'detection_rate': self.detection_count / (uptime / 60) if uptime > 0 else 0
                }
            
            # Emit individual sensor updates
            try:
                for sensor_id, data in sensor_data["sensors"].items():
                    sensor_type = 'radar' if 'radar' in sensor_id else 'imu'
                    
                    dashboard_sensor_data = {
                        'sensor_id': sensor_id,
                        'timestamp': data['timestamp'],
                        'type': sensor_type
                    }
                    
                    if sensor_type == 'radar':
                        dashboard_sensor_data['targets'] = data['num_targets']
                        dashboard_sensor_data['avg_distance'] = np.mean([t['distance'] for t in data['targets']]) if data['targets'] else 0
                    else:
                        dashboard_sensor_data['acceleration'] = data['acceleration']['magnitude']
                        dashboard_sensor_data['orientation'] = data['orientation']
                    
                    self.socketio.emit('sensor_data', {
                        'sensor_id': sensor_id,
                        'type': sensor_type,
                        'data': dashboard_sensor_data,
                        'timestamp': data['timestamp']
                    })
                
                # Emit system status
                self.socketio.emit('system_status', {
                    'status': 'running' if self.running else 'stopped',
                    'metrics': self.dashboard_data['metrics'],
                    'ai_accuracy': self.ai_service.accuracy
                })
                
            except Exception as e:
                logger.error(f"Error emitting socket data: {e}")
                
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def _print_summary(self, sensor_data, ai_result, total_targets):
        """Print formatted summary similar to original code"""
        try:
            print(f"\n=== Detection {self.detection_count} ===")
            print(f"Timestamp: {sensor_data['timestamp']}")
            print(f"Targets detected: {total_targets}")
            
            # GPS info
            gps = sensor_data['gps']
            print(f"GPS: Lat={gps['latitude']:.6f}, Lon={gps['longitude']:.6f}, "
                  f"Alt={gps['altitude']:.1f}m, Speed={gps['speed']:.1f}km/h")
            
            # IMU summary
            imu_data = [d for k, d in sensor_data["sensors"].items() if 'imu' in k]
            if imu_data:
                avg_accel = np.mean([d["acceleration"]["magnitude"] for d in imu_data])
                print(f"IMU Avg Acceleration: {avg_accel:.2f} m/s²")
            
            # Radar summary
            for sensor_id, data in [(k, v) for k, v in sensor_data["sensors"].items() if 'radar' in k]:
                print(f"{sensor_id.upper()}: {data['num_targets']} targets")
                for i, target in enumerate(data['targets'], 1):
                    print(f"  Target {i}: {target['distance']:5.1f}m | "
                          f"{target['azimuth']:5.1f}° | {target['velocity']:5.1f}km/h")
            
            # AI result
            if 'error' not in ai_result:
                status = "🚨 ANOMALY" if ai_result['classification'] == 'anomaly' else "✓ Normal"
                print(f"AI Analysis: {status} (Confidence: {ai_result['confidence']:.2f})")
            
            print(f"Total Anomalies: {self.anomaly_count}")
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
    
    def dashboard_update_thread(self):
        """Thread for periodic dashboard updates"""
        while self.running:
            try:
                # Emit periodic status updates
                with self.data_lock:
                    uptime = time.time() - self.start_time if self.start_time else 0
                    status_data = {
                        'status': 'running' if self.running else 'stopped',
                        'metrics': {
                            'total_detections': self.detection_count,
                            'total_anomalies': self.anomaly_count,
                            'uptime': uptime,
                            'detection_rate': self.detection_count / (uptime / 60) if uptime > 0 else 0
                        },
                        'ai_accuracy': self.ai_service.accuracy
                    }
                
                self.socketio.emit('system_status', status_data)
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
            
            time.sleep(2)  # Update every 2 seconds
    
    def start_background(self):
        """Start system in background (for API calls)"""
        if self.running:
            return
        
        logger.info("Starting C2 System in background...")
        self.running = True
        self.start_time = time.time()
        
        with self.data_lock:
            self.dashboard_data['system_status'] = 'running'
        
        # Start sensor simulator
        sensor_thread = threading.Thread(target=self.sensor_sim.start_simulation, daemon=True)
        sensor_thread.start()
        
        # Start UDP listener
        listener_thread = threading.Thread(target=self.udp_listener, daemon=True)
        listener_thread.start()
        
        # Start dashboard updater
        dashboard_thread = threading.Thread(target=self.dashboard_update_thread, daemon=True)
        dashboard_thread.start()
        
        logger.info("C2 System started successfully")
    
    def start(self):
        """Start the C2 system (console mode)"""
        logger.info("=== Simplified C2 System Starting ===")
        logger.info(f"4 Sensors: {[s['id'] for s in self.sensor_sim.sensors]}")
        logger.info(f"AI Accuracy: {self.ai_service.accuracy:.3f}")
        
        self.running = True
        self.start_time = time.time()
        
        # Start sensor simulator in background
        sensor_thread = threading.Thread(target=self.sensor_sim.start_simulation, daemon=True)
        sensor_thread.start()
        
        # Start dashboard updater
        dashboard_thread = threading.Thread(target=self.dashboard_update_thread, daemon=True)
        dashboard_thread.start()
        
        # Start UDP listener (main thread)
        try:
            self.udp_listener()
        except KeyboardInterrupt:
            logger.info("System stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the C2 system"""
        logger.info("Stopping C2 System...")
        self.running = False
        self.sensor_sim.running = False
        
        with self.data_lock:
            self.dashboard_data['system_status'] = 'stopped'
        
        if self.start_time:
            runtime = time.time() - self.start_time
            logger.info(f"Runtime: {runtime:.1f}s")
            logger.info(f"Total detections: {self.detection_count}")
            logger.info(f"Total anomalies: {self.anomaly_count}")
        
        logger.info("C2 System stopped")
    
    def run_dashboard(self, host='0.0.0.0', port=8081, debug=False):
        """Run the web dashboard"""
        logger.info(f"Starting web dashboard on http://{host}:{port}")
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
        except Exception as e:
            logger.error(f"Dashboard startup error: {e}")


if __name__ == "__main__":
    import sys
    
    # Initialize the C2 system
    c2_system = SimplifiedC2System()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--dashboard-only':
        # Run dashboard only mode
        logger.info("Starting dashboard-only mode")
        logger.info("Use API endpoints to start/stop the system")
        c2_system.run_dashboard(host='0.0.0.0', port=8081, debug=False)
    else:
        # Run console + dashboard mode
        try:
            # Start dashboard in background thread
            dashboard_thread = threading.Thread(
                target=lambda: c2_system.run_dashboard(host='0.0.0.0', port=8081, debug=False),
                daemon=True
            )
            dashboard_thread.start()
            
            # Give dashboard time to start
            time.sleep(3)
            logger.info("Dashboard available at http://localhost:8081")
            
            # Start the main system
            c2_system.start()
            
        except KeyboardInterrupt:
            logger.info("Shutting down C2 System...")
            c2_system.stop()
        except Exception as e:
            logger.error(f"System error: {e}")
            c2_system.stop()