import socket
import json
import time
import threading
from typing import Dict, Optional, Callable

class AssettoCorsaTelemetry:
    """
    Assetto Corsa Telemetry Interface
    Receives and processes telemetry data from the game
    """
    
    def __init__(self, port=9996, host='localhost'):
        self.port = port
        self.host = host
        self.socket = None
        self.running = False
        self.latest_data = {}
        self.callbacks = []
        
        # Default telemetry structure
        self.default_data = {
            'speed': 0.0,           # km/h
            'rpm': 0.0,             # RPM
            'gear': 1,              # Current gear
            'tire_temp_fl': 80.0,   # Front left tire temperature
            'tire_temp_fr': 80.0,   # Front right tire temperature
            'tire_temp_rl': 80.0,   # Rear left tire temperature
            'tire_temp_rr': 80.0,   # Rear right tire temperature
            'fuel': 100.0,          # Fuel level (%)
            'lap_time': 0.0,        # Current lap time
            'lap_count': 1,         # Current lap number
            'position': 1,          # Race position
            'track_position': 0.0,  # Position on track (0-1)
            'brake_temp': 80.0,     # Brake temperature
            'oil_temp': 90.0,       # Oil temperature
            'water_temp': 90.0,     # Water temperature
            'throttle': 0.0,        # Throttle input (0-1)
            'brake': 0.0,           # Brake input (0-1)
            'steering': 0.0,        # Steering input (-1 to 1)
            'clutch': 0.0,          # Clutch input (0-1)
            'drs': False,           # DRS active
            'abs': True,            # ABS active
            'tcs': True,            # Traction control active
            'damage': 0.0,          # Car damage (0-1)
            'g_force': 0.0,         # G-force
            'acceleration': 0.0,    # Acceleration
            'braking': 0.0,         # Braking force
            'cornering': 0.0,       # Cornering force
            'slip_angle': 0.0,      # Slip angle
            'tire_wear_fl': 0.0,    # Front left tire wear
            'tire_wear_fr': 0.0,    # Front right tire wear
            'tire_wear_rl': 0.0,    # Rear left tire wear
            'tire_wear_rr': 0.0,    # Rear right tire wear
            'brake_wear_fl': 0.0,   # Front left brake wear
            'brake_wear_fr': 0.0,   # Front right brake wear
            'brake_wear_rl': 0.0,   # Rear left brake wear
            'brake_wear_rr': 0.0,   # Rear right brake wear
        }
        
        self.latest_data = self.default_data.copy()
    
    def start(self):
        """Start telemetry reception"""
        if self.running:
            return
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(0.1)
            self.running = True
            
            # Start reception thread
            self.reception_thread = threading.Thread(target=self._receive_loop)
            self.reception_thread.daemon = True
            self.reception_thread.start()
            
            print(f"Telemetry started on {self.host}:{self.port}")
            
        except Exception as e:
            print(f"Failed to start telemetry: {e}")
            self.running = False
    
    def stop(self):
        """Stop telemetry reception"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("Telemetry stopped")
    
    def _receive_loop(self):
        """Main reception loop"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                telemetry = json.loads(data.decode())
                
                # Update latest data
                self.latest_data.update(telemetry)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.latest_data)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Telemetry reception error: {e}")
                time.sleep(0.1)
    
    def get_data(self) -> Dict:
        """Get latest telemetry data"""
        return self.latest_data.copy()
    
    def get_numpy_data(self) -> Optional[list]:
        """Get telemetry data as numpy array for RL"""
        if not self.latest_data:
            return None
        
        # Convert to numpy array format expected by RL environment
        data = [
            self.latest_data.get('speed', 0.0),
            self.latest_data.get('rpm', 0.0),
            self.latest_data.get('gear', 1),
            self.latest_data.get('tire_temp_fl', 80.0),
            self.latest_data.get('tire_temp_fr', 80.0),
            self.latest_data.get('tire_temp_rl', 80.0),
            self.latest_data.get('tire_temp_rr', 80.0),
            self.latest_data.get('fuel', 100.0),
            self.latest_data.get('lap_time', 0.0),
            self.latest_data.get('lap_count', 1),
            self.latest_data.get('position', 1),
            self.latest_data.get('track_position', 0.0),
            self.latest_data.get('brake_temp', 80.0),
            self.latest_data.get('oil_temp', 90.0),
            self.latest_data.get('water_temp', 90.0)
        ]
        
        return data
    
    def register_callback(self, callback: Callable[[Dict], None]):
        """Register a callback function to be called when new data arrives"""
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[Dict], None]):
        """Unregister a callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def is_connected(self) -> bool:
        """Check if telemetry is connected and receiving data"""
        return self.running and len(self.latest_data) > 0
    
    def get_speed(self) -> float:
        """Get current speed in km/h"""
        return self.latest_data.get('speed', 0.0)
    
    def get_rpm(self) -> float:
        """Get current RPM"""
        return self.latest_data.get('rpm', 0.0)
    
    def get_gear(self) -> int:
        """Get current gear"""
        return self.latest_data.get('gear', 1)
    
    def get_tire_temps(self) -> tuple:
        """Get tire temperatures (FL, FR, RL, RR)"""
        return (
            self.latest_data.get('tire_temp_fl', 80.0),
            self.latest_data.get('tire_temp_fr', 80.0),
            self.latest_data.get('tire_temp_rl', 80.0),
            self.latest_data.get('tire_temp_rr', 80.0)
        )
    
    def get_lap_info(self) -> tuple:
        """Get lap information (lap_time, lap_count, position)"""
        return (
            self.latest_data.get('lap_time', 0.0),
            self.latest_data.get('lap_count', 1),
            self.latest_data.get('position', 1)
        )
    
    def get_damage(self) -> float:
        """Get car damage level (0-1)"""
        return self.latest_data.get('damage', 0.0)
    
    def get_g_force(self) -> float:
        """Get current G-force"""
        return self.latest_data.get('g_force', 0.0)


class TelemetryAnalyzer:
    """
    Analyzes telemetry data for racing insights
    """
    
    def __init__(self, telemetry: AssettoCorsaTelemetry):
        self.telemetry = telemetry
        self.history = []
        self.max_history = 1000  # Keep last 1000 data points
    
    def update(self):
        """Update analyzer with latest data"""
        data = self.telemetry.get_data()
        self.history.append(data)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_optimal_shift_points(self) -> Dict[int, float]:
        """Calculate optimal shift points for each gear"""
        if len(self.history) < 10:
            return {}
        
        shift_points = {}
        for gear in range(1, 8):  # Gears 1-7
            gear_data = [d for d in self.history if d.get('gear') == gear]
            if len(gear_data) > 5:
                # Find RPM where power starts dropping
                rpms = [d.get('rpm', 0) for d in gear_data]
                speeds = [d.get('speed', 0) for d in gear_data]
                
                if rpms and speeds:
                    # Simple heuristic: shift at 90% of max RPM
                    max_rpm = max(rpms)
                    shift_points[gear] = max_rpm * 0.9
        
        return shift_points
    
    def get_tire_performance(self) -> Dict[str, float]:
        """Analyze tire performance"""
        if len(self.history) < 10:
            return {}
        
        recent_data = self.history[-10:]  # Last 10 data points
        
        # Calculate average tire temperatures
        avg_temps = {
            'fl': sum(d.get('tire_temp_fl', 80) for d in recent_data) / len(recent_data),
            'fr': sum(d.get('tire_temp_fr', 80) for d in recent_data) / len(recent_data),
            'rl': sum(d.get('tire_temp_rl', 80) for d in recent_data) / len(recent_data),
            'rr': sum(d.get('tire_temp_rr', 80) for d in recent_data) / len(recent_data)
        }
        
        # Calculate tire wear
        avg_wear = {
            'fl': sum(d.get('tire_wear_fl', 0) for d in recent_data) / len(recent_data),
            'fr': sum(d.get('tire_wear_fr', 0) for d in recent_data) / len(recent_data),
            'rl': sum(d.get('tire_wear_rl', 0) for d in recent_data) / len(recent_data),
            'rr': sum(d.get('tire_wear_rr', 0) for d in recent_data) / len(recent_data)
        }
        
        return {
            'avg_temps': avg_temps,
            'avg_wear': avg_wear,
            'temp_variance': np.var(list(avg_temps.values())),
            'wear_variance': np.var(list(avg_wear.values()))
        }
    
    def get_driving_style_analysis(self) -> Dict[str, float]:
        """Analyze driving style characteristics"""
        if len(self.history) < 20:
            return {}
        
        recent_data = self.history[-20:]
        
        # Calculate average inputs
        avg_throttle = sum(d.get('throttle', 0) for d in recent_data) / len(recent_data)
        avg_brake = sum(d.get('brake', 0) for d in recent_data) / len(recent_data)
        avg_steering = sum(abs(d.get('steering', 0)) for d in recent_data) / len(recent_data)
        
        # Calculate input variance (smoothness)
        throttle_variance = np.var([d.get('throttle', 0) for d in recent_data])
        brake_variance = np.var([d.get('brake', 0) for d in recent_data])
        steering_variance = np.var([d.get('steering', 0) for d in recent_data])
        
        return {
            'aggressiveness': (avg_throttle + avg_brake) / 2,
            'smoothness': 1.0 / (1.0 + throttle_variance + brake_variance + steering_variance),
            'steering_intensity': avg_steering,
            'throttle_usage': avg_throttle,
            'brake_usage': avg_brake
        }


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create telemetry interface
    telemetry = AssettoCorsaTelemetry()
    
    # Create analyzer
    analyzer = TelemetryAnalyzer(telemetry)
    
    # Example callback
    def print_speed(data):
        print(f"Speed: {data.get('speed', 0):.1f} km/h, RPM: {data.get('rpm', 0):.0f}")
    
    # Register callback
    telemetry.register_callback(print_speed)
    
    # Start telemetry
    telemetry.start()
    
    try:
        # Run for 10 seconds
        for i in range(10):
            time.sleep(1)
            analyzer.update()
            
            # Print analysis
            if telemetry.is_connected():
                print(f"Connected: {telemetry.get_speed():.1f} km/h")
                print(f"Tire temps: {telemetry.get_tire_temps()}")
                print(f"Lap info: {telemetry.get_lap_info()}")
            else:
                print("Waiting for telemetry data...")
    
    except KeyboardInterrupt:
        print("Stopping telemetry...")
    
    finally:
        telemetry.stop() 