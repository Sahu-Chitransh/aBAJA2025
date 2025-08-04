import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from vehiclecontrol.msg import Control

class LanePIDController(Node):
    def __init__(self):
        super().__init__('lane_pid_controller')

        # Parameters for PID
        self.kp = 0.05
        self.ki = 0.0001
        self.kd = 0.05
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05  # 20 Hz

        self.steering_target = 0.0

        self.create_subscription(Float32, '/steering_angle', self.steering_callback, 10)
        self.control_pub = self.create_publisher(Control, '/vehicle_control', 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Lane PID Controller Node Initialized.")

    def steering_callback(self, msg):
        self.steering_target = msg.data

    def control_loop(self):
        error = self.steering_target
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # PID output in degrees
        steer_angle_deg = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Clip within physical limits
        max_steer_deg = 30.0
        steer_angle_deg = max(min(steer_angle_deg, max_steer_deg), -max_steer_deg)

        # Normalize to [-1, 1]
        steer_command = steer_angle_deg / max_steer_deg 

        msg = Control()
        msg.steering = float(steer_command)
        msg.throttle = 0.3  # constant throttle for testing
        msg.brake = 0.0
        msg.latswitch = 1
        msg.longswitch = 0

        self.control_pub.publish(msg)
        self.get_logger().info(f"Steer PID: {steer_command:.2f}, Target: {self.steering_target:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = LanePIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
