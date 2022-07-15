from dynio import *
import time
import matplotlib.pyplot as plt

class Ctrl:
    def __init__(self): 
        self.five_previous_e = []
        self.prev_e = None

    def PID(self, current_pos, goal_pos):
        kp = 0.45
        ki = 0.01
        kd = 0.01
    
        e = goal_pos - current_pos
        if e > 2047:
            e = e-4095
        elif e < -2047:
            e = e+4095
        
        #print("Current: ", current_pos, "Goal: ", goal_pos, "Error: ", e)
        if len(self.five_previous_e) < 5:
            self.five_previous_e.append(e)
        else:
            self.five_previous_e.pop(0)
            self.five_previous_e.append(e)

        v = kp*e + ki*sum(self.five_previous_e)
        if self.prev_e is not None:
            v += kd*(e-self.prev_e)
        prev_e = e
        
        print("Current: ", current_pos, "Goal: ", goal_pos, "Error: ", e, "Speed: ", v)

        if v > 230:
            v = 230
        if v < -230:
            v = -230
        
        return int(v)

if __name__ == "__main__":

    cs = [Ctrl() for _ in range(6)]

    motor_ids = [1, 5, 4, 3, 2, 6]
    motors = []
    offsets = [0+3100-2000, 3100+3100-4095, 4095-3100+2000, 3400, 0+2000, 2500]
    
    dxl_io = dxl.DynamixelIO('/dev/ttyUSB0', baud_rate=1000000)
    #m = dxl_io.new_mx28(254, 2)
    
    for motor_id, offset in zip(motor_ids, offsets):
        motor = dxl_io.new_mx28(motor_id, 2)
        
        # Set to position mode
        dxl_io.write_control_table(2, motor_id, 0, 64, 1)
        dxl_io.write_control_table(2, motor_id, 3, 11, 1)
        dxl_io.write_control_table(2, motor_id, 1, 64, 1)
        
        motor.set_position(offset)
        motors.append(motor)

    # Wait
    time.sleep(3)

    for motor, motor_id in zip(motors, motor_ids):
        p = motor.get_position()
        print(p)
        
        # Set velocity mode
        dxl_io.write_control_table(2, motor_id, 0, 64, 1)
        dxl_io.write_control_table(2, motor_id, 1, 11, 1)
        dxl_io.write_control_table(2, motor_id, 1, 64, 1)

    
    goal_pos = 0

    #for motor, offset, count in zip(motors, offsets, range(6)):
    #    motor.set_velocity(count*10)

    """for i in range(10):
        p = motors[4].get_position()
        print(p)
        time.sleep(1)"""
        
    while goal_pos < 20000:
        for motor, motor_id, offset, c in zip(motors, motor_ids, offsets, cs):
            p = motor.get_position()
            if motor_id == 4 or motor_id == 6:
                motor.set_velocity(c.PID(p%4095, (-goal_pos + offset)%4095))
            else:
                motor.set_velocity(c.PID(p%4095, (goal_pos + offset)%4095))
        goal_pos += 500

    #time.sleep(10)

    for motor in motors:
        motor.set_velocity(0)
        
