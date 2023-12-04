class PID:
    def __init__(self, Kp, setpoint, initial_value=0.0, output_limits=(0, 1)):
        self.Kp = Kp
        self.output_limits = output_limits
        self.setpoint = setpoint
        self.ov = initial_value

    def __call__(self, value):
        self.ov = self.ov + (self.setpoint - value) * self.Kp
        self.ov = max(min(self.ov, self.output_limits[1]), self.output_limits[0])
        return self.ov

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def get_setpoint(self):
        return self.setpoint


__all__ = ["PID"]
