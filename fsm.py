from djitellopy import Tello
import time

class FSM:
    def __init__(self):
        self.tello = Tello()
        self.state = 'Idle'
        self.is_targeting_active = False
        self.targeting = False
        self.tello.connect()
        self.tello.streamoff()
        print(f"Battery Life Percentage: {self.tello.get_battery()}%")

    def handle_command(self, command):
        print(f"Received command: {command}")  # Print received command
        if command == 'TL':
            if self.state == 'Idle':
                self.state = 'Taking Off'
                self.tello.takeoff()
                time.sleep(2)  # Example delay - adjust based on typical takeoff time
                self.state = 'In Flight'  # Transition to 'In Flight' after takeoff
            elif self.state in ['In Flight', 'Taking Off']:
                self.state = 'Landing'
                self.tello.land()
        elif command == '$':
            self.state = 'Aborting'
            self.tello.emergency()
        elif command == 'T':
            if self.state == 'In Flight':
                self.state = 'Targeting'
                self.is_targeting_active = True  # Turn on targeting
                print(f"Targeting activated: {self.is_targeting_active}")  # Print when targeting is activated
            elif self.state == 'Targeting':
                self.state = 'In Flight'
                self.is_targeting_active = False  # Turn off targeting
                print(f"Targeting deactivated: {self.is_targeting_active}")  # Print when targeting is deactivated

    def start_video_stream(self):
        self.tello.streamon()

    def stop_video_stream(self):
        self.tello.streamoff()

    def cleanup(self):
        self.tello.end()
