extends RigidBody3D

class_name Helicopter

@export var main_rotor: Rotor
@export var tail_rotor: Rotor

@onready var helicopter_center_of_mass = compute_center_of_mass()

func _ready() -> void:
	print("Helicopter center of mass:")
	print(helicopter_center_of_mass)

func _process(delta: float) -> void:
	pass

func compute_center_of_mass():
	var fuselage_center_of_mass = self.center_of_mass
	var main_rotor_relative_position = main_rotor.global_position \
									 - self.global_position
	var tail_rotor_relative_position = tail_rotor.global_position \
									 - self.global_position
	var main_rotor_center_of_mass = main_rotor_relative_position \
								  + main_rotor.center_of_mass
	var tail_rotor_center_of_mass = tail_rotor_relative_position \
								  + tail_rotor.center_of_mass
	return (self.mass * fuselage_center_of_mass \
		   + main_rotor.mass * main_rotor_center_of_mass \
		   + tail_rotor.mass * tail_rotor_center_of_mass) \
		   / (self.mass + main_rotor.mass + tail_rotor.mass)
