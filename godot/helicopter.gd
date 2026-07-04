extends RigidBody3D

class_name Helicopter

@export var main_rotor: Rotor
@export var tail_rotor: Rotor
@export var center_of_mass_height: float = 0.0

@export_group("Controls")
@export var min_collective = deg_to_rad(2.0)
@export var max_collective = deg_to_rad(14.0)
@export var min_tail_rotor_cyclic = -deg_to_rad(12.0)
@export var max_tail_rotor_cyclic = deg_to_rad(12.0)
@export var min_lateral_cyclic = -deg_to_rad(8.0)
@export var max_lateral_cyclic = deg_to_rad(8.0)
@export var min_longitudinal_cyclic = -deg_to_rad(8.0)
@export var max_longitudinal_cyclic = deg_to_rad(8.0)

@export_group("Stabilizers")
@export_subgroup("Vertical Stabilizer")
@export_custom(PROPERTY_HINT_NONE, "suffix:m^2") var vertical_stabilizer_area = 0.0
@export_custom(PROPERTY_HINT_NONE, "suffix:m") var vertical_stabilizer_lever = 0.0
@export var vertical_stabilizer_lift_curve = 2.0 * PI


func _ready() -> void:
	compute_center_of_mass()
	
	print("Helicopter center of mass:")
	print(self.center_of_mass)

func _process(delta: float) -> void:	
	if Input.is_action_pressed("ui_page_up"):
		main_rotor.pitch += 0.01 * delta
	elif Input.is_action_pressed("ui_page_down"):
		main_rotor.pitch -= 0.01 * delta
	main_rotor.pitch = clampf(main_rotor.pitch, min_collective, max_collective)
		
		
	if Input.is_action_pressed("ui_alt_right"):
		tail_rotor.pitch = min_tail_rotor_cyclic
	elif Input.is_action_pressed("ui_alt_left"):
		tail_rotor.pitch = max_tail_rotor_cyclic
	else:
		tail_rotor.pitch = 0.0
		
		if Input.is_action_pressed("ui_right"):
			main_rotor.lateral_cyclic = min_lateral_cyclic
		elif Input.is_action_pressed("ui_left"):
			main_rotor.lateral_cyclic = max_lateral_cyclic
		else:
			main_rotor.lateral_cyclic = 0.0
	
	if Input.is_action_pressed("ui_up"):
		main_rotor.longitudinal_cyclic = min_longitudinal_cyclic
	elif Input.is_action_pressed("ui_down"):
		main_rotor.longitudinal_cyclic = max_longitudinal_cyclic
	else:
		main_rotor.longitudinal_cyclic = 0.0
	
	#print("main_rotor: %.4f | tail_rotor: %.4f" % [main_rotor.pitch, tail_rotor.pitch])

func compute_center_of_mass():
	var total_mass = self.mass + main_rotor.mass + tail_rotor.mass
	var main_rotor_relative_position = main_rotor.global_position \
									 - self.global_position
	var tail_rotor_relative_position = tail_rotor.global_position \
									 - self.global_position
	var main_rotor_center_of_mass = main_rotor_relative_position \
								  + main_rotor.center_of_mass
	var tail_rotor_center_of_mass = tail_rotor_relative_position \
								  + tail_rotor.center_of_mass
	var target_center_of_mass = Vector3(0.0, center_of_mass_height, 0.0)
	var fuselage_center_of_mass = (
		target_center_of_mass * total_mass 
		- main_rotor.mass * main_rotor_center_of_mass
		- tail_rotor.mass * tail_rotor_center_of_mass
	) / self.mass
	
	self.center_of_mass = fuselage_center_of_mass
