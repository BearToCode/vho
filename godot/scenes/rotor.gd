extends HingeJoint3D

class_name Rotor

@export_custom(PROPERTY_HINT_NONE, "suffix:kg/m^3") var air_density: float = 1.225
@export var profile_drag_coefficient: float = 0.01
@export_custom(PROPERTY_HINT_NONE, "suffix:rad^-1") var lift_curve: float = 2 * PI

@export_group("Geometry")
@export_custom(PROPERTY_HINT_NONE, "suffix:m") var radius: float = 10.0
@export var blade_count: int = 2
@export_custom(PROPERTY_HINT_NONE, "suffix:m") var chord: float = 0.

@export_group("Inputs")
@export_custom(PROPERTY_HINT_NONE, "suffix:W") var power: float = 10.0
@export_range(-15.0/180*PI, 15.0/180*PI, 0.001, "suffix:rad") var pitch: float = 0.0

@onready var solidity: float = (blade_count * chord) / (PI * radius)
@onready var fuselage: RigidBody3D = get_node(node_a)
@onready var rotor: RigidBody3D = get_node(node_b)


func _ready() -> void:
	pass

func _process(_delta: float) -> void:
	pass

func _physics_process(_delta: float) -> void:	
	var omega = get_rotation_velocity()
	
	var inflow_ratio = (solidity * lift_curve) / 16 * \
					   (sqrt(1 + (32 * pitch) / (solidity * lift_curve)) - 1)
	var thrust_coefficient = solidity * lift_curve / 2 * \
							 (pitch / 3 - inflow_ratio / 2)
	
	var torque = compute_rotor_torque(omega, inflow_ratio, thrust_coefficient)
	var force = compute_rotor_force(omega, thrust_coefficient)
		
	rotor.apply_torque(torque)
	fuselage.apply_torque(-torque)
	fuselage.apply_force(force)
	
	
func get_local_velocity(state: PhysicsDirectBodyState3D) -> Vector3:
	var b = state.transform.basis
	var v_len = state.linear_velocity.length()
	var v_nor = state.linear_velocity.normalized()

	var vel : Vector3
	vel.x = b.x.dot(v_nor) * v_len
	vel.y = b.y.dot(v_nor) * v_len
	vel.z = b.z.dot(v_nor) * v_len
	
	return vel
	
func get_rotation_velocity() -> float:
	var global_angular_velocity = rotor.angular_velocity
	var omega = global_angular_velocity.length()
	return omega
	
func compute_rotor_torque(
	omega: float,
	inflow_ratio: float, 
	thrust_coefficient: float
) -> Vector3:
	var torque_motor = power / max(omega, 1e-3)
	
	var torque_coefficient = (solidity * profile_drag_coefficient) / 8 + \
							 inflow_ratio * thrust_coefficient
	var torque_drag = torque_coefficient * air_density * PI * pow(radius, 2) \
					* pow(omega * radius, 2) * radius
	
	const MAX_TORQUE = 10_000
	var torque_net = torque_motor - torque_drag
	torque_net = clamp(torque_net, -MAX_TORQUE, MAX_TORQUE)
	
	print("torque_net: %s" % [torque_net])
	
	return rotor.global_transform.basis.y * torque_net

func compute_rotor_force(
	omega: float, 
	thrust_coefficient: float
) -> Vector3:
	var thrust = air_density * PI * pow(radius, 2) * pow(omega * radius, 2) * \
				 thrust_coefficient
	print("thrust: %s, omega: %s" % [thrust, omega])
	return rotor.global_transform.basis.y * thrust
