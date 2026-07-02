@tool
extends Helicopter

@export var enabled: bool = false

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var draw = Draw3D.new()
	add_child(draw)
	#draw.cube(tail_rotor_position, Basis.IDENTITY.scaled(Vector3.ONE) * 0.1)
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
