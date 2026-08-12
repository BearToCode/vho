extends Helicopter


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if Input.is_action_pressed("ui_right") and \
	 	not Input.is_action_pressed("ui_alt_right"):
		self.lateral_cyclic = 1.0
	elif Input.is_action_pressed("ui_left") and \
	 	not Input.is_action_pressed("ui_alt_left"):
		self.lateral_cyclic = -1.0
	else:
		self.lateral_cyclic = 0.0
	
	if Input.is_action_pressed("ui_up"):
		self.longitudinal_cyclic = 1.0
	elif Input.is_action_pressed("ui_down"):
		self.longitudinal_cyclic = -1.0
	else:
		self.longitudinal_cyclic = 0.0
	
	if Input.is_action_pressed("ui_page_up"):
		self.collective = 1.0
	elif Input.is_action_pressed("ui_page_down"):
		self.collective = -1.0
	else:
		self.collective = 0.0
	
	if Input.is_action_pressed("ui_alt_right"):
		self.tail_rotor_cyclic = -1.0
	elif Input.is_action_pressed("ui_alt_left"):
		self.tail_rotor_cyclic = 1.0
	else:
		self.tail_rotor_cyclic = 0.0
