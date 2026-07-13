

# Lets make our own box2d gymnasium environment!

# 2D Game proposal
# arm control: control base and gripper
#   > 3d: x and y translation accelerations
#           clockwise vs. counterclockwise acceleration
#       game should bound max velocity to be somewhat small
#
#   > parallel jaw gripper: 1d continuous open vs. close
# 
#   can render as 2 jaws coming out of a 
#   rectangular base
# 
#
# pick and place objective (all 2d):
# > pick up square block by aligning jaws
#       and closing gripper
# > only "picks" if the angle between
#       the grippers and the block is small
# > place block on slightly bigger block.
#       succeed when small block is 100% inside
#       big block.

# Q? does box2d give us some basic physics
#       or do we have to implement for ourselves?