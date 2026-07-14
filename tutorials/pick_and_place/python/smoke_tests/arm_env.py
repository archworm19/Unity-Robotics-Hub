"""Custom Box2D Gymnasium environment: a simplified 2D pick-and-place task,
used the same way lunar_lander_hil.py uses LunarLanderContinuous -- to
validate RLPD + human-in-the-loop machinery against manipulation specifically
(alignment, grasp, place), in a fast, visually-debuggable 2D sandbox, before
trusting the slow/finicky real Unity+IK setup with it. See arm_pygame.py for
the original design notes this implements.

Physics: Box2D, top-down (no gravity -- this is an overhead view of a table,
not a side view), arena bounds standing in for the table's edges.

Bodies:
  - the base: a single free-floating dynamic body, driven directly by applied
    force and torque -- there's deliberately no simulated arm linkage here,
    just a body that can translate and spin. Translation is forward/backward
    only (no strafe/lateral thrust) along the base's *own* forward axis, not
    world xy, rotated into world coordinates each substep (see step()) -- so
    it always tracks wherever the gripper currently faces, the same way
    LunarLander's main engine thrusts along the lander's own orientation
    rather than a fixed world direction. Full 2D positioning still works, the
    same way a car or tank does it: drive, rotate, drive again -- this is a
    deliberately simpler (nonholonomic) locomotion model than free
    omnidirectional translation, and one fewer action dimension to cover.
  - the gripper "jaws": not physics bodies at all, just two points offset
    from the base in its own local frame, spread apart by gripper_openness
    (see _jaw_positions). Purely geometric -- the grasp mechanic below
    doesn't need them to be physical, and giving them real collision fixtures
    would fight the alignment step (the block would just get pushed away
    before the geometric grip check ever triggers).
  - the block: a small dynamic square, free in the arena.
  - the goal: not a body either, just a stored (x, y) -- "inside the goal"
    is a plain geometric containment check against the block's vertices, the
    same rule-based philosophy as the grasp mechanic.

Grasp mechanic (deliberately rule-based, not physics/friction-based, to keep
it robust and easy to tune): each tick, while not already gripping, compute
every block vertex's distance to whichever jaw is nearest to it; if every
vertex is within GRIP_DISTANCE_METERS of its nearest jaw, weld the block
rigidly to the base (a real Box2D weld joint -- from that point on the block
moves as part of the base). Release happens the instant the gripper action
commands "open" while gripped (action[3] > 0) -- a direct response to intent,
not a distance or timing condition.

Reward mirrors rlpd_controller_v2.py's structure directly: sparse
SUCCESS_REWARD/FAILURE_REWARD (block fully inside the goal / block or base
pushed out of the arena) plus dense shaping -- HELD_REWARD/UNHELD_REWARD
(is_gripped stands in for "lifted", since there's no gravity/height here) and
per-meter penalties pulling the gripper toward the block and the block
toward the goal.

Timing: step()'s own dt (default TICK_INTERVAL_SECONDS, matching the HIL
driver's human-reaction-time cadence) is substepped internally at a stable
1/FPS -- so one env.step() call always advances exactly dt of *simulated*
time, keeping sim time and wall-clock time in sync for a human watching,
while each individual physics substep stays small enough for Box2D's
collision resolution to behave. A pure-autonomous trainer with no human to
watch could construct ArmEnv(dt=1/FPS) instead for much faster throughput.
"""

import numpy as np

try:
    import Box2D
    from Box2D.b2 import fixtureDef, polygonShape, weldJointDef
except ImportError as e:
    raise ImportError(
        'Box2D is not installed -- run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

import gymnasium as gym
from gymnasium import spaces

FPS = 50

VIEWPORT_SIZE_PIXELS = 600
SCALE_PIXELS_PER_METER = 30.0

ARENA_HALF_WIDTH_METERS = 8.0
ARENA_HALF_HEIGHT_METERS = 8.0

BASE_HALF_SIZE_METERS = 0.3
BLOCK_HALF_SIZE_METERS = 0.3
GOAL_HALF_SIZE_METERS = 0.6  # "slightly bigger" than the block, per arm_pygame.py

JAW_LENGTH_METERS = 0.9  # how far the (virtual) jaws sit out in front of the base
JAW_MAX_SPREAD_METERS = 0.5  # half-spread between the jaws at gripper_openness == 1 (fully open)
GRIP_DISTANCE_METERS = 0.5  # how close every block vertex must be to its nearest jaw to grip; must
# comfortably exceed the block's half-diagonal (BLOCK_HALF_SIZE_METERS * sqrt(2) =~ 0.42) -- when
# the jaws are fully closed they collapse to a single point, and *that* point still needs to be
# within range of the block's farthest corner for a fully-closed grab (the intuitive "align, then
# squeeze shut" motion) to actually trigger, not just a specific intermediate opening width.

MIN_BLOCK_GOAL_SEPARATION_METERS = 2.0  # reset() keeps the block and goal at least this far apart

MAIN_FORCE_NEWTONS = 3.0
TORQUE_SCALE_NEWTON_METERS = 0.15
GRIPPER_SPEED_PER_SECOND = 2.0  # fraction of full open<->closed range per second

MAX_LINEAR_VELOCITY_METERS_PER_SECOND = 1.5
MAX_ANGULAR_VELOCITY_RADIANS_PER_SECOND = 1.5

# See rlpd_controller_v2.py's identically-named constants -- same structure,
# ported over wholesale per the design discussion in arm_pygame.py.
SUCCESS_REWARD = 50.0
FAILURE_REWARD = -5.0
HELD_REWARD = 0.0
UNHELD_REWARD = -0.1
BLOCK_TO_GOAL_DISTANCE_PENALTY_PER_METER = -0.1
GRIPPER_TO_BLOCK_DISTANCE_PENALTY_PER_METER = -0.1

TICK_INTERVAL_SECONDS = 0.2  # see module docstring's "Timing" section
DEFAULT_MAX_EPISODE_STEPS = 500  # bumped alongside the velocity/force reductions above, so slower
# movement still has enough time (at TICK_INTERVAL_SECONDS=0.2, 300 steps == 60 seconds) to reach the goal

OBSERVATION_DIM = 20  # see _get_observation: gripper xy, base xy/orientation/velocity/angular
# velocity, gripper openness/is_gripped, block xy/orientation/velocity/angular velocity, goal xy
ACTION_DIM = 3  # (forward, angular_accel, gripper_delta) -- forward is in the base's own frame
# (forward == the jaws' own local +x), not world xy -- see step(). No lateral/strafe dimension --
# translation is forward/backward only, nonholonomic (see module docstring's "Bodies" section)


class ArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        dt: float = TICK_INTERVAL_SECONDS,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ) -> None:
        self.render_mode = render_mode
        self.dt = dt
        self._max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self.world = None
        self.base = None
        self.block = None
        self._block_local_vertices = None
        self._weld_joint = None
        self._is_gripped = False
        self._gripper_openness = 0.0
        self._goal_position = np.zeros(2, dtype=np.float32)
        self._step_count = 0

        self.screen = None
        self.surf = None

    @property
    def is_gripped(self) -> bool:
        return self._is_gripped

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.world = Box2D.b2World(gravity=(0, 0))
        self._weld_joint = None
        self._is_gripped = False
        self._gripper_openness = 0.0
        self._step_count = 0

        self.base = self.world.CreateDynamicBody(
            position=(0.0, 0.0),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(BASE_HALF_SIZE_METERS, BASE_HALF_SIZE_METERS)), density=1.0, friction=0.3
            ),
        )
        self.base.linearDamping = 0.5
        self.base.angularDamping = 0.5

        block_position = self._sample_arena_position(margin=BLOCK_HALF_SIZE_METERS)
        goal_position = self._sample_arena_position(margin=GOAL_HALF_SIZE_METERS)
        while np.linalg.norm(block_position - goal_position) < MIN_BLOCK_GOAL_SEPARATION_METERS:
            goal_position = self._sample_arena_position(margin=GOAL_HALF_SIZE_METERS)
        self._goal_position = goal_position

        self.block = self.world.CreateDynamicBody(
            position=(float(block_position[0]), float(block_position[1])),
            angle=float(self.np_random.uniform(-np.pi, np.pi)),
            fixtures=fixtureDef(
                shape=polygonShape(box=(BLOCK_HALF_SIZE_METERS, BLOCK_HALF_SIZE_METERS)), density=1.0, friction=0.5
            ),
        )
        self.block.linearDamping = 0.3
        self.block.angularDamping = 0.3
        self._block_local_vertices = self.block.fixtures[0].shape.vertices

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), {}

    def _sample_arena_position(self, margin: float) -> np.ndarray:
        return self.np_random.uniform(
            [-ARENA_HALF_WIDTH_METERS + margin, -ARENA_HALF_HEIGHT_METERS + margin],
            [ARENA_HALF_WIDTH_METERS - margin, ARENA_HALF_HEIGHT_METERS - margin],
        ).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        forward, angular_accel, gripper_delta = action

        if self._is_gripped and gripper_delta > 0:
            self._release()

        num_substeps = max(1, round(self.dt * FPS))
        substep_dt = 1.0 / FPS
        for _ in range(num_substeps):
            # forward is in the base's own frame (forward == the jaws' own
            # local +x, see _jaw_positions) -- GetWorldVector rotates that
            # into the world-frame vector ApplyForceToCenter actually needs,
            # so translation tracks wherever the gripper is currently facing,
            # the same way LunarLander's main engine does. No lateral
            # component -- see module docstring's "Bodies" section.
            local_force = (float(forward) * MAIN_FORCE_NEWTONS, 0.0)
            world_force = self.base.GetWorldVector(local_force)
            self.base.ApplyForceToCenter(world_force, True)
            self.base.ApplyTorque(float(angular_accel) * TORQUE_SCALE_NEWTON_METERS, True)
            self.world.Step(substep_dt, 6 * 30, 2 * 30)
            self._clamp_velocity(self.base)
            if not self._is_gripped:
                self._clamp_velocity(self.block)

        self._gripper_openness = float(
            np.clip(self._gripper_openness + gripper_delta * GRIPPER_SPEED_PER_SECOND * self.dt, 0.0, 1.0)
        )

        # gripper_delta <= 0 (closing, or just holding steady), not > 0: without
        # this, releasing (gripper_delta > 0) on a tick where the base hasn't
        # moved away yet would immediately re-satisfy the geometric grip check
        # and re-grip in that same tick, making release impossible in practice.
        if not self._is_gripped and gripper_delta <= 0:
            self._try_grip()

        self._step_count += 1
        reward, terminated, success = self._compute_reward()
        truncated = self._step_count >= self._max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, {"success": success}

    @staticmethod
    def _clamp_velocity(body) -> None:
        speed = np.linalg.norm(body.linearVelocity)
        if speed > MAX_LINEAR_VELOCITY_METERS_PER_SECOND:
            body.linearVelocity *= MAX_LINEAR_VELOCITY_METERS_PER_SECOND / speed
        if abs(body.angularVelocity) > MAX_ANGULAR_VELOCITY_RADIANS_PER_SECOND:
            body.angularVelocity = MAX_ANGULAR_VELOCITY_RADIANS_PER_SECOND * np.sign(body.angularVelocity)

    def _jaw_positions(self) -> tuple[np.ndarray, np.ndarray]:
        spread = self._gripper_openness * JAW_MAX_SPREAD_METERS
        jaw1 = self.base.GetWorldPoint((JAW_LENGTH_METERS, spread))
        jaw2 = self.base.GetWorldPoint((JAW_LENGTH_METERS, -spread))
        return np.array(jaw1), np.array(jaw2)

    def _block_vertices(self) -> list[np.ndarray]:
        return [np.array(self.block.GetWorldPoint(v)) for v in self._block_local_vertices]

    def _try_grip(self) -> None:
        jaw1, jaw2 = self._jaw_positions()
        nearest_jaw_distances = [
            min(np.linalg.norm(vertex - jaw1), np.linalg.norm(vertex - jaw2)) for vertex in self._block_vertices()
        ]
        if max(nearest_jaw_distances) <= GRIP_DISTANCE_METERS:
            weld_def = weldJointDef()
            weld_def.Initialize(self.base, self.block, self.block.position)
            self._weld_joint = self.world.CreateJoint(weld_def)
            self._is_gripped = True

    def _release(self) -> None:
        self.world.DestroyJoint(self._weld_joint)
        self._weld_joint = None
        self._is_gripped = False

    def _out_of_bounds(self, body) -> bool:
        x, y = body.position
        return abs(x) > ARENA_HALF_WIDTH_METERS or abs(y) > ARENA_HALF_HEIGHT_METERS

    def _block_fully_inside_goal(self) -> bool:
        gx, gy = self._goal_position
        return all(
            gx - GOAL_HALF_SIZE_METERS <= vx <= gx + GOAL_HALF_SIZE_METERS
            and gy - GOAL_HALF_SIZE_METERS <= vy <= gy + GOAL_HALF_SIZE_METERS
            for vx, vy in self._block_vertices()
        )

    def _gripper_position(self) -> np.ndarray:
        """Midpoint between the two jaws -- the natural single "where is the
        gripper" point, distinct from the base's own position by a fixed
        offset along wherever the base currently faces (see _jaw_positions)."""
        jaw1, jaw2 = self._jaw_positions()
        return (jaw1 + jaw2) / 2.0

    def _compute_reward(self) -> tuple[float, bool, bool]:
        """Returns (reward, terminated, success)."""
        if self._out_of_bounds(self.block) or self._out_of_bounds(self.base):
            return FAILURE_REWARD, True, False
        if self._block_fully_inside_goal():
            return SUCCESS_REWARD, True, True

        gripper_position = self._gripper_position()
        block_position = np.array(self.block.position)
        block_to_goal_distance = float(np.linalg.norm(block_position - self._goal_position))
        gripper_to_block_distance = float(np.linalg.norm(gripper_position - block_position))

        held_term = HELD_REWARD if self._is_gripped else UNHELD_REWARD
        distance_term = (
            BLOCK_TO_GOAL_DISTANCE_PENALTY_PER_METER * block_to_goal_distance
            + GRIPPER_TO_BLOCK_DISTANCE_PENALTY_PER_METER * gripper_to_block_distance
        )
        return held_term + distance_term, False, False

    def _get_observation(self) -> np.ndarray:
        gripper_position = self._gripper_position()
        return np.array(
            [
                gripper_position[0],
                gripper_position[1],
                self.base.position[0],
                self.base.position[1],
                np.cos(self.base.angle),
                np.sin(self.base.angle),
                self.base.linearVelocity[0],
                self.base.linearVelocity[1],
                self.base.angularVelocity,
                self._gripper_openness,
                float(self._is_gripped),
                self.block.position[0],
                self.block.position[1],
                np.cos(self.block.angle),
                np.sin(self.block.angle),
                self.block.linearVelocity[0],
                self.block.linearVelocity[1],
                self.block.angularVelocity,
                self._goal_position[0],
                self._goal_position[1],
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode is None:
            return

        import pygame

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_SIZE_PIXELS, VIEWPORT_SIZE_PIXELS))
            pygame.display.set_caption("ArmEnv")

        self.surf = pygame.Surface((VIEWPORT_SIZE_PIXELS, VIEWPORT_SIZE_PIXELS))
        self.surf.fill((255, 255, 255))

        def to_screen(world_xy) -> tuple[float, float]:
            return (
                VIEWPORT_SIZE_PIXELS / 2 + world_xy[0] * SCALE_PIXELS_PER_METER,
                VIEWPORT_SIZE_PIXELS / 2 + world_xy[1] * SCALE_PIXELS_PER_METER,
            )

        def draw_square(center, half_size, angle, color, width=0):
            local = [(-half_size, -half_size), (half_size, -half_size), (half_size, half_size), (-half_size, half_size)]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            points = [
                to_screen((center[0] + lx * cos_a - ly * sin_a, center[1] + lx * sin_a + ly * cos_a))
                for lx, ly in local
            ]
            pygame.draw.polygon(self.surf, color, points, width)

        draw_square(self._goal_position, GOAL_HALF_SIZE_METERS, 0.0, (200, 230, 200), width=0)
        draw_square(self._goal_position, GOAL_HALF_SIZE_METERS, 0.0, (60, 150, 60), width=2)
        draw_square(self.block.position, BLOCK_HALF_SIZE_METERS, self.block.angle, (210, 140, 60))
        draw_square(self.base.position, BASE_HALF_SIZE_METERS, self.base.angle, (60, 60, 200))

        jaw1, jaw2 = self._jaw_positions()
        jaw_color = (200, 60, 60) if self._is_gripped else (60, 60, 60)
        pygame.draw.line(self.surf, jaw_color, to_screen(self.base.position), to_screen(jaw1), 3)
        pygame.draw.line(self.surf, jaw_color, to_screen(self.base.position), to_screen(jaw2), 3)

        # to_screen maps world +y to increasing screen-y (downward, pygame's
        # native convention) -- without this, the whole scene renders
        # vertically mirrored relative to the standard math convention
        # (world +y up), which makes a mathematically-CCW rotation (positive
        # angular velocity) visually appear clockwise. Same fix lunar_lander.py
        # applies for the same reason.
        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
