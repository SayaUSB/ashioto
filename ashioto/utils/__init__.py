# Copyright (c) 2025 ICHIRO ITS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

# Footsteps Planning Environments
register(
    id="footsteps-planning-right-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-left-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-any-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-right-withball-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightWithBallEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-left-withball-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftWithBallEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-any-withball-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyWithBallEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-right-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-left-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-any-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-right-withball-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightWithBallMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-left-withball-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftWithBallMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-any-withball-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyWithBallMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-right-obstacle-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightObstacleMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-left-obstacle-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftObstacleMultiGoalEnv",
    max_episode_steps=200,
)

register(
    id="footsteps-planning-any-obstacle-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyObstacleMultiGoalEnv",
    max_episode_steps=200,
)
