

import random
import contest.util as util
import time
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point
import numpy as np


AGENT_ROLES = {}  # tracks current roles of agents


# messing with ImprovedAgent to add role switchin abilities
def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    makes a team with two agents
    """
    return [eval(first)(first_index), eval(second)(second_index)]


class ImprovedAgent(CaptureAgent):
    """
    base class for the better agents
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.last_observed_food = {}
        self.food_was_eaten = {}
        self.last_score = 0
        self.patrol_points = []
        self.initial_food_count = 0
        self.move_counter = 0
        # stuff for role switching
        self.original_role = None
        self.current_role = None
        self.blocked_count = 0
        self.switch_cooldown = 0
        self.last_position = None
        self.recent_positions = []  # store last N positions
        self.exploration_mode = False
        self.exploration_counter = 0


    def register_initial_state(self, game_state):
        """
        sets up the agent and figures out the key spots
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # figurin out the middle of the map
        self.mid_width = game_state.data.layout.width // 2
        self.mid_height = game_state.data.layout.height // 2

        # checkin which side is ours and which aint
        if self.red:
            self.our_side = lambda x: x < self.mid_width
            self.enemy_side = lambda x: x >= self.mid_width
        else:
            self.our_side = lambda x: x >= self.mid_width
            self.enemy_side = lambda x: x < self.mid_width

        # makin patrol points at the border
        self.calculate_patrol_points(game_state)

        # keepin an eye on the food
        self.observe_food(game_state)

        # remembrin how much food we got at first
        if self.red:
            self.initial_food_count = len(self.get_food(game_state).as_list())
        else:
            self.initial_food_count = len(self.get_food(game_state).as_list())

        # setup the roles dictionary
        global AGENT_ROLES
        if self.original_role is not None:
            AGENT_ROLES[self.index] = self.original_role

    def calculate_patrol_points(self, game_state):
        """
        wrks out spots to patrol on our territory border
        """
        layout = game_state.data.layout
        border_x = self.mid_width - 1 if self.red else self.mid_width

        # lookin for spots we can actually walk through
        self.patrol_points = []
        for y in range(layout.height):
            if not layout.walls[border_x][y]:
                self.patrol_points.append((border_x, y))

        # if we cant find any good spots just use where we started
        if not self.patrol_points:
            self.patrol_points = [self.start]

    def observe_food(self, game_state):
        """
        keeps track of changes in food locations
        """
        # what food exists on enemy turf right now
        current_food = self.get_food(game_state).as_list()

        # comparin to what we saw before
        if hasattr(self, 'previous_food') and self.previous_food:
            for food in self.previous_food:
                if food not in current_food:
                    self.food_was_eaten[food] = self.move_counter

        self.previous_food = current_food

    def choose_action(self, game_state):
        """
        choose the best action using A* for critical situations and falling back to feature evaluation
        """
        # countin moves and keepin track of where we been
        self.move_counter += 1
        current_pos = game_state.get_agent_state(self.index).get_position()

        # settin up position history if we dont have one
        if not hasattr(self, 'position_history'):
            self.position_history = []
            self.exploration_mode = False
            self.exploration_counter = 0

        # checkin if we re goin in circles
        loop_detected = False
        if current_pos in self.position_history[-6:]:
            recent_visits = self.position_history[-10:].count(current_pos)
            if recent_visits >= 3:
                loop_detected = True
                self.exploration_mode = True
                self.exploration_counter = 5

        # addin current pos to history
        self.position_history.append(current_pos)
        if len(self.position_history) > 20:
            self.position_history.pop(0)

        # seein if food changed
        self.observe_food(game_state)

        # maybe switchin roles
        if hasattr(self, 'current_role') and self.current_role is not None:
            self.check_and_switch_roles(game_state)

        # checkin if we re in big trouble
        if self.is_critical_situation(game_state):
            goal_positions = self.get_critical_goal_positions(game_state)
            best_action = self.astar_search(game_state, goal_positions)

            # if A* found somethin good, let's do it
            if best_action and best_action in game_state.get_legal_actions(self.index):
                if loop_detected:
                    print(f"Agent {self.index} detected loop, using A* to escape")
                return best_action

        # if not in trouble just use the usual way to decide
        actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # figurin out the best move
        start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        # addin penalties for goin back to same spots
        if self.exploration_mode:
            for i, action in enumerate(actions):
                successor = self.get_successor(game_state, action)
                next_pos = successor.get_agent_state(self.index).get_position()

                if next_pos in self.position_history[-10:]:
                    visits = self.position_history[-10:].count(next_pos)
                    values[i] -= 20 * visits

            self.exploration_counter -= 1
            if self.exploration_counter <= 0:
                self.exploration_mode = False

        # pickin the best action
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # special case when only a lil food left
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        if loop_detected:
            print(f"agent {self.index} detected loop at position {current_pos}, usingg exploration mode")

        return random.choice(best_actions)

    def astar_search(self, game_state, goal_positions, excluded_positions=None):
        """
        A* search to find the best path to one of the goal positions
        it returnss the first action in that path
        """
        # settin up our search stuff
        start_pos = game_state.get_agent_state(self.index).get_position()
        pq = util.PriorityQueue()
        visited = set()

        # addin positions we wanna avoid
        if excluded_positions:
            visited.update(excluded_positions)

        # puttin start pos in the queue
        pq.push((start_pos, [], 0), 0)  # (position, path, cost)

        # if we re already at the goal just chilling
        if start_pos in goal_positions:
            return Directions.STOP

        # doin the A* search
        while not pq.heap:
            pos, path, cost = pq.pop()

            # if we found a goal return first step
            if pos in goal_positions:
                if not path:
                    return Directions.STOP
                return path[0]

            # skip if we already checked this spot
            if pos in visited:
                continue

            # mark as visited
            visited.add(pos)

            # gettin legal moves from here
            x, y = int(pos[0]), int(pos[1])
            legal_actions = []
            for action, (dx, dy) in [
                (Directions.NORTH, (0, 1)),
                (Directions.SOUTH, (0, -1)),
                (Directions.EAST, (1, 0)),
                (Directions.WEST, (-1, 0))
            ]:
                next_x, next_y = x + dx, y + dy
                if not game_state.data.layout.walls[next_x][next_y]:
                    legal_actions.append((action, (next_x, next_y)))

            # addin neighbors to the queue
            for action, next_pos in legal_actions:
                if next_pos not in visited:
                    # calculatin heuristic == closest distance to any goal
                    h = min([self.get_maze_distance(next_pos, goal) for goal in goal_positions])
                    new_cost = cost + 1
                    new_path = path + [action]
                    pq.push((next_pos, new_path, new_cost), new_cost + h)

        # if no path found thans just return None
        return None

    def is_critical_situation(self, game_state):
        """
        determines if the current state is a critical situation where A* should be used
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # case 1 == carryin lots of food and needin to get home
        carrying = my_state.num_carrying
        if carrying >= 5:
            return True

        # case 2 ==  ghost gettin too close
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None
                  and e.scared_timer <= 2]  # not scared ghosts

        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            if min_ghost_dist <= 3:  # ghost is dangrusly close
                return True

        # case 3 == we re goin in circles
        if hasattr(self, 'exploration_mode') and self.exploration_mode:
            return True

        # case 4 == times almost up and we got food
        if carrying > 0 and game_state.data.timeleft < 100:
            return True

        return False

    def get_critical_goal_positions(self, game_state):
        """
        returns goal positions for A* based on the critical situation
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying

        # case 1 == gettin home with food
        if carrying > 0:
            # findin border spots to return home
            home_positions = []
            border_x = self.mid_width - 1 if self.red else self.mid_width
            for y in range(game_state.data.layout.height):
                if not game_state.data.layout.walls[border_x][y]:
                    home_positions.append((border_x, y))
            if home_positions:
                return home_positions

        # case 2 == runnin from ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None
                  and e.scared_timer <= 2]

        if ghosts:
            # findin safe spots away from ghosts
            candidates = []
            for x in range(game_state.data.layout.width):
                for y in range(game_state.data.layout.height):
                    if not game_state.data.layout.walls[x][y]:
                        pos = (x, y)
                        min_ghost_dist = min([self.get_maze_distance(pos, g.get_position()) for g in ghosts])
                        # only consider spots safer than where we re
                        current_min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
                        if min_ghost_dist > current_min_ghost_dist:
                            candidates.append((pos, min_ghost_dist))

            if candidates:
                # sortin by distance from ghosts (furthest first)
                candidates.sort(key=lambda x: (-x[1], self.get_maze_distance(my_pos, x[0])))
                # return the top 3 safest spots
                return [pos for pos, _ in candidates[:3]]

        # case 3 == breakin out of loops so find spots we havent been
        if hasattr(self, 'exploration_mode') and self.exploration_mode:
            # findin positions we havent visited lately
            candidates = []
            for x in range(game_state.data.layout.width):
                for y in range(game_state.data.layout.height):
                    if not game_state.data.layout.walls[x][y]:
                        pos = (x, y)
                        if pos not in self.position_history[-10:] and self.get_maze_distance(my_pos, pos) < 10:
                            candidates.append(pos)

            if candidates:
                # sortin by distance (closest first)
                candidates.sort(key=lambda x: self.get_maze_distance(my_pos, x))
                return candidates[:3]

        # default == if no specific goal just go for food or capsules
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        if food_list:
            # return the closest 3 food spots
            food_distances = [(food, self.get_maze_distance(my_pos, food)) for food in food_list]
            food_distances.sort(key=lambda x: x[1])
            return [food for food, _ in food_distances[:3]]

        if capsules:
            return capsules

        # if nothin else  patrol the border
        return self.patrol_points

    def check_and_switch_roles(self, game_state):
        """
        checks if roles should be switched with the partner agent
        """
        # waitin for cooldown if its active
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1
            return

        # only switch roles after we ve played a bit
        if self.move_counter < 20:
            return

        # gettin my position and state
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos is None:  # might be if we just got eaten
            return

        # findin our partner
        team_indices = self.get_team(game_state)
        partner_index = [i for i in team_indices if i != self.index][0]

        # gettin partner info if available
        partner_state = game_state.get_agent_state(partner_index)
        partner_pos = partner_state.get_position()
        if partner_pos is None:  # partner might be eaten
            return

        # border location
        border_x = self.mid_width - 1 if self.red else self.mid_width

        # checkin if im offensive and stuck at border
        if self.current_role == "Offensive":
            # checkin if at border
            at_border = abs(my_pos[0] - border_x) <= 1
            on_our_side = (self.red and my_pos[0] < self.mid_width) or (not self.red and my_pos[0] >= self.mid_width)

            if at_border and on_our_side:
                # checkin for enemy defense agents nearby
                enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                blockers = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

                if blockers:
                    min_blocker_dist = min([self.get_maze_distance(my_pos, e.get_position()) for e in blockers])

                    # if enemy is close and we re stuck
                    if min_blocker_dist <= 2:
                        if self.last_position and self.get_maze_distance(my_pos, self.last_position) <= 1:
                            self.blocked_count += 1
                        else:
                            self.blocked_count = 0

                        # if blocked for a while check if partner can help
                        if self.blocked_count >= 3 and partner_index in AGENT_ROLES:
                            # partner must be defensive and on our side
                            partner_on_our_side = (self.red and partner_pos[0] < self.mid_width) or \
                                                  (not self.red and partner_pos[0] >= self.mid_width)

                            if AGENT_ROLES[partner_index] == "Defensive" and partner_on_our_side:
                                # check if partner is just patrollin (no invaders)
                                enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                                invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

                                if not invaders:
                                    # lets switch roles
                                    self.switch_roles_with_partner(game_state)

            self.last_position = my_pos

        # checkin if im defensive and free while partner might be stuck
        elif self.current_role == "Defensive":
            # only when on our side
            on_our_side = (self.red and my_pos[0] < self.mid_width) or (not self.red and my_pos[0] >= self.mid_width)

            if on_our_side:
                # checkin for invaders
                enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

                if not invaders:
                    # we re free to switch == check if partner is stuck at border
                    partner_at_border = abs(partner_pos[0] - border_x) <= 1
                    partner_on_our_side = (self.red and partner_pos[0] < self.mid_width) or \
                                          (not self.red and partner_pos[0] >= self.mid_width)

                    if partner_at_border and partner_on_our_side and partner_index in AGENT_ROLES and AGENT_ROLES[
                        partner_index] == "Offensive":
                        # check if enemies near partner
                        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                        blockers = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

                        if blockers:
                            min_blocker_dist = min(
                                [self.get_maze_distance(partner_pos, e.get_position()) for e in blockers])
                            if min_blocker_dist <= 2:
                                # partner is prolly stuck les switch
                                self.switch_roles_with_partner(game_state)

    def switch_roles_with_partner(self, game_state):
        """
        switches roles with partner agent
        """
        global AGENT_ROLES

        # gettin partner index
        team_indices = self.get_team(game_state)
        partner_index = [i for i in team_indices if i != self.index][0]

        # switchin my role
        old_role = self.current_role
        new_role = "Defensive" if self.current_role == "Offensive" else "Offensive"
        self.current_role = new_role

        # updatin the shared roles dictionary
        AGENT_ROLES[self.index] = new_role
        AGENT_ROLES[partner_index] = "Offensive" if new_role == "Defensive" else "Defensive"

        # resettin cooldown and blocked count
        self.switch_cooldown = 20
        self.blocked_count = 0

        print(f"Agent {self.index} switched from {old_role} to {new_role}")

    def get_successor(self, game_state, action):
        """
        gets the successor state after doin an action
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # only moved half a grid cell
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        calculates how good a move is based on features
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        figures out features for evaluatin an action
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        returns weights for the features
        """
        return {'successor_score': 1.0}

    def get_enemy_positions(self, game_state):
        """
        gets known enemy positions
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_enemies = [e for e in enemies if e.get_position() is not None]
        return [(e.get_position(), e.is_pacman) for e in visible_enemies]

    def closest_food(self, pos, food_list, game_state):
        """
        finds closest food and how far away it is
        """
        if not food_list:
            return None, float('inf')

        food_distances = [(food, self.get_maze_distance(pos, food)) for food in food_list]
        return min(food_distances, key=lambda x: x[1])

    def is_dead_end(self, pos, game_state):
        """
        checks if a spot is a dead end
        """
        walls = game_state.get_walls()
        # usin integers to avoid messy float stuff
        x, y = int(pos[0]), int(pos[1])
        valid_directions = 0

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            # checkin if we can go that way
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                valid_directions += 1

        return valid_directions <= 1

    def get_safe_food(self, my_pos, food_list, game_state):
        """
        figures out which food is safe to grab
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

        if not ghosts or not food_list:
            return food_list

        safe_food = []
        for food in food_list:
            food_to_me = self.get_maze_distance(my_pos, food)
            min_ghost_dist = min([self.get_maze_distance(food, g.get_position()) for g in ghosts])

            # food is safe if we can get there and back before ghosts catch us
            if food_to_me < min_ghost_dist - 1:
                safe_food.append(food)

        return safe_food if safe_food else food_list

    def get_scared_time(self, game_state, enemy_index):
        """
        returns how long an enemy is scared for
        """
        return game_state.get_agent_state(enemy_index).scared_timer

    def is_late_game(self):
        """
        checks if we re in the late game
        """
        # late game is when w ve used up most of our moves
        return self.move_counter > 900


class OffensiveAgent(ImprovedAgent):
    """
    agent that focuses on gettin food from enemy territory
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.original_role = "Offensive"
        self.current_role = "Offensive"

    def get_features(self, game_state, action):
        # checkin if we re offense or defense right now
        if hasattr(self, 'current_role') and self.current_role == "Defensive":
            # use defensive features when in defensive role
            return self.get_defensive_features(game_state, action)
        else:
            # default to offensive features
            return self.get_offensive_features(game_state, action)

    def get_weights(self, game_state, action):
        # checkin if we re offense or defense right now
        if hasattr(self, 'current_role') and self.current_role == "Defensive":
            # use defensive weights when in defensive role
            return self.get_defensive_weights(game_state, action)
        else:
            # default to offensive weights
            return self.get_offensive_weights(game_state, action)

    def get_offensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # basic features
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # distance to closest food
        if len(food_list) > 0:
            closest_food, min_distance = self.closest_food(my_pos, food_list, successor)
            features['distance_to_food'] = min_distance

        # distance to enemies
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

        # stayin away from ghosts
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            if min_ghost_dist <= 1:
                features['ghost_distance'] = -1
            else:
                features['ghost_distance'] = min_ghost_dist

        # power capsules
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_capsule_dist

        # feature for goin home with food
        carrying = my_state.num_carrying
        if carrying > 0:
            closest_home_dist = float('inf')
            border_x = self.mid_width - 1 if self.red else self.mid_width
            for y in range(successor.data.layout.height):
                if not successor.data.layout.walls[border_x][y]:
                    dist = self.get_maze_distance(my_pos, (border_x, y))
                    if dist < closest_home_dist:
                        closest_home_dist = dist

            # motivate goin home more when we got more food
            features['return_home'] = closest_home_dist * carrying

            # if carryin lots of food or ghosts nearby  really wanna go home
            if carrying > 5 or (ghosts and min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]) < 5):
                features['return_home'] *= 2

        # feature for avoidin dead ends
        if self.is_dead_end(my_pos, successor):
            features['dead_end'] = 1

        return features

    def get_offensive_weights(self, game_state, action):
        # basic weights
        weights = {'successor_score': 100, 'distance_to_food': -2, 'ghost_distance': 2,
                   'distance_to_capsule': -3, 'return_home': -1, 'dead_end': -10}

        # tweak weights based on situation
        my_state = game_state.get_agent_state(self.index)

        # if we got lots of food focus on gettin home
        if my_state.num_carrying > 5:
            weights['return_home'] = -5
            weights['distance_to_food'] = -1

        # if runnin out of time focus on grabbin food fast
        if self.is_late_game():
            weights['successor_score'] = 200

        return weights

    def get_defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # checkin if we re in defense mode
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # distance to invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # if there invaders go after the closest one
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # if no invaders patrol the border or check where food was eaten
            patrol_distances = [self.get_maze_distance(my_pos, p) for p in self.patrol_points]
            features['patrol_distance'] = min(patrol_distances)

            # checkin if food was eaten recently
            recent_eaten = []
            for pos, time_eaten in self.food_was_eaten.items():
                if self.move_counter - time_eaten < 40:  # food eaten recently
                    recent_eaten.append(pos)

            if recent_eaten:
                eaten_distances = [self.get_maze_distance(my_pos, pos) for pos in recent_eaten]
                features['food_eaten_distance'] = min(eaten_distances)

        # dont like stoppin or goin backwards
        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # reward stayin on our side
        x, y = my_pos
        if (self.red and x < self.mid_width) or (not self.red and x >= self.mid_width):
            features['on_our_side'] = 1
        else:
            features['on_our_side'] = 0

        return features

    def get_defensive_weights(self, game_state, action):
        # basic weights
        weights = {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10,
                   'stop': -100, 'reverse': -2, 'patrol_distance': -1, 'food_eaten_distance': -5,
                   'on_our_side': 50}

        # if no invaders patrol more aggressively
        features = self.get_defensive_features(game_state, action)
        if 'invader_distance' not in features:
            weights['patrol_distance'] = -10
        else:
            weights['patrol_distance'] = 0

        return weights


class DefensiveAgent(ImprovedAgent):
    """
    agent that focuses on protectin our territory
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.original_role = "Defensive"
        self.current_role = "Defensive"

    def get_features(self, game_state, action):
        # checkin if we re offense or defense right now
        if hasattr(self, 'current_role') and self.current_role == "Offensive":
            # use offensive features when in offensive role
            return self.get_offensive_features(game_state, action)
        else:
            # default to defensive features
            return self.get_defensive_features(game_state, action)

    def get_weights(self, game_state, action):
        # checkin if we re offense or defense right now
        if hasattr(self, 'current_role') and self.current_role == "Offensive":
            # use offensive weights when in offensive role
            return self.get_offensive_weights(game_state, action)
        else:
            # default to defensive weights
            return self.get_defensive_weights(game_state, action)

    def get_defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # checkin if we re in defense mode
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # distance to invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # if theres invaders go after the closest one
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # if no invaders patrol the border or check where food was eaten
            patrol_distances = [self.get_maze_distance(my_pos, p) for p in self.patrol_points]
            features['patrol_distance'] = min(patrol_distances)

            # checkin if food was eaten recently
            recent_eaten = []
            for pos, time_eaten in self.food_was_eaten.items():
                if self.move_counter - time_eaten < 40:  # food eaten recently
                    recent_eaten.append(pos)

            if recent_eaten:
                eaten_distances = [self.get_maze_distance(my_pos, pos) for pos in recent_eaten]
                features['food_eaten_distance'] = min(eaten_distances)

        # dont like stoppin or goin backwards
        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # reward stayin on our side
        x, y = my_pos
        if (self.red and x < self.mid_width) or (not self.red and x >= self.mid_width):
            features['on_our_side'] = 1
        else:
            features['on_our_side'] = 0

        return features

    def get_defensive_weights(self, game_state, action):
        # basic weights
        weights = {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10,
                   'stop': -100, 'reverse': -2, 'patrol_distance': -1, 'food_eaten_distance': -5,
                   'on_our_side': 50}

        # if no invaders patrol more aggressively
        if 'invader_distance' not in self.get_defensive_features(game_state, action):
            weights['patrol_distance'] = -10
        else:
            weights['patrol_distance'] = 0

        return weights

    def get_offensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # basic features
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # distance to closest food
        if len(food_list) > 0:
            closest_food, min_distance = self.closest_food(my_pos, food_list, successor)
            features['distance_to_food'] = min_distance

        # distance to enemies
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

        # stayin away from ghosts
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            if min_ghost_dist <= 1:
                features['ghost_distance'] = -1
            else:
                features['ghost_distance'] = min_ghost_dist

        # power capsules
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_capsule_dist

        # feature for goin home with food
        carrying = my_state.num_carrying
        if carrying > 0:
            closest_home_dist = float('inf')
            border_x = self.mid_width - 1 if self.red else self.mid_width
            for y in range(successor.data.layout.height):
                if not successor.data.layout.walls[border_x][y]:
                    dist = self.get_maze_distance(my_pos, (border_x, y))
                    if dist < closest_home_dist:
                        closest_home_dist = dist

            # motivate goin home more when we got more food
            features['return_home'] = closest_home_dist * carrying

            # if carryin lots of food or ghosts nearby really wanna go home
            if carrying > 5 or (ghosts and min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]) < 5):
                features['return_home'] *= 2

        # feature for avoidin dead ends
        if self.is_dead_end(my_pos, successor):
            features['dead_end'] = 1

        return features

    def get_offensive_weights(self, game_state, action):
        # basic weights
        weights = {'successor_score': 100, 'distance_to_food': -2, 'ghost_distance': 2,
                   'distance_to_capsule': -3, 'return_home': -1, 'dead_end': -10}

        # tweak weights based on situation
        my_state = game_state.get_agent_state(self.index)

        # if we got lots of food focus on gettin home
        if my_state.num_carrying > 5:
            weights['return_home'] = -5
            weights['distance_to_food'] = -1

        # if runnin out of time focus on grabbin food fast
        if self.is_late_game():
            weights['successor_score'] = 200

        return weights