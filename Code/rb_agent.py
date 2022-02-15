from lux.game import game_state
from utils import *

def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    map = game_state.map
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    
    # add debug statements like so!
    if game_state.turn == 0:
        print("Agent is running!", file=sys.stderr)

    resource_tiles = find_resources(game_state)
    
    next_pos = set()
    citytilesPos = set([citytile.pos for citytile in player.citytiles])
    for unit in player.units:
        # if the unit is a worker (can mine resources) and can perform an action this turn
        #if unit.is_worker() and unit.can_act():
        if unit.can_act():
            if game_state.turn % 40 > 27:
                closest_city_tile = find_closest_city_tile(unit.pos, player)
                if closest_city_tile is not None:
                    # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                    action = unit.move(unit.pos.direction_to(closest_city_tile.pos))
                    actions.append(action)
                    continue
            # we want to mine only if there is space left in the worker's cargo
            if unit.get_cargo_space_left() > 0:
                # find the closest resource if it exists to this unit
                closest_resource_tile = find_closest_resources(unit.pos, player, resource_tiles)
                if closest_resource_tile is not None:
                    # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                    path = find_path(unit.pos, closest_resource_tile.pos, next_pos - citytilesPos)
                    if path:
                        action = unit.move(path[0])
                        next_pos.add(unit.pos.translate(path[0], 1))
                        actions.append(action)
                    else:
                        next_pos.add(unit.pos)
            else:
                if game_state.turn % 40 < 30 and safe_to_build(player):
                    if unit.can_build(map) and next_to_city(unit.pos, player):
                        action = unit.build_city()
                        actions.append(action)
                        continue
                    else:
                        closest_suburb = find_closest_suburb(unit.pos, player)
                        if closest_suburb is not None:
                            path = find_path(unit.pos, closest_suburb, avoid = next_pos | citytilesPos)
                            if path:
                                action = unit.move(path[0])
                                next_pos.add(unit.pos.translate(path[0], 1))
                                actions.append(action)
                                continue
                            else:
                                next_pos.add(unit.pos)
                # find the closest citytile and move the unit towards it to drop resources to a citytile to fuel the city
                closest_city_tile = find_closest_city_tile(unit.pos, player)
                if closest_city_tile is not None:
                    # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                    action = unit.move(unit.pos.direction_to(closest_city_tile.pos))
                    actions.append(action)
         
    population = len(player.units)
    population_cell = len(player.citytiles)
    for k, city in player.cities.items():
        if city.fuel < 120 * population_cell:
            for citytile in city.citytiles:
                if citytile.can_act() and population < population_cell:
                    action = citytile.build_worker()
                    population += 1
                    actions.append(action)
        else:
            for citytile in city.citytiles:
                if citytile.can_act():
                    action = citytile.research()
                    actions.append(action)
    
    
    return actions
