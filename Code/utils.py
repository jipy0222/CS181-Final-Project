from lux.game import Game, game_state
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import sys

allDIRECTIONS = [Constants.DIRECTIONS.NORTH, Constants.DIRECTIONS.EAST, Constants.DIRECTIONS.SOUTH, Constants.DIRECTIONS.WEST, Constants.DIRECTIONS.CENTER]
moveDIRECTIONS = [Constants.DIRECTIONS.NORTH, Constants.DIRECTIONS.EAST, Constants.DIRECTIONS.SOUTH, Constants.DIRECTIONS.WEST]

# this snippet finds all resources stored on the map and puts them into a list so we can search over them
def find_resources(game_state):
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles

# the next snippet finds the closest resources that we can mine given position on a map
def find_closest_resources(pos, player, resource_tiles):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        # we skip over resources that we can't mine due to not having researched them
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal():continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile

# snippet to find the closest city tile to a position
def find_closest_city_tile(pos, player):
    closest_city_tile = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        # the cities are stored as a dictionary mapping city id to the city object, which has a citytiles field that
        # contains the information of all citytiles in that city
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
    return closest_city_tile

def find_min_fuel_city(player):
    min_fuel_city = None
    min_fuel = math.inf
    if len(player.cities) > 0:
        for k, city in player.cities.items():
            fuel = city.fuel
            if fuel < min_fuel:
                min_fuel = fuel
                min_fuel_city = city
    return min_fuel_city

def find_suburb(player):
    suburbs = set()
    if len(player.cities) > 0:
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                suburbs |= neighbourhoods(city_tile.pos)
    suburbs_with_resource = set()
    for suburb in suburbs:
        if game_state.map.get_cell_by_pos(suburb).has_resource():
            suburbs_with_resource.add(suburb)
    return suburbs - suburbs_with_resource - set([citytile.pos for citytile in player.citytiles])

def find_closest_suburb(pos, player):
    closest_suburb = None
    suburbs = find_suburb(player)
    if len(suburbs) > 0:
        closest_dist = math.inf
        for suburb in suburbs:
            dist = suburb.distance_to(pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_suburb = suburb
    return closest_suburb

# all empty position next to a city
def neighbourhoods(pos):
    neighbourhoods = set()
    for direction in allDIRECTIONS:
        neighbourhoods.add(pos.translate(direction, 1))
    return neighbourhoods

def unit_num(player):
    return len(player.units)

def city_num(player):
    return len(player.cities)

def get_citytiles(player):
    return len(player.citytiles)

def find_path(start, end, avoid = None):
    l = []
    l.append(start)
    m = {}
    m[start] = (None, None)
    width, height = game_state.map.width, game_state.map.height
    while l:
        curentPos = l.pop()
        if curentPos == end:
            actions = []
            while curentPos != start:
                curentPos, action = m[curentPos]
                actions.append(action)
            return actions[::-1]
        for direction in moveDIRECTIONS:
            nextPos = curentPos.translate(direction, 1)
            if nextPos not in m and nextPos.x >= 0 and nextPos.x < width and nextPos.y >= 0 and nextPos.y < height and nextPos not in avoid:
                l.append(nextPos)
                m[nextPos] = (curentPos, direction)
    return None

def safe_to_build(player):
    for k, city in player.cities.items():
        safe_threshold = len(city.citytiles) * 200
        if city.fuel < safe_threshold:
            return False
    return True

def next_to_city(pos, player):
    for citytile in player.citytiles:
        if pos.is_adjacent(citytile.pos):
            return True
    return False