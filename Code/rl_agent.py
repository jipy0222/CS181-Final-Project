TRAIN = False

from tqdm import tqdm, trange
import random

from kaggle_environments import make
from lux.game import Game

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

class LUXAI(nn.Module):
    def __init__(self,
                 input_channel=16,
                 out_channels=32,
                 kernel_size=3,
                 hidden_layers=10,
                 worker_channel=7,
                 ct_channel=2,
                 kt_channel=2,
                 ):
        super().__init__()
        self.out_channels = out_channels
        self.worker_channel = worker_channel
        self.ct_channel = ct_channel
        self.kt_channel = kt_channel

        self.conv2ds = nn.ModuleList(
            [nn.Conv2d(input_channel, out_channels, kernel_size)] + 
            [nn.Conv2d(out_channels, out_channels, kernel_size) for i in range(hidden_layers - 1)]
            )
        self.worker_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.worker_channel))

        self.ct_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.ct_channel))

    def forward(self, x):
        n = x.shape[0]
        for conv2d in self.conv2ds:
            x = F.max_pool2d(F.relu(conv2d(x)), kernel_size = 2, stride = 1)
        x = x.reshape(n, -1)
        worker_score = self.worker_layer(x).reshape(n, -1)
        ct_score = self.ct_layer(x).reshape(n, -1)

        return worker_score, ct_score

game_state = None
net = LUXAI().cuda()
#net.load_state_dict(torch.load('net_state_dict_00350000.pth'))
optimizer = Adam(net.parameters(), lr = 0.0005)

def extract_feature_map(state):
    features = torch.zeros(15, 32, 32).cuda()

    for entity in (state['updates']):
        strs = entity.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'r':
            r_type = strs[1]
            x = int(strs[2])
            y = int(strs[3])
            amt = int(float(strs[4]))
            features[0, x, y] = 5
            features[1, x, y] = amt / 500
            if r_type == 'wood': 
                for tempx in range(max(x-2,0),min(x+2,31)):
                    for tempy in range(max(y-2,0),min(y+2,31)):
                        features[11, tempx, tempy] += 1

        elif input_identifier == 'u':
            team = -1 
            if int(strs[2]) != 0: team = 1
            x = int(strs[4])
            y = int(strs[5])
            cooldown = int(strs[6])
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            features[2, x, y] = team
            features[3, x, y] = max(wood*20 + coal*50 + uranium*80, 1) / 1000 * team
            features[4, x, y] = cooldown * team
            for tempx in range(max(x-2,0),min(x+2,31)):
                for tempy in range(max(y-2,0),min(y+2,31)):
                    if int(strs[1]) == 0:
                        features[13,tempx,tempy] += 1
                    else:
                        features[14,tempx,tempy] += 1
                    

        elif input_identifier == 'ct':
            team = -1 
            if int(strs[1]) != 0: team = 1
            x = int(strs[3])
            y = int(strs[4])
            cooldown = int(strs[5])
            features[5, x, y] = 1 * team
            features[6, x, y] = cooldown * team
            for tempx in range(max(x-2,0),min(x+2,31)):
                    for tempy in range(max(y-2,0),min(y+2,31)):
                        features[12, tempx, tempy] += 1
            
        elif input_identifier == 'c':
            team = 1 if int(strs[1]) == 0 else -1
            fuel = int(strs[3])
            lightkeepup = int(strs[4])
            features[7, :, :] = fuel / 100
            features[8, :, :] = lightkeepup / 100

        elif input_identifier == 'ccd':
            x = int(strs[1])
            y = int(strs[2])
            level = int(strs[3])
            features[9, x, y] = level
            
    features[10, :, :] = state['step'] % 40

    return features


def add_location(features, x):
    location_layer = torch.zeros(1, 32, 32).cuda()
    if x != None:
        pos = x.pos
        location_layer[0, pos.x, pos.y]
    features = torch.cat((location_layer, features))
    inputs = features[None, ...]
    return inputs

def get_reward(state, next_state):
    player = game_state.players[state.player]
    def total_wood(state):
        return sum([unit.cargo.wood for unit in player.units])
    def total_coal(state):
        return sum([unit.cargo.coal for unit in player.units])
    def total_uranium(state):
        return sum([unit.cargo.uranium for unit in player.units])
    def total_fuel(state):
        return sum([city.fuel for city in player.cities.values()])
    def total_citytiles(state):
        return len([citytile for city in list(player.cities.values()) for citytile in city.citytiles])
    def total_workers(state):
        return len(player.units)
    def research_points(state):
        return min(player.research_points, 200)
    def total_score(state):
        score = total_wood(state) * 1 + \
                total_coal(state) * 4 + \
                total_uranium(state) * 10 + \
                total_fuel(state) * 0.5 + \
                total_citytiles(state) * 1000 + \
                total_workers(state) * 100 + \
                research_points(state) * 2
        return score
    
    return total_score(next_state) - total_score(state)

def compute_loss(batch):
    factor = 0.9
    features = []
    actions = []
    rewards = []
    next_features = []
    mask = []
    for feature, category, action, reward, next_feature in batch:
        features.append(feature)
        actions.append(action)
        rewards.append(reward)
        next_features.append(next_feature)
        mask.append(category == 'u')
    features, actions, rewards, next_features, mask = torch.stack(features), torch.stack(actions), torch.tensor(reward).cuda(), torch.stack(next_features), torch.tensor(mask, dtype = torch.bool).cuda()
    worker_score, ct_score = net(features)
    Q_value = torch.cat((worker_score[mask][range(len(worker_score[mask])), actions[mask]], ct_score[~mask][range(len(ct_score[~mask])), actions[~mask]]))
    with torch.no_grad():
        worker_score, ct_score = net(next_features)
    Q_value_ = reward + factor * torch.cat((torch.max(worker_score[mask], dim = -1)[0], torch.max(ct_score[~mask], dim = -1)[0]))
    return F.mse_loss(Q_value, Q_value_)


unit_action = ['n', 'w', 's', 'e', 'c']
take_actions = {'u':lambda x, a: x.move(unit_action[a]) if a < 5 else x.build_city() if a == 5 else x.pillage(),
           'ct': lambda x, a: x.research() if a == 0 else x.build_worker()}

def my_agent(state):
    global game_state
    global net

    ### Do not edit ###
    if state["step"] == 0:
        game_state = Game()
        game_state._initialize(state["updates"])
        game_state._update(state["updates"][2:])
        game_state.id = state.player
    else:
        game_state._update(state["updates"])
    player = game_state.players[state.player]

    features = extract_feature_map(state)

    units = [('u', unit) for unit in player.units]
    citytiles = [('ct', citytile) for city in list(player.cities.values()) for citytile in city.citytiles]

    actions = {}
    for category, entity in units + citytiles:
        board_inputs = add_location(features, entity).cuda()
        with torch.no_grad():
            worker_score, ct_score = net(board_inputs)
            w_action, ct_action = torch.argmax(worker_score.squeeze()), torch.argmax(ct_score.squeeze())
            action = w_action if category == 'u' else ct_action
            if random.random() > 0.6 and TRAIN:
                action_ = random.randint(0, 6)  if category == 'u' else random.randint(0, 1)
            action_ = take_actions[category](entity, action)
            actions[entity] = (category, action_, action)

    return actions

if __name__ == '__main__':
    TRAIN = True
    buff = []
    steps = 1
    for games in trange(1, 30000):

        env = make("lux_ai_2021",
                    configuration={
                        "seed": 562124215,
                        "loglevel": 1,
                        "annotations": True
                    },
                    debug=True)

        trainer = env.train([None, 'simple_agent'])
        next_state, gameover = trainer.reset(), False

        while not gameover:
            state = next_state
            actions_ = my_agent(state)
            actions = [i[1] for i in actions_.values()]
            next_state, reward, gameover, _ = trainer.step(actions)
            reward = get_reward(state, next_state)
            player = game_state.players[state.player]
            
            for entity, action_ in actions_.items():
                category, action_, action = action_
                feature = add_location(extract_feature_map(state), entity)
                next_entity = None
                if category == 'u':
                    for unit in player.units:
                        if unit.id == entity.id:
                            next_entity = unit
                else:
                    for city in player.cities.values():
                        for citytile in city.citytiles:
                            if citytile.cityid == entity.cityid and citytile.pos == entity.pos:
                                next_entity = unit
                next_feature = add_location(extract_feature_map(state), entity)
                buff.append((feature.squeeze(), category, action, reward, next_feature.squeeze()))
            while len(buff) > 32768:
                buff.pop()

            if steps % 40 == 0:
                batch = random.sample(buff, min(2048, len(buff)))
                
                loss = compute_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if steps % 50000 == 0:
                    torch.save(net.state_dict(), 'net_state_dict_{:0>8d}.pth'.format(steps))
            steps += 1
