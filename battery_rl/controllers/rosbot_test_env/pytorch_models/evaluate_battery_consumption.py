import numpy as np

with open('vanilla.log', 'r') as fin:
    result_vanilla = fin.readlines()

with open('battery.log', 'r') as fin:
    result_battery = fin.readlines()

with open('real2sim.log', 'r') as fin:
    result_real2sim = fin.readlines()

len_episode_vanilla = eval(result_vanilla[0].replace('episode length:', '').strip())
battery_penalty_vanilla = eval(result_vanilla[1].replace('battery penalty:', '').strip())
len_episode_battery = eval(result_battery[0].replace('episode length:', '').strip())
battery_penalty_battery = eval(result_battery[1].replace('battery penalty:', '').strip())
len_episode_real2sim = eval(result_real2sim[0].replace('episode length:', '').strip())
battery_penalty_real2sim = eval(result_real2sim[1].replace('battery penalty:', '').strip())

print('='*20)
print(f'Mean episode length')
print(f'Baseline: {np.mean(len_episode_vanilla)}')
print(f'Battery: {np.mean(len_episode_battery)}')
print(f'{np.round((np.mean(len_episode_vanilla) - np.mean(len_episode_battery)) / np.mean(len_episode_vanilla)*100, 2)}% reduced (improvement)')
print(f'Real2sim: {np.mean(len_episode_real2sim)}')
print(f'{np.round((np.mean(len_episode_vanilla) - np.mean(len_episode_real2sim)) / np.mean(len_episode_vanilla)*100, 2)}% reduced (improvement)')
print()

print(f'Mean battery consumption')
print(f'Baseline: {np.mean(battery_penalty_vanilla)}')
print(f'Battery: {np.mean(battery_penalty_battery)}')
print(f'{np.round((np.mean(battery_penalty_vanilla) - np.mean(battery_penalty_battery)) / np.mean(battery_penalty_vanilla)*100, 2)}% reduced (improvement)')
print(f'Real2sim: {np.mean(battery_penalty_real2sim)}')
print(f'{np.round((np.mean(battery_penalty_vanilla) - np.mean(battery_penalty_real2sim)) / np.mean(battery_penalty_vanilla)*100, 2)}% reduced (improvement)')
print('='*20)