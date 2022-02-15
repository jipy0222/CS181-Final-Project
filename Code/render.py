from kaggle_environments import make

env = make("lux_ai_2021",
                configuration={
                    "seed": 562124215,
                    "loglevel": 1,
                    "annotations": True
                },
                debug=True)
a_run = env.run(['rl_agent.py', 'rl_agent.py'])
out = env.render(mode="html", width=1200, height=800)
with open('kaggle.html', 'w') as f:
    f.write(out)