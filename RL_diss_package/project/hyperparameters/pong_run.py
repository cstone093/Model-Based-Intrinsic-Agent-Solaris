# hyp from https://github.com/bhctsntrk/OpenAIPong-DQN/blob/master/OpenAIPong_DQN.ipynb
pong = {
    "DO_CROP":True,
    "DO_RESCALE":True,
    "RESCALE_DIMS":(84,84),
    "EVALUATION":False,
    "CROP":(32,15,0,0),
    "SEED":0,

    "BUFFER_SIZE":10_000,
    "BATCH_SIZE":32,
    # "EPs":2000,
    "ENV":"PongDeterministic-v4",
    "GAMMA":0.99,

    "REPLAY_FREQ":4,
    "UPDATE_TARGET":1_000,

    "EPS_FRAMES_INIT":10_000,
    "EPS_FRAMES_INTER":200_000,
    "EPS_FRAMES_FINAL":1_000_000,

    "EPS_STEPS_INIT":1,
    "EPS_STEPS_INTER":0.1,
    "EPS_STEPS_FINAL":0.01,
    # "CONV_LAYERS":5,
    # "DROPOUT":0.2,
    "INIT_LEARNING_RATE":0.0001,

    # "NUM_STEPS": 7500,
    "STACK_SIZE": 4,
    "MAX_EP_ACTIONS":20_000, # Need to adjust this
    
    "RENDER_EVERY_N": 50,  # Render gif and save model every N episodes
    "UPDATE_EVERY_N": 5,  # Print update every N episodes
    "SAVE_EVERY_N": 50, # Save every N episodes
}
