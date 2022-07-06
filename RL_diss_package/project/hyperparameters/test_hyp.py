# hyp from https://github.com/bhctsntrk/OpenAIPong-DQN/blob/master/OpenAIPong_DQN.ipynb
test_hyp = {
    "DO_CROP":True,
    "DO_RESCALE":True,
    "RESCALE_DIMS":(84,84),
    "EVALUATION":False,
    "CROP":(20,0,0,0),

    "BUFFER_SIZE":500_000,
    "BATCH_SIZE":32,
    # "EPs":2000,
    "ENV":"PongDeterministic-v4",
    "GAMMA":0.97,

    "REPLAY_FREQ":4,
    "UPDATE_TARGET":10_000,

    "EPS_FRAMES_INIT":50_000,
    "EPS_FRAMES_INTER":1_000_000,
    "EPS_FRAMES_FINAL":2_000_000,

    "EPS_STEPS_INIT":1,
    "EPS_STEPS_INTER":0.1,
    "EPS_STEPS_FINAL":0.01,
    # "CONV_LAYERS":5,
    # "DROPOUT":0.2,
    "INIT_LEARNING_RATE":0.00025,

    # "NUM_STEPS": 7500,
    "STACK_SIZE": 4,
    "MAX_EP_ACTIONS":20_000, # Need to adjust this
    
    "RENDER_EVERY_N": 100_000,  # Render gif and save model every N frames
    "UPDATE_EVERY_N": 10_000,  # Print update every N episodes
}


# test_hyp = {
#     "DO_CROP":False,
#     "DO_RESCALE":True,
#     "RESCALE_DIMS":(84,84),
#     "EVALUATION":False,
#     "CROP":(0,0,0,0),

#     "BUFFER_SIZE":500_000,
#     "BATCH_SIZE":32,
#     # "EPs":2000,
#     "ENV":"ALE/Pong",
#     "GAMMA":0.99,

#     "REPLAY_FREQ":4,
#     "UPDATE_TARGET":5_000,

#     "EPS_FRAMES_INIT":100_000,
#     "EPS_FRAMES_INTER":1_000_000,
#     "EPS_FRAMES_FINAL":5_000_000,

#     "EPS_STEPS_INIT":1,
#     "EPS_STEPS_INTER":0.1,
#     "EPS_STEPS_FINAL":0.01,
#     # "CONV_LAYERS":5,
#     # "DROPOUT":0.2,
#     "INIT_LEARNING_RATE":0.001,

#     # "NUM_STEPS": 7500,
#     "STACK_SIZE": 4,
#     "MAX_EP_ACTIONS":20_000, # Need to adjust this
    
#     "RENDER_EVERY_N": 100_000,  # Render gif and save model every N frames
#     "UPDATE_EVERY_N": 10_000,  # Print update every N episodes
# }