
test_hyp = {
    "DO_CROP":False,
    "DO_RESCALE":True,
    "RESCALE_DIMS":(84,84),
    "EVALUATION":False,
    "CROP":(0,0,0,0),

    "BUFFER_SIZE":1000,
    "BATCH_SIZE":10,
    # "EPs":2000,
    "ENV":"ALE/Pong",
    "GAMMA":0.95,

    "REPLAY_FREQ":4,
    "UPDATE_TARGET":1_000,

    "EPS_FRAMES_INIT":250_000,
    "EPS_FRAMES_INTER":1_000_000,
    "EPS_FRAMES_FINAL":5_000_000,

    "EPS_STEPS_INIT":1,
    "EPS_STEPS_INTER":0.1,
    "EPS_STEPS_FINAL":0.01,
    # "CONV_LAYERS":5,
    # "DROPOUT":0.2,
    "INIT_LEARNING_RATE":0.001,

    # "NUM_STEPS": 7500,
    "STACK_SIZE": 4,
    "MAX_EP_ACTIONS":20_000, # Need to adjust this
    
    "RENDER_EVERY_N": 100_000,  # Render gif and save model every N frames
    "UPDATE_EVERY_N": 10000,  # Print update every N episodes
}