
test_hyp = {
    "BUFFER_SIZE":1000,
    "BATCH_SIZE":10,
    # "EPs":2000,

    "GAMMA":0.95,

    "REPLAY_FREQ":4,
    "UPDATE_TARGET":1000,

    "EPS_FRAMES_INIT":50000,
    "EPS_FRAMES_INTER":500000,
    "EPS_FRAMES_FINAL":1000000,

    "EPS_STEPS_INIT":1,
    "EPS_STEPS_INTER":0.1,
    "EPS_STEPS_FINAL":0.01,
    # "CONV_LAYERS":5,
    # "DROPOUT":0.2,
    "INIT_LEARNING_RATE":0.001,

    # "NUM_STEPS": 7500,
    "STACK_SIZE": 4,
    "MAX_EP_ACTIONS":20_000, # Need to adjust this
    
    "RENDER_EVERY_N": 1000,  # Render gif and save model every N frames
    "UPDATE_EVERY_N": 1000,  # Print update every N episodes
}