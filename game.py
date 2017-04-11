#Written by Ying Chan
#Code purpose: run a reinforcement learning network (Q-learning) with neural network to allow the agent to self play a game of Gomoku


import numpy as np

board_size = 5

grid = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0]

#Create left and right border indexes
def left_border(board_size):
    left_border = []
    for i in range(board_size):
        k = i*board_size
        left_border.append(k)
    return left_border

def right_border(board_size):
    right_border = []
    for i in range(board_size):
        k = i*board_size + board_size - 1
        right_border.append(k)
    return right_border


#function to create output of all pieces played by player id in that row_number
def row_pos(grid, row_number, color_id):
    result = []
    for i in range(board_size):
        j = board_size*(row_number-1)+i
        if grid[j] == color_id:
            result.append(j)
    return result


#function to create output of all pieces played by player id in that col_number
def col_pos(grid, col_number, color_id):
    result = []
    for i in range(board_size):
        j = board_size*(i)+col_number
        if grid[j-1] == color_id:
            result.append(j-1)
    return result


def diaga_pos(grid, start_id, color_id):
    result = []
    for j in range(board_size):
        indices = start_id + (board_size - 1) * j
        if grid[indices] == color_id and indices < board_size * board_size:
            result.append(indices)
    return result

def diagb_pos(grid, start_id, color_id):
    result = []
    for j in range(board_size):
        indices = start_id + (board_size + 1) * j
        if grid[indices] == color_id and indices < board_size * board_size:
            result.append(indices)
    return result



#function to find the maximum length of joint pieces in a given row or column based on diff.
#if diff is set to 1, then this searches a row
#if diff is set to board_size then this searches a column
def find_max_length(array, diff):
    j = 1
    k = 1
    beg_id = -1

    if len(array) == 0:
        k = 0

    for i in range(1,len(array)):
        if array[i]-array[i-1] == diff: #add counter to count how many consecutive items
            j += 1
            if j > k:
                k = j #store the last ones
                beg_id = array[i - j + 1]
        else:
            j=1 #reset counter
            beg_id = array[i-1]
    max_length = k
    end_id = beg_id + (k - 1)*diff

    if beg_id == -1:
        end_id = -1

    if len(array) ==1:
        max_length = 1
        beg_id = array[0]
        end_id = array[0]

    return [max_length, beg_id, end_id]

def find_adjacent_spaces(col_or_row, grid, board_size, beg_id, end_id):
    space_id = []
    if col_or_row == 'row':

        if beg_id == 0:
            space_id.append(-1)
        elif grid[beg_id - 1] == 0 and beg_id - 1 not in right_border(board_size):
            space_id.append(beg_id - 1)
        else:
            space_id.append(-1)

        if end_id == board_size*board_size-1:
            space_id.append(-1)
        elif grid[end_id + 1] == 0 and end_id + 1 not in left_border(board_size):
            space_id.append(end_id + 1)
        else:
            space_id.append(-1)

    if col_or_row == 'col':

        if beg_id == 0:
            space_id.append(-1)
        elif grid[beg_id - board_size] == 0 and beg_id - board_size >= 0:
            space_id.append(beg_id - board_size)
        else:
            space_id.append(-1)

        if end_id >= board_size*board_size-board_size:
            space_id.append(-1)
        elif grid[end_id + board_size] == 0 and end_id + board_size <= board_size*board_size-1:
            space_id.append(end_id + board_size)
        else:
            space_id.append(-1)

    if col_or_row == 'diaga':

        if beg_id == 0:
            space_id.append(-1)
        elif grid[beg_id - board_size + 1] == 0 and beg_id - board_size + 1 >= 0 and beg_id not in right_border(board_size):
            space_id.append(beg_id - board_size +1)
        else:
            space_id.append(-1)

        if end_id >= board_size*board_size - board_size :
            space_id.append(-1)
        elif grid[end_id + board_size - 1] == 0 and end_id not in left_border(board_size):
            space_id.append(end_id + board_size - 1)
        else:
            space_id.append(-1)

    if col_or_row == 'diagb':

        if beg_id == 0:
            space_id.append(-1)
        elif grid[beg_id - board_size - 1] == 0 and beg_id - board_size - 1 >= 0 and beg_id not in left_border(board_size):
            space_id.append(beg_id - board_size - 1)
        else:
            space_id.append(-1)

        if end_id >= board_size*board_size - board_size :
            space_id.append(-1)
        elif grid[end_id + board_size + 1] == 0 and end_id not in right_border(board_size):
            space_id.append(end_id + board_size + 1)
        else:
            space_id.append(-1)

    #exception handling
    if beg_id == -1:
        space_id[0] = -1
        space_id[1] = -1
    return space_id


def convert_to_row_status(grid, color_id):
    #Print row status
    import numpy as np
    arr = np.empty((0, 6), int)
    for i in range(1,board_size+1):
        max_length = find_max_length(row_pos(grid, i, color_id), 1)[0]
        beg_id = find_max_length(row_pos(grid, i, color_id), 1)[1]
        end_id = find_max_length(row_pos(grid, i, color_id), 1)[2]
        left_space = find_adjacent_spaces('row', grid, board_size, beg_id, end_id)[0]
        right_space = find_adjacent_spaces('row', grid, board_size, beg_id, end_id)[1]
        if left_space == -1 and right_space == -1 and max_length != 5: #update the max length to be zero for blocked pieces
            max_length = 0
        array = np.array([i, max_length, beg_id, end_id, left_space, right_space])
        arr = np.vstack((arr, array))
    return arr


def convert_to_col_status(grid, color_id):
    #Print column status
    import numpy as np
    arr = np.empty((0, 6), int)
    for i in range(1,board_size+1):
        max_length = find_max_length(col_pos(grid, i, color_id), board_size)[0]
        beg_id = find_max_length(col_pos(grid, i, color_id), board_size)[1]
        end_id = find_max_length(col_pos(grid, i, color_id), board_size)[2]
        left_space = find_adjacent_spaces('col', grid, board_size, beg_id, end_id) [0]
        right_space = find_adjacent_spaces('col', grid, board_size, beg_id, end_id)[1]
        if left_space == -1 and right_space == -1 and max_length != 5: #update the max length to be zero for blocked pieces
            max_length = 0
        array = np.array([i, max_length, beg_id, end_id, left_space, right_space])
        arr = np.vstack((arr, array))
    return arr

def convert_to_diaga_status(grid, color_id):
    #print diagonal a-type status
    import numpy as np
    arr = np.empty((0,6), int)
    for i in range(4, board_size):
        max_length = find_max_length(diaga_pos(grid, i, color_id), board_size)[0]
        beg_id = find_max_length(diaga_pos(grid, i, color_id), board_size)[1]
        end_id = find_max_length(diaga_pos(grid, i, color_id), board_size)[2]
        left_space = find_adjacent_spaces('diaga', grid, board_size, beg_id, end_id) [0]
        right_space = find_adjacent_spaces('diaga', grid, board_size, beg_id, end_id) [1]
        if left_space == -1 and right_space == -1 and max_length != 5: #update the max length to be zero for blocked pieces
            max_length = 0
        array = np.array([i, max_length, beg_id, end_id, left_space, right_space])
        arr = np.vstack((arr, array))

    for i in range(board_size*2 - 1, board_size*board_size - 4*board_size - 1):
        max_length = find_max_length(diaga_pos(grid, i, color_id), board_size)[0]
        beg_id = find_max_length(diaga_pos(grid, i, color_id), board_size)[1]
        end_id = find_max_length(diaga_pos(grid, i, color_id), board_size)[2]
        left_space = find_adjacent_spaces('diaga', grid, board_size, beg_id, end_id) [0]
        right_space = find_adjacent_spaces('diaga', grid, board_size, beg_id, end_id) [1]
        if left_space == -1 and right_space == -1 and max_length != 5: #update the max length to be zero for blocked pieces
            max_length = 0
        array = np.array([i, max_length, beg_id, end_id, left_space, right_space])
        arr = np.vstack((arr, array))
    return arr

def convert_to_diagb_status(grid, color_id):
    #print diagonal a-type status
    import numpy as np
    arr = np.empty((0,6), int)
    for i in range(0, board_size - 5 + 1):
        max_length = find_max_length(diagb_pos(grid, i, color_id), board_size)[0]
        beg_id = find_max_length(diagb_pos(grid, i, color_id), board_size)[1]
        end_id = find_max_length(diagb_pos(grid, i, color_id), board_size)[2]
        left_space = find_adjacent_spaces('diagb', grid, board_size, beg_id, end_id) [0]
        right_space = find_adjacent_spaces('diagb', grid, board_size, beg_id, end_id) [1]
        if left_space == -1 and right_space == -1 and max_length != 5: #update the max length to be zero for blocked pieces
            max_length = 0
        array = np.array([i, max_length, beg_id, end_id, left_space, right_space])
        arr = np.vstack((arr, array))

    for i in range(board_size, board_size*board_size - board_size*(5-1)):
        max_length = find_max_length(diagb_pos(grid, i, color_id), board_size)[0]
        beg_id = find_max_length(diagb_pos(grid, i, color_id), board_size)[1]
        end_id = find_max_length(diagb_pos(grid, i, color_id), board_size)[2]
        left_space = find_adjacent_spaces('diagb', grid, board_size, beg_id, end_id) [0]
        right_space = find_adjacent_spaces('diagb', grid, board_size, beg_id, end_id) [1]
        if left_space == -1 and right_space == -1 and max_length != 5: #update the max length to be zero for blocked pieces
            max_length = 0
        array = np.array([i, max_length, beg_id, end_id, left_space, right_space])
        arr = np.vstack((arr, array))
    return arr


def get_state(grid, color_id):
    array_row = convert_to_row_status(grid,color_id)
    array_col = convert_to_col_status(grid,color_id)
    array_diaga = convert_to_diaga_status(grid,color_id)
    array_diagb = convert_to_diagb_status(grid,color_id)
    final_array = np.vstack((array_row, array_col, array_diaga, array_diagb))
    return final_array


def get_full_state(grid, color_id):
    if color_id == 1:
        array_row_first = convert_to_row_status(grid,1)
        array_col_first = convert_to_col_status(grid,1)
        array_diaga_first = convert_to_diaga_status(grid,1)
        array_diagb_first = convert_to_diagb_status(grid,1)
        array_row_second = convert_to_row_status(grid,-1)
        array_col_second = convert_to_col_status(grid,-1)
        array_diaga_second = convert_to_diaga_status(grid, -1)
        array_diagb_second = convert_to_diagb_status(grid, -1)
    else:
        array_row_first = convert_to_row_status(grid, -1)
        array_col_first = convert_to_col_status(grid, -1)
        array_diaga_first = convert_to_diaga_status(grid, -1)
        array_diagb_first = convert_to_diagb_status(grid, -1)
        array_row_second = convert_to_row_status(grid, 1)
        array_col_second = convert_to_col_status(grid, 1)
        array_diaga_second = convert_to_diaga_status(grid, 1)
        array_diagb_second = convert_to_diagb_status(grid, 1)
    final_array = np.vstack((array_row_first, array_col_first, array_diaga_first, array_diagb_first,
                             array_row_second, array_col_second, array_diaga_second, array_diagb_second))
    return final_array

def getReward(grid, color_id):
    reward = 0
    if color_id == 1:
        my_max_length = get_state(grid, 1)[:,1]
        other_max_length = get_state(grid, -1)[:,1]
    else:
        my_max_length = get_state(grid, -1)[:, 1]
        other_max_length = get_state(grid, 1)[:, 1]
    count = len(my_max_length)
    for i in range(0,count):
        if my_max_length[i] == 5:
            reward = 1000
            #win game
    for i in range(0,count):
        if other_max_length[i] == 5:
            reward = -1000
            #lose game

    if grid.count(0) == 0:
        reward = 10
        #tie
    return reward


def make_action(grid, action, color_id):
    import random
    #copy grid into a new grid
    new_grid = []
    for i in range(0, len(grid)):
        new_grid.append(grid[i])


    if color_id == 1:
        my_state = get_state(grid, 1)
        other_state = get_state(grid, -1)
    else:
        my_state = get_state(grid, -1)
        other_state = get_state(grid, 1)

    if action == 0: #attack action
        max_col = my_state[:,1]
        id = np.argmax(max_col, axis=0)
        if np.max(max_col, axis=0) != 0:
            avail_left = my_state[id, 4]
            avail_right = my_state[id, 5]
            avail = []
            if avail_left != -1:
                avail.append(avail_left)
            if avail_right != -1:
                avail.append(avail_right)
            print avail
            place_stone = random.choice(avail)
            new_grid[place_stone] = color_id
        else:
            list_of_zeros = [i for i, x in enumerate(grid) if x == 0]
            random_stone = random.choice(list_of_zeros)
            new_grid[random_stone] = color_id

    if action == 1: #defend action
        max_col = other_state[:,1]
        id = np.argmax(max_col, axis=0)
        if np.max(max_col, axis=0) != 0:
            avail_left = other_state[id, 4]
            avail_right = other_state[id, 5]
            avail = []
            if avail_left != -1:
                avail.append(avail_left)
            if avail_right != -1:
                avail.append(avail_right)
            print avail
            place_stone = random.choice(avail)
            new_grid[place_stone] = color_id
        else:
            list_of_zeros = [i for i, x in enumerate(grid) if x == 0]
            random_stone = random.choice(list_of_zeros)
            new_grid[random_stone] = color_id

    return new_grid

#run machine learning/q learning algorithm
#create deep learning neural network model layer to obtain the Q function


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(6*board_size*4 + 6*(board_size-4)*4,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(2, init='lecun_uniform')) #limit to two outputs, one is attack and the other will be defend
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

#Start machine learning model below

epochs = 1000
gamma = 0.9  #can play around with this variable
epsilon = 0.3  #can play around with this variable

import random


for i in range(epochs):
    grid = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]

    j = 0 #odd and even counter for players 1 and -1
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        # the state is set as from the perspective of player 1

        #Loop across player 1 and player -1
        if (j % 2 == 0):  # even
            color_id = 1
        else:
            color_id = -1

        state = get_full_state(grid, color_id) #get the state with the perspective of this color id

        qval = model.predict(state.reshape(1, 6*board_size*4 + 6*(board_size-4)*4), batch_size=1)
        if random.random() < epsilon:  # choose random action
            action = np.random.randint(0, 2)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #take the action
        #need to modify the color id to dynamically switch players
        new_grid = make_action(grid, action, color_id) #MODIFY
        new_state = get_full_state(new_grid, color_id) #MODIFY

        #check the reward for this state
        reward = getReward(new_grid, color_id) #MODIFY
        # Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1, 6*board_size*4 + 6*(board_size-4)*4), batch_size=1)
        maxQ = np.max(newQ)


        #Update function for the y value
        y = np.zeros((1, 2))
        y[:] = qval[:]

        if reward == 0:  # non-terminal state
            update = (reward + (gamma * maxQ))

        else:  # terminal state
            update = reward
        y[0][action] = update
        print("Game #: %s" % (i,))

        #Fit the model using the updated y value for the perspective of player 1
        model.fit(state.reshape(1, 6*board_size*4 + 6*(board_size-4)*4), y, batch_size=1, nb_epoch=1, verbose=1)

        #Get the new state and grd and print the grid
        state = new_state
        grid = new_grid
        print grid

        #iterate against different players
        j += 1
        print reward
        #End game if rewarded on this move
        if reward != 0:
            status = 0
    if epsilon > 0.1:
        epsilon -= (1 / epochs)

def predict_next_move(prev_grid, color_id):
    state = get_full_state(prev_grid, color_id)
    pred_q = model.predict(state.reshape(1, 6*board_size*4 + 6*(board_size-4)*4), batch_size=1)

    action_to_make = np.argmax(pred_q)
    new_grid = make_action(prev_grid, action_to_make, color_id)

    return new_grid

new_grid = predict_next_move([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 1, 1, 1, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0], 1)

print new_grid