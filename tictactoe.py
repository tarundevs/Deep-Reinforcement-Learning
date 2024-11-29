# Import necessary modules
import numpy as np
import random,os,warnings,matplotlib,time
import matplotlib.pyplot as plt
# Use Agg backend to prevent Tkinter issues with 
matplotlib.use('Agg')  
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
from collections import deque,Counter

from concurrent.futures import ProcessPoolExecutor, as_completed
class Environment:
    def __init__(self):
        '''
            This class sets up the gaming environment to simulate a game of tic-tac-toe. It implements both training and testing functions.
        '''
        self.players=[-1,1]
        self.player=random.choice(self.players)
        self.results_train={"X":{"win": 0, "loss": 0, "draw": 0,"invalid move":0},
                      "O":{"win": 0, "loss": 0, "draw": 0,"invalid move":0}}
        self.results_test={"X":{"win": 0, "loss": 0, "draw": 0,"invalid move":0},
                      "O":{"win": 0, "loss": 0, "draw": 0,"invalid move":0}}
        self.smartness=0
    
    def is_valid_move(self, position,board):
        return board[position] == 0
    
    def empty_positions(self,board):
        return [i for i in range(9) if board[i] == 0]
    
    def is_full(self,board):
        return all(x != 0 for x in board)
        
    def print_board(self,board):
        state=[i if i!=0 else " " for i in board.copy()]
        
        for i in range(3):
            print(f"|{state[3*i]}|{state[3*i+1]}|{state[3*i+2]}|")
            print("__ __ __")
    
    def check_winner(self,player,board):
        '''
            Function to check if -1 or 1 is the winner
        '''
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        
        for condition in win_conditions:
            if all(board[i] == player for i in condition):
                return True
        return False

class Run_Game:
    def __init__(self):
        '''
            Class to handle running the game
        '''
        self.train_results ={
                "X": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
                "O": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
            }
        
        self.test_results ={
                "X": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
                "O": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
            }
        
    def run_test(self, agent, env, q_learning, X=True):
        '''
            Function to run an episode in a multiprocessing-safe manner.
        '''
        agent.reset()
        test_results={
                "X": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
                "O": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
            }
        
        while not env.is_full(agent.board) and agent.winner is None and not agent.invalid_move[0]:
            if X:
                agent.player_X(env, q_learning, hard_coded=True)
                X = not X
            else:
                agent.player_O(env, q_learning, hard_coded=False)
                X = not X
            
            # Update results based on game outcome
            if agent.winner == "X":
                test_results["X"]["win"] += 1
                test_results["O"]["loss"] += 1
            elif agent.winner == "O":
                test_results["O"]["win"] += 1
                test_results["X"]["loss"] += 1
            elif env.is_full(agent.board):
                test_results["X"]["draw"] += 1
                test_results["O"]["draw"] += 1
            elif agent.invalid_move[0]:
                test_results["X"]["invalid move"] += 1
                test_results["O"]["invalid move"] += 1
            
        return test_results

    def test(self, games=100):
        """
        Function to test Q-learning with multiprocessing and progress tracking.
        """
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.run_test, agent, env, q_learning): e for e in range(games)}

            # Wrap as_completed with tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Testing episodes",leave=False):
                test_results = future.result()
                for player, stats in test_results.items():
                    for key, value in stats.items():
                        self.test_results[player][key] += value

        # Convert results back to a normal dictionary
        final_results = self.test_results
        print("Test:", final_results)

    def run_episode(self, agent, env, q_learning, X=True):
        '''
            Function to run an episode in a multiprocessing-safe manner.
        '''
        agent.reset()
        train_results={
                "X": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
                "O": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
            }
        
        while not env.is_full(agent.board) and agent.winner is None and not agent.invalid_move[0]:
            if X:
                agent.player_X(env, q_learning, hard_coded=True)
                X = not X
            else:
                agent.player_O(env, q_learning, hard_coded=False)
                X = not X

            # Update results based on game outcome
            if agent.winner == "X":
                train_results["X"]["win"] += 1
                train_results["O"]["loss"] += 1
            elif agent.winner == "O":
                train_results["O"]["win"] += 1
                train_results["X"]["loss"] += 1
            elif env.is_full(agent.board):
                train_results["X"]["draw"] += 1
                train_results["O"]["draw"] += 1
            elif agent.invalid_move[0]:
                train_results["X"]["invalid move"] += 1
                train_results["O"]["invalid move"] += 1
        
        # Build representation for memory
        done = [False] * (len(agent.action) - 2) + [True, True]
        reward = ([0] * (len(agent.action) - 2) + [-1, 1]) if not env.is_full(agent.board) else [0] * len(agent.action)
        agent.memory.append(agent.memory[-1])
        representation = [(agent.memory[i], agent.action[i], reward[i], agent.memory[i + 2], done[i]) for i in range(len(agent.action))]
        
        return train_results,representation

    def run(self, batches=10, episodes_per_batch=100):
        '''
            Train using Q-learning with multiprocessing support.
        '''
        
        with ProcessPoolExecutor() as executor:
            for _ in range(batches):
                futures = {executor.submit(self.run_episode, agent, env, q_learning): e for e in range(episodes_per_batch)}

                # Wrap as_completed with tqdm for progress tracking
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing episodes",leave=False):
                    train_results,representation = future.result()
                    
                    for player, stats in train_results.items():
                        for key, value in stats.items():
                            self.train_results[player][key] += value

                    # Extend the agent's buffer with the representations from all episodes
                    for rep in representation:
                        agent.buffer.append(rep)

                # Train the model
                q_learning.update()
                
                # Test the model
                self.test()
            
                print("Train Results:", self.train_results)
                print("-"*150)
                print()
                
                # Reset results
                self.train_results={
                "X": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
                "O": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
            }
                self.test_results={
                "X": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
                "O": {"win": 0, "loss": 0, "draw": 0, "invalid move": 0},
            }
        
class Agent:
    def __init__(self):
        '''
            This class sets up the agent to maximise number of wins
        '''
        self.state=[0]*9
        self.next_state=[0]*9
        self.action=[]
        self.invalid_move=[False,"X"]
        self.buffer=deque(maxlen=30_000)
        self.memory=[[0]*9]
        self.board=[0]*9
        self.winner=None
   
    def move(self, env, position, player):
        if env.is_valid_move(position,self.board):
            self.board[position] = player
  
    def reset(self):
        self.state=[0]*9
        self.next_state=[0]*9
        self.action=[]
        self.invalid_move=[False,"X"]
        self.memory=[[0]*9]
        self.board=[0]*9
        self.winner=None
        
    def hard_coded_player(self, env, player):
        '''
            Function to implement a hardcoded player with varying smartness from 0 to 1.
            0=Random Plays
            1=Win/Block moves
        '''
        if random.random() < env.smartness:
            position = self.hard_player_action(env, player)
            if position is None:
                position = random.choice(env.empty_positions(self.board))
        else:
            position = random.choice(env.empty_positions(self.board))
        return position
    
    def hard_player_action(self, env, player):
        opponent = "O" if player == "X" else "X"
        
        for position in env.empty_positions():
            self.board[position] = player
            if env.check_winner(player, self.board):
                self.board[position] = 0
                return position
            self.board[position] = 0

        for position in env.empty_positions():
            self.board[position] = opponent
            if env.check_winner(opponent, self.board):
                self.board[position] = 0
                return position
            self.board[position] = 0

        return None
    
    def random_player(self):
        '''
            Function to simulate a random player
        '''
        valid_actions = [i for i, spot in enumerate(self.board) if spot == 0]
        action=random.choice(valid_actions)
        return action
    
    def player_X(self, env, q_learning, hard_coded=True):
        '''
            Function to simulate the X player
        '''
        if hard_coded:
            action=self.hard_coded_player(env, "X")
        else:
            action=q_learning.predict(self.board.copy())
            
        if not(env.is_valid_move(action,self.board)):
            self.invalid_move=[True,"X"]
        
        self.move(env, action ,"X")
        
        self.action.append(action)
        
        if env.check_winner("X",self.board):
            self.winner="X"   
        else:
            self.winner=None
                    
        self.memory.append(self.board.copy())
          
    def player_O(self, env, q_learning, hard_coded=True):
        '''
            Function to simulate the O player
        '''
        if hard_coded:
            action=self.hard_coded_player(env, "O")
        else:
            action=q_learning.predict(self.board.copy())
            
        if not(env.is_valid_move(action,self.board)):
            self.invalid_move=[True,"O"]
        
        self.move(env, action, "O")
        
        self.action.append(action)
        
        if env.check_winner("O",self.board):
            self.winner="O"   
        else:
            self.winner=None
                    
        self.memory.append(self.board.copy())
   
class Reduce_Retracing:
    '''
        Class to reduce retraining
    '''

    def predict_fn(self,model,state):
        return model(state, training=False)[0]

    @tf.function(reduce_retracing=True)
    def train_on_batch(self, model, states, q_values):
        # Perform the training step
        return model.fit(states, q_values, epochs=1, verbose=0).history['loss'][0]

    @tf.function(reduce_retracing=True)
    def get_predictions(self, model, state):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        return model(state_tensor, training=False)

class Q_Learning:
    def __init__(self,model_path,retracing):
        '''
            This class sets up the Q learning interface to implement tabular updates.
        '''
        self.model_path=model_path
        self.learning_rate=0.005
        self.retracing=retracing
        if os.path.exists(self.model_path):
            self.model=self.load_model(self.model_path)
        else:
            self.model=self.build_model()
        self.gamma=0.99
        
    def load_model(self,model_path):
        return load_model(model_path)
        
    def build_model(self):
        '''
            This function builds a model
        '''
        model = Sequential([
            Dense(64, input_dim=9, activation='relu'),
            Dense(64, activation='relu'),
            Dense(9, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def predict(self,state,epsilon=1):
        '''
            Function to predict Q values
        '''
        if random.random()>=epsilon:
            action=agent.random_player()
            return action
        
        state=self.change_state(state)
        valid_actions = [i for i, spot in enumerate(state[0]) if spot == 0]
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        q_values = self.retracing.predict_fn(self.model,state)
        q_valid = [q_values[i] for i in valid_actions]
        action=valid_actions[np.argmax(q_valid)]
        return action
        
    def change_state(self,state):
        map={"X":1,"O":-1,0:0}
        new_state=[map[i] for i in state]
        return np.array(new_state).reshape(1,9)
    
    def update(self,batch_size=32):
        num_batches = len(agent.buffer) // batch_size
        
        # Initialize tqdm progress bar
        with tqdm(total=num_batches, desc="Replay Batches",leave=False) as pbar:
            for _ in range(num_batches):
                
                mini_batch = random.sample(agent.buffer, batch_size)
                states = np.zeros((batch_size, 9))
                q_values = np.zeros((batch_size, 9))
                
                for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
                    # Modify state if player turn is -1
                    state = self.change_state(state)
                    next_state = self.change_state(next_state)

                    # Use the get_predictions function to avoid retracing
                    current_q_values =self.retracing. get_predictions(self.model, tf.convert_to_tensor(state, dtype=tf.float32))[0]
                    next_q_values = self.retracing.get_predictions(self.model, tf.convert_to_tensor(next_state, dtype=tf.float32))[0]
                    # Soft Q-value update
                    if done:
                        q_target = reward
                    else:
                        q_target = reward + self.gamma * np.max(next_q_values)

                    current_q_values = current_q_values.numpy()  # Convert to a NumPy array
                    current_q_values[action] = q_target  # Perform the assignment
                    current_q_values = tf.convert_to_tensor(current_q_values, dtype=tf.float32)  # Convert back to a tensor

                    states[i] = state
                    q_values[i] = current_q_values


                # Train the model in a single step using the train_on_batch helper
                loss = self.retracing.train_on_batch(self.model, tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(q_values, dtype=tf.float32))

                pbar.update(1)  # Update progress bar after each batch

if __name__=='__main__':
    # Creating instances of classes
    env=Environment()
    agent=Agent()
    run=Run_Game()
    retracing=Reduce_Retracing()
    q_learning=Q_Learning(r"",retracing=retracing)
    
    # Run the program
    modes=['training','testing']
    mode=modes[1]
    
    if mode==modes[0]:
        # Run the training function
        run.run()
    else:
        # Run the testing function
        run.test()
