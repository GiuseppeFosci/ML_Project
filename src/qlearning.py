import tkinter as tk
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import threading
from PIL import Image, ImageTk  

stop_flag = False

def run(is_training=True, render=False, learning_rate_a=0.1, discount_factor_g=0.99, epsilon=1, epsilon_decay_rate=0.00001, output_widget=None, button_run=None, canvas=None):

    global stop_flag
    
    env = gym.make('CartPole-v1', render_mode='rgb_array' if render else None)

    
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))  # Init Q-table
    else:
        f = open('cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.00001  
    rng = np.random.default_rng() 

    rewards_per_episode = []

    i = 0

    while True:
        state = env.reset()[0]  
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False  
        rewards = 0

        while not terminated and rewards < 10000:

            if stop_flag:
                print("Stopping game due to exit.")
                env.close()
                if button_run: 
                    button_run.config(state=tk.NORMAL)
                return  

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

           #Render environment in canvas
            if render:
                frame = env.render()  
                if frame is not None and canvas:
                    
                    img = Image.fromarray(frame)
                    img = img.resize((300, 300))  
                    img_tk = ImageTk.PhotoImage(img)

                   
                    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    canvas.image = img_tk  

            if not is_training and rewards % 100 == 0:
                if output_widget:
                    output_widget.insert(tk.END, f'Episode: {i}  Rewards: {rewards}\n')
                    output_widget.see(tk.END) 

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if is_training and i % 100 == 0:
            if output_widget:
                output_widget.insert(tk.END, f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}\n')
                output_widget.see(tk.END)  

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1

    env.close()  

    
    if is_training:
        f = open('cartpole.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')

    if button_run:  
        button_run.config(state=tk.NORMAL)


def start_program():
    global stop_flag
    stop_flag = False  

    learning_rate = float(entry_learning_rate.get())
    discount_factor = float(entry_discount_factor.get())
    epsilon_value = float(entry_epsilon.get())
    epsilon_decay = float(entry_epsilon_decay.get())
    
    is_training = var_training.get() == 1
    render = var_render.get() == 1

    params_text = f"Starting simulation with parameters:\nLearning Rate: {learning_rate}\nDiscount Factor: {discount_factor}\nEpsilon: {epsilon_value}\nEpsilon Decay: {epsilon_decay}\n"
    params_text += "-" * 60 + "\n" 

    text_output.insert(tk.END, params_text)
    text_output.tag_add("black", "1.0", "end")  
    text_output.tag_config("black", foreground="black")
    text_output.see(tk.END) 
    button_run.config(state=tk.DISABLED)

    thread = threading.Thread(target=run, args=(is_training, render, learning_rate, discount_factor, epsilon_value, epsilon_decay, text_output, button_run, canvas))
    thread.start()


# Funzione per fermare il programma (Exit)
def stop_program():
    global stop_flag
    stop_flag = True  # Imposta il flag per fermare il ciclo di gioco
    root.quit()  # Chiudi la finestra principale





# Creazione della finestra principale con Tkinter
root = tk.Tk()
root.title("CartPole Q-learning")

root.geometry("900x800")
root.configure(bg="#fffffa")  


font_label = ("Liberation Sans", 12)
font_entry = ("Liberation Sans", 12)

frame_left = tk.Frame(root, bg="#ffffff")
frame_left.pack(side=tk.LEFT, padx=20)

frame_right = tk.Frame(root, bg="#ffffff")
frame_right.pack(side=tk.RIGHT, padx=20)

label_learning_rate = tk.Label(frame_left, text="Learning Rate (alpha)", font=font_label, bg="#ffffff")
label_learning_rate.pack(pady=(5, 5))  
entry_learning_rate = tk.Entry(frame_left, font=font_entry)
entry_learning_rate.insert(0, "0.1")
entry_learning_rate.pack(pady=(0, 5))  


label_discount_factor = tk.Label(frame_left, text="Discount Factor (gamma)", font=font_label, bg="#ffffff")
label_discount_factor.pack()

entry_discount_factor = tk.Entry(frame_left, font=font_entry)
entry_discount_factor.insert(0, "0.99")
entry_discount_factor.pack(pady=(0, 5))

label_epsilon = tk.Label(frame_left, text="Epsilon", font=font_label, bg="#ffffff")
label_epsilon.pack()

entry_epsilon = tk.Entry(frame_left, font=font_entry)
entry_epsilon.insert(0, "1")
entry_epsilon.pack(pady=(0,5))

label_epsilon_decay = tk.Label(frame_left, text="Epsilon Decay Rate", font=font_label, bg="#ffffff")
label_epsilon_decay.pack()

entry_epsilon_decay = tk.Entry(frame_left, font=font_entry)
entry_epsilon_decay.insert(0, "0.00001")
entry_epsilon_decay.pack(pady=(0,5))

#Checkbox 
var_training = tk.IntVar()
var_training.set(1)
checkbox_training = tk.Checkbutton(frame_left, text="Train", font=(font_label, 16), variable=var_training, bg="#ffffff")
checkbox_training.pack()

var_render = tk.IntVar()
var_render.set(0)
checkbox_render = tk.Checkbutton(frame_left, text="Render", font=(font_label, 16), variable=var_render, bg="#ffffff")
checkbox_render.pack()

#Button
button_run = tk.Button(frame_left, text="Start Simulation", background= "#51ed66", font=(font_label, 14), command=start_program)
button_run.pack(pady=10)

button_stop = tk.Button(frame_left, text="Stop Simulation", background= "#ff3838", font=(font_label, 14), command=stop_program)
button_stop.pack(pady=(0,50))

#Canvas 
canvas = tk.Canvas(frame_left, width=300, height=300, bg="#f5f5f5")
canvas.pack()

#TODO fix center text
text_output = tk.Text(frame_right, height=90, width=60, wrap=tk.WORD, font=("Courier", 10))
text_output.tag_configure("center", justify="center")
text_output.tag_add("center", "1.0", "end")
text_output.pack(pady=10)


root.mainloop()
