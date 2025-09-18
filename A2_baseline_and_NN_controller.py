# Third-party libraries

import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# Keep track of data / history
HISTORY = []

def controller (model, data, to_track) :
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Simple 3-layer neural network
    input_size = len (data. qpos)

    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    w1 = np.random.randn(hidden_size, output_size) * 0.1
    w2 = np.random.randn(hidden_size, hidden_size) * 0.1
    w3 = np.random.randn(hidden_size, output_size) * 0.1

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data. qpos
    # Run the inputs through the lays of the network.
    layer1 = sigmoid(np. dot(inputs, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    outputs = sigmoid(np.dot(layer2, w3))

    # Scale outputs to [-pi/2, pi/2]
    data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY. append (to_track[0].xpos.copy())


def show_qpos_history(history:list):
    """
    Parameters:
        history : list
    Return:
        Plot
    """
    # Convert the list of [x,y,z] positions to a numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')

    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3) # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show()


def main():
    """
    Main function to run the simulation with random movements.
    """

    # Initialise the 'controller to controller' movement to None, always in the beginning.

    mujoco.set_mjcb_control(None) # DO NOT REMOVE

    # Initialise world and Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body. Important: YOU MUST USE THE GECKO BODY
    gecko_core = gecko() # DO NOT CHANGE

    # Spawn robot in the world. Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[float(0), float(0), float(0)])

    # Generate the model and data. These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step, YOU DO NOT NEED TO TOUCH IT

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]


    # Set the control callback function
    # This is called every time step to get the next action.
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track))

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch( model=model,
                   # type: ignore
                   data=data,
                   )

    show_qpos_history(HISTORY)



    """
    If you want to record a video of your simulation, you can use the video renderer.
        PATH_TO_VIDEO_FOLDER = "./__videos__"
        video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

        # Render with video recorder
        video_renderer( model,
                        data,
                        duration=30,
                        video_recorder=video_recorder,
                        )
    """


if __name__ == "__main__":

    main()
